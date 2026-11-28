"""Estadísticos de varianza de muestreo, compartidos por los analizadores.

Pregunta que responden: si se envía **el mismo prompt** ``R`` veces cambiando
solo la semilla, ¿cuánto cambia la respuesta y cuánto hace falta para que la
métrica agregada sea estable?

Este módulo no sabe de datasets ni de tareas: trabaja sobre registros con
``window_id_opaque``, ``repetition``, ``status``, ``decision`` y ``confidence``,
y recibe una función de métricas (``metric_fn``) que traduce una lista de
registros votados a ``{accuracy, balanced_accuracy, ...}``. Así lo usan el
barrido de WESAD, DREAMT y PhysioNet sin duplicar la estadística.

Qué mide, por condición:

1. **Estabilidad por ventana** — proporción modal, entropía y número de
   decisiones distintas.
2. **Dispersión de la confianza** — desviación típica intra-ventana y valores
   distintos. Un modelo que solo emite dos valores de confianza no puede ordenar
   ventanas; promediar repeticiones crea resolución aunque la decisión no cambie.
3. **Variabilidad de la métrica con una pasada** — su desviación típica es el
   error de muestreo de un experimento de una pasada.
4. **Convergencia por votación mayoritaria** — ``k = 1..R`` repeticiones
   agregadas por voto (empate → ``abstain``).
5. **Extrapolación** — ``k`` necesario para bajar el error de muestreo de 1, 2 o
   3 puntos, con la ley ``sd_k ≈ sd_1 / sqrt(k)``.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np

#: Estados que aportan una decisión (un ``timeout`` o un ``invalid_output`` no).
USABLE_STATUSES: tuple[str, ...] = ("valid_response", "valid_abstention")
#: Umbrales de error de muestreo (en fracción) para extrapolar ``k``.
DEFAULT_TARGETS: tuple[float, ...] = (0.01, 0.02, 0.03)

#: ``metric_fn(voted, labels) -> {nombre: valor}``.
MetricFn = Callable[[Sequence[Mapping[str, Any]], Mapping[str, str]], dict[str, float]]


def vote(decisions: Sequence[str]) -> str:
    """Voto mayoritario; empate → ``abstain`` (criterio conservador)."""
    counts = Counter(decisions)
    top = max(counts.values())
    winners = [decision for decision, count in counts.items() if count == top]
    return winners[0] if len(winners) == 1 else "abstain"


def window_stability(
    records: Sequence[Mapping[str, Any]],
    id_field: str = "window_id_opaque",
) -> dict[str, Any]:
    """Estadísticos de una ventana a lo largo de sus repeticiones."""
    usable = [r for r in records if r.get("status") in USABLE_STATUSES]
    decisions = [str(r.get("decision")) for r in usable]
    confidences = [
        float(r["confidence"]) for r in usable if r.get("confidence") is not None
    ]
    counts = Counter(decisions)
    n = len(decisions)
    modal_share = max(counts.values()) / n if n else 0.0
    entropy = -sum(
        (count / n) * math.log2(count / n) for count in counts.values()
    ) if n else 0.0
    return {
        "window_id_opaque": (
            records[0].get(id_field) or records[0].get("window_id_opaque", "")
        ),
        "n": len(records),
        "n_usable": n,
        "n_invalid": len(records) - n,
        "decisions": dict(counts),
        "modal_decision": vote(decisions) if decisions else "",
        "modal_share": modal_share,
        "entropy_bits": entropy,
        "n_distinct_decisions": len(counts),
        "confidence_mean": float(np.mean(confidences)) if confidences else None,
        "confidence_std": float(np.std(confidences)) if confidences else None,
        "confidence_min": float(np.min(confidences)) if confidences else None,
        "confidence_max": float(np.max(confidences)) if confidences else None,
        "confidence_n_distinct": len(set(confidences)),
        "abstain_share": (
            sum(1 for d in decisions if d == "abstain") / n if n else 0.0
        ),
    }


def summarize_stability(
    stabilities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Agrega la estabilidad por ventana de una condición."""
    if not stabilities:
        return {}
    conf_stds = [
        s["confidence_std"] for s in stabilities if s.get("confidence_std") is not None
    ]
    return {
        "n_windows": len(stabilities),
        "mean_modal_share": float(np.mean([s["modal_share"] for s in stabilities])),
        "frac_deterministic": float(
            np.mean([1.0 if s["modal_share"] == 1.0 else 0.0 for s in stabilities])
        ),
        "frac_single_decision": float(
            np.mean([1.0 if s["n_distinct_decisions"] == 1 else 0.0 for s in stabilities])
        ),
        "mean_entropy_bits": float(
            np.mean([s["entropy_bits"] for s in stabilities])
        ),
        "mean_confidence_std": float(np.mean(conf_stds)) if conf_stds else 0.0,
        "frac_confidence_constant": float(
            np.mean([1.0 if s["confidence_std"] == 0.0 else 0.0 for s in stabilities])
        ),
        "mean_abstain_share": float(
            np.mean([s["abstain_share"] for s in stabilities])
        ),
    }


def aggregate_window(
    records: Sequence[Mapping[str, Any]],
    reps: Sequence[int],
    id_field: str = "window_id_opaque",
) -> dict[str, Any] | None:
    """Voto mayoritario de las repeticiones indicadas y confianza media.

    La confianza agregada es la media de la confianza de las repeticiones que
    votaron la decisión ganadora: promediar réplicas da resolución a un modelo
    que solo emite dos valores distintos.
    """
    by_rep = {int(r.get("repetition", 0) or 0): r for r in records}
    selected = [by_rep[rep] for rep in reps if rep in by_rep]
    usable = [r for r in selected if r.get("status") in USABLE_STATUSES]
    if not usable:
        return None
    decisions = [str(r.get("decision")) for r in usable]
    voted = vote(decisions)
    if voted == "abstain":
        confidence = 0.0
        status = "valid_abstention"
    else:
        agreeing = [
            float(r["confidence"])
            for r in usable
            if r.get("decision") == voted and r.get("confidence") is not None
        ]
        confidence = float(np.mean(agreeing)) if agreeing else 0.0
        status = "valid_response"
    return {
        "window_id_opaque": (
            usable[0].get(id_field) or usable[0].get("window_id_opaque", "")
        ),
        "decision": voted,
        "confidence": confidence,
        "status": status,
    }


def subsets(
    n: int,
    k: int,
    rng: np.random.Generator,
    max_combos: int,
) -> list[tuple[int, ...]]:
    """Combinaciones de tamaño ``k`` (todas si son pocas; si no, aleatorias)."""
    if k < 1 or k > n:
        return []
    total = math.comb(n, k)
    if total <= max_combos:
        return list(combinations(range(n), k))
    seen: set[tuple[int, ...]] = set()
    while len(seen) < max_combos:
        draw = tuple(sorted(int(i) for i in rng.choice(n, size=k, replace=False)))
        seen.add(draw)
    return sorted(seen)


def convergence(
    windows: Mapping[str, Sequence[Mapping[str, Any]]],
    metric_fn: MetricFn,
    reps: Sequence[int],
    rng: np.random.Generator,
    max_combos: int = 400,
) -> list[dict[str, Any]]:
    """Métricas por número de repeticiones agregadas (``k = 1..R``)."""
    rows: list[dict[str, Any]] = []
    for k in range(1, len(reps) + 1):
        acc: list[float] = []
        bal: list[float] = []
        cov: list[float] = []
        abst: list[float] = []
        for subset in subsets(len(reps), k, rng, max_combos):
            chosen = [reps[i] for i in subset]
            voted = [
                aggregated
                for records in windows.values()
                if (aggregated := aggregate_window(records, chosen)) is not None
            ]
            if not voted:
                continue
            metrics = metric_fn(voted, {})
            acc.append(metrics["accuracy"])
            bal.append(metrics["balanced_accuracy"])
            cov.append(metrics["coverage"])
            abst.append(metrics["abstention_rate"])
        if not acc:
            continue
        rows.append(
            {
                "k": k,
                "n_combinations": len(acc),
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_sd": float(np.std(acc)),
                "balanced_mean": float(np.nanmean(bal)),
                "balanced_sd": float(np.nanstd(bal)),
                "coverage_mean": float(np.mean(cov)),
                "abstention_mean": float(np.mean(abst)),
            }
        )
    return rows


def extrapolate(
    sd1: float,
    targets: Sequence[float] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    """``k`` para bajar el error de muestreo de ``sd1`` a cada umbral.

    Supone independencia entre repeticiones y ``sd_k ≈ sd_1 / sqrt(k)``; es una
    cota optimista (si el modelo es determinista, ``sd1 = 0`` y no hace falta
    ninguna repetición).
    """
    if not math.isfinite(sd1) or sd1 <= 0:
        return {
            "sd_single": float(sd1) if math.isfinite(sd1) else None,
            "implied_k": {f"{pct}pp": 1 for pct in (1, 2, 3)},
            "note": "sin varianza de muestreo: las repeticiones no cambiarían la métrica",
        }
    return {
        "sd_single": float(sd1),
        "implied_k": {
            f"{int(round(target * 100))}pp": int(math.ceil((sd1 / target) ** 2))
            for target in targets
        },
    }


def format_metric(value: Any, width: int = 0, spec: str = ".3f") -> str:
    """Formatea una métrica para consola: ``0.0`` se respeta, ``nan`` es ``—``.

    Existe porque el idioma ``valor or nan`` convierte un **0.0 legítimo** en
    indefinido: la cobertura de un brazo que se abstiene al 100 % (texto `C1` de
    DREAMT, 2026-11-28) se imprimía como ``nan`` cuando vale exactamente 0, y el
    número que sostiene la conclusión es precisamente ese 0.

    Args:
        value: métrica (``None``, ``nan`` o número).
        width: ancho mínimo del campo (0 = sin relleno).
        spec: formato de ``format()`` para el número (p. ej. ``".3f"``).

    Returns:
        str: celda lista para imprimir.
    """
    if value is None:
        text = "—"
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = "—"
        else:
            text = "—" if not math.isfinite(number) else format(number, spec)
    return f"{text:>{width}}" if width else text


def distribution(values: Sequence[float]) -> dict[str, Any]:
    """Media, desviación típica y valores de una muestra."""
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"mean": None, "sd": None, "values": []}
    return {
        "mean": float(np.mean(clean)),
        "sd": float(np.std(clean)),
        "values": [float(v) for v in clean],
    }
