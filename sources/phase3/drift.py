"""Deriva de la decisión bajo degradación: hacia dónde se mueve la respuesta.

Pregunta que responde: cuando la evidencia se degrada, **¿hacia qué clase se
desplaza** el modelo, **qué parte del error induce la degradación** (frente a la
que ya existía en limpio) y **qué señales acompañan** al desplazamiento?

Los informes de degradación (`analyze_degradation.py`) miden cada condición por
separado: dicen que en ``C4`` la accuracy cae y que se predice más ``stress``,
pero no dicen *qué ventanas* cambian ni *de dónde vienen*. Este módulo añade el
nivel de ventana:

1. **Marginal predicha** frente al **prior real** y **tasa de reclamación**
   (``P(predicha) / P(real)``), que es la forma correcta de leer «el modelo
   sobrerreclama estrés» (no la desviación respecto al equilibrio: en un diseño
   con prior 0,30 un modelo perfecto no es 50/50).
2. **Matriz pareada** ``C0 → Cx`` por ventana, con la **descomposición 2×2**
   (acierta/falla antes × acierta/falla después). Separa el **error inducido**
   (``correct_wrong``) del **error estable** (``wrong_wrong``).
3. **Destino del error** por clase real: a qué clase van las ventanas que
   aciertan en limpio y fallan al degradar.
4. **Flujos de abstención**: cuántas ventanas que se abstenían en limpio pasan a
   decidir (y a qué clase) y cuántas deciden en limpio y se abstienen después.
5. **Control con la referencia supervisada**: las mismas tablas sobre las
   predicciones del modelo features-only (``table_from_predictions``). Si la
   referencia no deriva y el LLM sí, la deriva es de la **política de decisión**,
   no de la información disponible.

El módulo es puro (sin LLM, sin sklearn): recibe registros y etiquetas y
devuelve tablas serializables. El script `scripts/analyze_drift.py` se encarga
de la lectura de artefactos, la referencia y los intervalos por sujeto.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from sources.phase2.response_schema import (
    DECISION_ALIASES,
    DECISION_MODE_AFFECTIVE,
    DECISION_MODE_STRESS_BINARY,
)

#: Espacio de decisión de cada tarea (sin contar ``abstain``).
DECISION_SPACES: dict[str, tuple[str, ...]] = {
    DECISION_MODE_AFFECTIVE: ("baseline", "stress", "amusement", "meditation"),
    DECISION_MODE_STRESS_BINARY: ("stress", "non-stress"),
}

#: Estados en los que la respuesta contiene una decisión del modelo.
USABLE_STATUSES: tuple[str, ...] = ("valid_response", "valid_abstention")

#: La compuerta de consistencia decide por regla, no por el modelo. Por defecto
#: se excluye para medir la política del LLM; ``include_gate=True`` la incluye
#: como un ``abstain`` más (sistema compuesto).
GATE_STATUS = "gate_abstention"

DECISION_MODE_ALIASES = {
    "binary": DECISION_MODE_STRESS_BINARY,
    "affective": DECISION_MODE_AFFECTIVE,
    "4class": DECISION_MODE_AFFECTIVE,
    "four_class": DECISION_MODE_AFFECTIVE,
}

ABSTAIN = "abstain"


# ---------------------------------------------------------------------------
# Espacios de decisión
# ---------------------------------------------------------------------------
def resolve_decision_mode(
    value: str | None, default: str = DECISION_MODE_STRESS_BINARY
) -> str:
    """Normaliza el nombre de la tarea (acepta el alias ``binary``)."""
    if not value:
        return default
    mode = DECISION_MODE_ALIASES.get(str(value).strip().lower(), str(value).strip())
    if mode not in DECISION_SPACES:
        raise ValueError(
            f"decision_mode desconocido: {value!r}. "
            f"Disponibles: {sorted(DECISION_SPACES)} (alias: binary, 4class)"
        )
    return mode


def decision_classes(decision_mode: str) -> tuple[str, ...]:
    """Clases del espacio de decisión (sin ``abstain``)."""
    return DECISION_SPACES[resolve_decision_mode(decision_mode)]


def normalize_decision(value: Any) -> str | None:
    """Decisión en el vocabulario canónico, o ``None`` si no es reconocible."""
    if value is None:
        return None
    decision = str(value).strip().lower()
    decision = DECISION_ALIASES.get(decision, decision)
    if decision in (ABSTAIN, "abstention", "no-decision", "no_decision"):
        return ABSTAIN
    for classes in DECISION_SPACES.values():
        if decision in classes:
            return decision
    return None


def truth_in_space(
    label: str | None,
    decision_mode: str,
    nonstress_classes: Iterable[str] | None = None,
) -> str | None:
    """Etiqueta real traducida al espacio de decisión.

    En binario, cualquier etiqueta que no sea ``stress`` se convierte en
    ``non-stress`` **solo si pertenece al conjunto negativo de la variante**.
    ``meditation`` con la variante ``schmidt`` devuelve ``None``: no forma parte
    del endpoint y no debe entrar en las tablas de deriva.
    """
    if not label:
        return None
    mode = resolve_decision_mode(decision_mode)
    label = str(label).strip().lower()
    if mode == DECISION_MODE_AFFECTIVE:
        return label if label in DECISION_SPACES[mode] else None
    allowed = set(nonstress_classes or ("baseline", "amusement")) | {"stress"}
    if label not in allowed:
        return None
    return "stress" if label == "stress" else "non-stress"


# ---------------------------------------------------------------------------
# Tabla de decisiones (una por ventana y condición)
# ---------------------------------------------------------------------------
def _vote(decisions: Sequence[str]) -> str:
    """Decisión mayoritaria; empate → ``abstain`` (criterio conservador)."""
    counts = Counter(decisions)
    top = max(counts.values())
    winners = [d for d, n in counts.items() if n == top]
    return winners[0] if len(winners) == 1 else ABSTAIN


def build_decision_table(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    *,
    decision_mode: str = DECISION_MODE_STRESS_BINARY,
    nonstress_classes: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    include_gate: bool = False,
) -> dict[str, Any]:
    """Construye ``{ventana: {verdad, decisiones por condición}}``.

    Si un corpus tiene repeticiones (varias filas por ventana y condición), se
    agrega por voto mayoritario y se guarda ``n_reps``: la deriva se analiza
    sobre la respuesta agregada, y la varianza entre repeticiones es otro
    experimento (``analyze_repetitions.py``).

    Las ventanas fuera de la variante (``meditation`` en ``schmidt``), sin
    etiqueta o con estado no usable se descartan y se cuentan en ``dropped``.
    """
    mode = resolve_decision_mode(decision_mode)
    allowed = set(conditions) if conditions else None
    keep_statuses = set(USABLE_STATUSES) | ({GATE_STATUS} if include_gate else set())
    dropped: Counter[str] = Counter()
    grouped: dict[str, dict[str, list[tuple[str, float | None]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    subject_of: dict[str, str] = {}

    for record in records:
        window_id = record.get("window_id_opaque")
        condition = record.get("condition_id")
        if not window_id or not condition:
            dropped["sin_identificador"] += 1
            continue
        if allowed is not None and condition not in allowed:
            continue
        if record.get("status") not in keep_statuses:
            dropped["estado_no_usable"] += 1
            continue
        decision = normalize_decision(record.get("decision"))
        if decision is None:
            dropped["decision_desconocida"] += 1
            continue
        truth = truth_in_space(labels.get(window_id), mode, nonstress_classes)
        if truth is None:
            dropped["fuera_de_la_variante_o_sin_etiqueta"] += 1
            continue
        confidence = record.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        grouped[window_id][condition].append((decision, confidence))
        subject_of[window_id] = str(record.get("subject") or "")

    windows: dict[str, dict[str, Any]] = {}
    for window_id, by_condition in grouped.items():
        decisions: dict[str, str] = {}
        confidences: dict[str, float | None] = {}
        n_reps: dict[str, int] = {}
        for condition, entries in by_condition.items():
            decisions[condition] = _vote([entry[0] for entry in entries])
            values = [entry[1] for entry in entries if entry[1] is not None]
            confidences[condition] = (
                float(sum(values) / len(values)) if values else None
            )
            n_reps[condition] = len(entries)
        windows[window_id] = {
            "truth": truth_in_space(labels.get(window_id), mode, nonstress_classes),
            "subject": subject_of.get(window_id, ""),
            "decisions": decisions,
            "confidence": confidences,
            "n_reps": n_reps,
        }

    return {
        "decision_mode": mode,
        "classes": list(DECISION_SPACES[mode]),
        "nonstress_classes": (
            sorted(set(nonstress_classes)) if nonstress_classes else None
        ),
        "include_gate": bool(include_gate),
        "windows": windows,
        "dropped": dict(dropped),
        "n_windows": len(windows),
    }


def table_from_predictions(
    predictions: Mapping[str, Mapping[str, str]],
    truth_by_window: Mapping[str, str],
    *,
    decision_mode: str = DECISION_MODE_STRESS_BINARY,
    subject_by_window: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Tabla equivalente para una referencia supervisada (``{cond: {wid: pred}}``).

    Permite aplicar **las mismas** tablas de deriva a las predicciones
    features-only y comparar la política del LLM con la de un modelo que ve
    exactamente la misma evidencia.
    """
    mode = resolve_decision_mode(decision_mode)
    windows: dict[str, dict[str, Any]] = {}
    window_ids: set[str] = set()
    for by_window in predictions.values():
        window_ids.update(by_window)
    for window_id in sorted(window_ids):
        truth = truth_by_window.get(window_id)
        if truth is None:
            continue
        decisions = {
            condition: by_window[window_id]
            for condition, by_window in predictions.items()
            if window_id in by_window
        }
        windows[window_id] = {
            "truth": truth,
            "subject": (subject_by_window or {}).get(window_id, ""),
            "decisions": decisions,
            "confidence": {condition: None for condition in decisions},
            "n_reps": {condition: 1 for condition in decisions},
        }
    return {
        "decision_mode": mode,
        "classes": list(DECISION_SPACES[mode]),
        "nonstress_classes": None,
        "include_gate": False,
        "windows": windows,
        "dropped": {},
        "n_windows": len(windows),
    }


# ---------------------------------------------------------------------------
# Marginal predicha y tasa de reclamación
# ---------------------------------------------------------------------------
def _rate(count: int, total: int) -> float | None:
    return (count / total) if total else None


def prediction_marginal(
    table: Mapping[str, Any],
    conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Distribución predicha por condición frente al prior real.

    - ``predicted_decided``: reparto entre las clases, sin ``abstain``.
    - ``predicted_all``: incluye ``abstain`` como categoría.
    - ``claim_rate``: ``P(predicha) / P(real)`` por clase. Es la lectura
      correcta del sesgo: ``>1`` sobrerreclama, ``<1`` infra-reclama. Se calcula
      sobre las ventanas decididas y con prior > 0; si no, ``None``.
    """
    windows = table["windows"]
    classes = tuple(table["classes"])
    conditions = list(conditions or sorted(_conditions_of(windows)))
    report: dict[str, Any] = {"decision_space": list(classes), "conditions": {}}
    for condition in conditions:
        rows = [w for w in windows.values() if condition in w["decisions"]]
        if not rows:
            continue
        decisions = [w["decisions"][condition] for w in rows]
        decided = [d for d in decisions if d != ABSTAIN]
        predicted = Counter(decided)
        truth = Counter(w["truth"] for w in rows if w.get("truth"))
        n_decided = len(decided)
        n_truth = sum(truth.values())
        claim_rate: dict[str, float | None] = {}
        for klass in classes:
            pred_share = _rate(predicted.get(klass, 0), n_decided)
            real_share = _rate(truth.get(klass, 0), n_truth)
            if pred_share is None or real_share in (None, 0.0):
                claim_rate[klass] = None
            else:
                claim_rate[klass] = pred_share / real_share
        report["conditions"][condition] = {
            "n": len(rows),
            "n_decided": n_decided,
            "abstain_share": _rate(len(decisions) - n_decided, len(decisions)),
            "predicted_decided": {
                k: _rate(v, n_decided) for k, v in sorted(predicted.items())
            },
            "predicted_all": {
                k: _rate(decisions.count(k), len(decisions))
                for k in [*classes, ABSTAIN]
                if decisions.count(k)
            },
            "truth": {k: _rate(v, n_truth) for k, v in sorted(truth.items())},
            "claim_rate": claim_rate,
        }
    return report


def _conditions_of(windows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    conditions: set[str] = set()
    for window in windows.values():
        conditions.update(window["decisions"])
    return conditions


# ---------------------------------------------------------------------------
# Matriz de transición pareada
# ---------------------------------------------------------------------------
def _decomposition(
    pairs: Sequence[tuple[Mapping[str, Any], str, str]],
) -> dict[str, Any]:
    """Acierta/falla antes × acierta/falla después (solo pares decididos)."""
    counts = {
        "correct_correct": 0,
        "correct_wrong": 0,
        "wrong_correct": 0,
        "wrong_wrong": 0,
    }
    for window, before, after in pairs:
        truth = window.get("truth")
        if truth is None:
            continue
        correct_before = before == truth
        correct_after = after == truth
        key = (
            "correct" if correct_before else "wrong"
        ) + "_" + ("correct" if correct_after else "wrong")
        counts[key] += 1
    total = sum(counts.values())
    wrong_after = counts["correct_wrong"] + counts["wrong_wrong"]
    wrong_before = counts["wrong_correct"] + counts["wrong_wrong"]
    return {
        **counts,
        "n_with_truth": total,
        "accuracy_before": _rate(
            counts["correct_correct"] + counts["correct_wrong"], total
        ),
        "accuracy_after": _rate(
            counts["correct_correct"] + counts["wrong_correct"], total
        ),
        # Cuánto del error final lo ha inducido la degradación.
        "induced_error_share": _rate(counts["correct_wrong"], wrong_after),
        # Cuánto del error inicial se ha reparado.
        "repaired_share": _rate(counts["wrong_correct"], wrong_before),
        # Cuánto del error final ya existía en limpio.
        "preexisting_error_share": _rate(counts["wrong_wrong"], wrong_after),
    }


def _by_truth_class(
    pairs: Sequence[tuple[Mapping[str, Any], str, str]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Por clase real: destino de las ventanas que fallan tras la degradación."""
    block: dict[str, Any] = {}
    for klass in classes:
        rows = [(w, b, a) for w, b, a in pairs if w.get("truth") == klass]
        if not rows:
            continue
        errors = [(w, b, a) for w, b, a in rows if a != klass]
        drifted = [(w, b, a) for w, b, a in errors if b == klass]
        block[klass] = {
            "n": len(rows),
            "n_errors": len(errors),
            "error_share": _rate(len(errors), len(rows)),
            "destination_of_errors": dict(
                Counter(a for _, _, a in errors).most_common()
            ),
            "drifted_from_correct": len(drifted),
            "drifted_destination": dict(
                Counter(a for _, _, a in drifted).most_common()
            ),
        }
    return block


def _by_subject(
    pairs: Sequence[tuple[Mapping[str, Any], str, str]],
) -> dict[str, dict[str, int]]:
    """Recuentos por sujeto, listos para un bootstrap agrupado.

    El sujeto es la unidad de generalización: una tasa sobre 424 ventanas de 12
    personas no es un intervalo de 424 observaciones. El script remuestrea estas
    filas para obtener el IC de ``induced_error_share``.
    """
    block: dict[str, dict[str, int]] = {}
    for window, before, after in pairs:
        subject = window.get("subject") or "?"
        bucket = block.setdefault(
            subject,
            {
                "n_pairs": 0,
                "n_changed": 0,
                "n_correct_before": 0,
                "n_correct_after": 0,
                "n_correct_wrong": 0,
                "n_wrong_wrong": 0,
            },
        )
        bucket["n_pairs"] += 1
        if before != after:
            bucket["n_changed"] += 1
        truth = window.get("truth")
        if truth is None:
            continue
        correct_before = before == truth
        correct_after = after == truth
        bucket["n_correct_before"] += int(correct_before)
        bucket["n_correct_after"] += int(correct_after)
        if correct_before and not correct_after:
            bucket["n_correct_wrong"] += 1
        elif not correct_before and not correct_after:
            bucket["n_wrong_wrong"] += 1
    return dict(sorted(block.items()))


def transition_analysis(
    table: Mapping[str, Any],
    *,
    base: str = "C0_clean",
    conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Tablas pareadas ``base → condición`` por ventana.

    Devuelve por condición:

    - ``matrix_all``: matriz completa (incluye ``abstain`` como estado).
    - ``matrix_decided``: solo pares decididos en ambas.
    - ``decomposition``: acierta/falla antes × después, ``induced_error_share``
      (parte del error que induce la degradación) y
      ``preexisting_error_share`` (parte que ya estaba).
    - ``by_truth``: por clase real, a qué clase van los errores y, entre ellos,
      los que aciertan en limpio (la deriva inducida de verdad).
    - ``abstain_flow``: decidido→abstain, abstain→decidido, abstain→abstain.
    """
    windows = table["windows"]
    classes = tuple(table["classes"])
    available = _conditions_of(windows)
    conditions = [c for c in (conditions or sorted(available)) if c != base]
    report: dict[str, Any] = {
        "base": base,
        "decision_space": list(classes),
        "conditions": {},
        "missing_base": sum(
            1 for w in windows.values() if base not in w["decisions"]
        ),
    }
    for condition in conditions:
        if condition not in available:
            continue
        both = [
            (w, w["decisions"][base], w["decisions"][condition])
            for w in windows.values()
            if base in w["decisions"] and condition in w["decisions"]
        ]
        if not both:
            continue
        matrix_all = Counter((before, after) for _, before, after in both)
        decided = [(w, b, a) for w, b, a in both if b != ABSTAIN and a != ABSTAIN]
        matrix_decided = Counter((b, a) for _, b, a in decided)
        changed = [
            ((before, after), count)
            for (before, after), count in matrix_decided.items()
            if before != after
        ]
        matrix_all_block: dict[str, dict[str, int]] = {}
        for (before, after), count in matrix_all.items():
            matrix_all_block.setdefault(before, {})[after] = count
        matrix_decided_block: dict[str, dict[str, int]] = {}
        for (before, after), count in matrix_decided.items():
            matrix_decided_block.setdefault(before, {})[after] = count
        abstain_flow = Counter()
        for _, before, after in both:
            if before == ABSTAIN and after == ABSTAIN:
                abstain_flow["abstain_to_abstain"] += 1
            elif before == ABSTAIN:
                abstain_flow[f"abstain_to_{after}"] += 1
            elif after == ABSTAIN:
                abstain_flow[f"{before}_to_abstain"] += 1
        report["conditions"][condition] = {
            "n_windows_both": len(both),
            "n_decided_pairs": len(decided),
            "matrix_all": matrix_all_block,
            "matrix_decided": matrix_decided_block,
            "n_changed": sum(n for _, n in changed),
            "changed_share": _rate(sum(n for _, n in changed), len(decided)),
            "direction": [
                {"from": before, "to": after, "n": n}
                for (before, after), n in sorted(
                    changed, key=lambda item: (-item[1], item[0])
                )
            ],
            "decomposition": _decomposition(decided),
            "by_truth": _by_truth_class(decided, classes),
            "by_subject": _by_subject(decided),
            "abstain_flow": dict(sorted(abstain_flow.items())),
        }
    return report


def window_groups(
    table: Mapping[str, Any],
    *,
    base: str = "C0_clean",
    condition: str,
) -> dict[str, list[str]]:
    """Identificadores de ventana agrupados por el efecto de la degradación.

    Grupos: ``drifted`` (cambia de decisión), ``stable_correct``,
    ``stable_wrong``, ``abstained_before`` y ``abstained_after``. Los dos
    primeros exigen decisión en ambas condiciones; los de abstención no.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for window_id, window in table["windows"].items():
        decisions = window["decisions"]
        if base not in decisions or condition not in decisions:
            continue
        before, after = decisions[base], decisions[condition]
        if after == ABSTAIN:
            groups["abstained_after"].append(window_id)
            continue
        if before == ABSTAIN:
            groups["abstained_before"].append(window_id)
            continue
        if before != after:
            groups["drifted"].append(window_id)
        elif window.get("truth") == after:
            groups["stable_correct"].append(window_id)
        else:
            groups["stable_wrong"].append(window_id)
    return {key: sorted(value) for key, value in groups.items()}


# ---------------------------------------------------------------------------
# Coherencia de claims en las ventanas que derivan
# ---------------------------------------------------------------------------
def coherence_summary(
    blocks: Sequence[Mapping[str, Any]],
    *,
    min_responses: int = 5,
) -> dict[str, Any]:
    """Agrega bloques de :func:`sources.phase3.claims.response_coherence`.

    Responde a: en las ventanas que **derivan**, ¿el modelo cita evidencia que
    apoya la nueva clase o que la contradice? Un claim verdadero y contrario a
    la decisión cuenta en ``against_supported_share``.
    """
    if not blocks:
        return {
            "n_responses": 0,
            "n_with_signature": 0,
            "insufficient": True,
            "min_responses": int(min_responses),
        }
    usable = [b for b in blocks if (b.get("n_signature_claims") or 0) > 0]
    n = len(blocks)
    n_confirming = sum(int(b.get("n_confirming") or 0) for b in blocks)
    n_against = sum(int(b.get("n_against") or 0) for b in blocks)
    n_supported = sum(int(b.get("n_supported") or 0) for b in blocks)
    n_against_supported = sum(
        int(b.get("n_against_supported") or 0) for b in blocks
    )
    decided = n_confirming + n_against
    return {
        "n_responses": n,
        "n_with_signature": len(usable),
        "insufficient": len(usable) < int(min_responses),
        "min_responses": int(min_responses),
        "zero_confirmation_share": (
            _rate(
                sum(1 for b in usable if not b.get("has_confirmation")),
                len(usable),
            )
            if usable
            else None
        ),
        "evidence_conflict_rate": _rate(
            sum(1 for b in blocks if b.get("evidence_conflict")), n
        ),
        "coherence_rate": _rate(n_confirming, decided),
        "against_supported_share": _rate(n_against_supported, n_supported),
        "n_claims_with_signature": decided,
        "n_supported": n_supported,
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción (stdlib, sin scipy)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
