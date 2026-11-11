#!/usr/bin/env python
"""P3 — Análisis del control de generalidad PhysioNet/CinC 2017.

Relee las respuestas y las etiquetas de ``results/output/PHYSIONET2017/P3`` y
produce un informe con:

1. **Corrección** del LLM por clase, confusión y comparación con la referencia
   features-only (regresión logística entrenada en desarrollo), incluyendo el
   **exceso de coste** y la línea base trivial (clase mayoritaria).
2. **Abstención** sobre la clase ``noisy`` (evidencia no interpretable según el
   experto) con intervalos de Wilson, y abstención por clase.
3. **Respaldo de claims** (H5) y **certeza lingüística** (H2b) por clase.
4. **Señal de calidad en la entrada** (H6): AUROC univariante de cada feature
   para ``noisy`` vs resto, detector multivariante entrenado en desarrollo y
   curva de compuerta (qué cobertura se pierde para detectar ruido).
5. **Consistencia interna** (compuerta de P1) y su transferibilidad a este
   dataset.
6. **Ablación few-shot** en desarrollo (mismos 40 registros, con y sin
   ejemplos).

No hace llamadas al LLM: todo sale de artefactos en disco.

Uso:
    python scripts/analyze_p3_physionet.py
    python scripts/analyze_p3_physionet.py --output /tmp/opencode/pn17.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.common.provenance import utc_now
from sources.external.physionet2017 import (
    CANONICAL_FEATURES,
    RHYTHM_CLASSES,
    RhythmRecord,
    population_reference,
    z_scores,
)
from sources.phase2.consistency import check_consistency, should_abstain
from sources.phase2.response_schema import DECISION_MODE_RHYTHM4
from sources.phase3.calibration import wilson_interval
from sources.phase3.certainty import linguistic_certainty
from sources.phase3.claims import summarize_verdicts, verify_claims
from sources.phase3.recovery import apply_recovery

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "results" / "output" / "PHYSIONET2017" / "P3"
DEFAULT_OUTPUT = DEFAULT_DIR / "physionet2017_analysis.json"

VALID_STATUSES = ("valid_response", "valid_abstention")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Carga un JSONL completo (ignora líneas vacías o truncadas)."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dedupe_responses(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Una respuesta por registro (gana la última, como en la reanudación)."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[row["record_id"]] = dict(row)
    return list(latest.values())


def load_labels(path: Path) -> dict[str, str]:
    """``record_id -> etiqueta de experto``."""
    return {row["record_id"]: row["label"] for row in load_jsonl(path)}


def rhythm_records(
    cache: Mapping[str, Mapping[str, Any]],
    record_ids: Sequence[str],
    labels: Mapping[str, str],
) -> list[RhythmRecord]:
    """Reconstruye objetos ``RhythmRecord`` desde la caché de features."""
    records: list[RhythmRecord] = []
    for record_id in record_ids:
        entry = cache[record_id]
        records.append(
            RhythmRecord(
                record_id=record_id,
                label=labels[record_id],
                fs=float(entry["fs"]),
                duration_s=float(entry["duration_s"]),
                values=dict(entry["values"]),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Métricas del LLM
# ---------------------------------------------------------------------------
def claims_vs_correctness(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """¿Predice el respaldo de claims la corrección de la decisión? (H5)"""
    correct: list[float] = []
    incorrect: list[float] = []
    for record in rows:
        if record["status"] not in VALID_STATUSES or not record.get("claims"):
            continue
        if record["decision"] == "abstain":
            continue
        verdicts = verify_claims(record["claims"], record["prompt"])
        evaluable = [v for v in verdicts if v.direction_ok is not None]
        if not evaluable:
            continue
        support = sum(1 for v in evaluable if v.category == "supported") / len(evaluable)
        if record["decision"] == labels[record["record_id"]]:
            correct.append(support)
        else:
            incorrect.append(support)
    return {
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "supported_rate_correct": float(np.mean(correct)) if correct else float("nan"),
        "supported_rate_incorrect": float(np.mean(incorrect)) if incorrect else float("nan"),
        "auroc_support_for_correctness": auroc(correct, incorrect),
    }


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """AUROC por la estadística U de Mann-Whitney (empates a 0.5)."""
    if not positive or not negative:
        return float("nan")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def average_precision(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Precisión media (área bajo precision-recall) con positivos = noisy."""
    if not positive or not negative:
        return float("nan")
    scores = [(score, 1) for score in positive] + [(score, 0) for score in negative]
    scores.sort(key=lambda item: item[0], reverse=True)
    hits = 0
    total = 0
    ap = 0.0
    previous_recall = 0.0
    for score, label in scores:
        total += 1
        if label:
            hits += 1
        recall = hits / len(positive)
        precision = hits / total
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def llm_metrics(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Corrección, abstención y confusión del LLM sobre registros etiquetados."""
    valid = [r for r in rows if r["status"] in VALID_STATUSES]
    decided = [r for r in valid if r["decision"] != "abstain"]
    abstained = [r for r in valid if r["decision"] == "abstain"]

    truth = [labels[r["record_id"]] for r in decided]
    predicted = [r["decision"] for r in decided]
    hits = sum(1 for t, p in zip(truth, predicted) if t == p)

    per_class: dict[str, dict[str, Any]] = {}
    for label in RHYTHM_CLASSES:
        in_class = [r for r in valid if labels[r["record_id"]] == label]
        decided_class = [r for r in decided if labels[r["record_id"]] == label]
        class_hits = sum(1 for r in decided_class if r["decision"] == label)
        predicted_as = sum(1 for r in decided if r["decision"] == label)
        per_class[label] = {
            "n": len(in_class),
            "n_decided": len(decided_class),
            "abstentions": len(in_class) - len(decided_class),
            "abstention_rate": (
                (len(in_class) - len(decided_class)) / len(in_class)
                if in_class
                else float("nan")
            ),
            "recall": class_hits / len(decided_class) if decided_class else float("nan"),
            "precision": class_hits / predicted_as if predicted_as else float("nan"),
        }

    recalls = [
        entry["recall"]
        for entry in per_class.values()
        if not np.isnan(entry["recall"])
    ]
    confusion = {
        label: {
            predicted_label: sum(
                1
                for r in valid
                if labels[r["record_id"]] == label
                and r["decision"] == predicted_label
            )
            for predicted_label in list(RHYTHM_CLASSES) + ["abstain"]
        }
        for label in RHYTHM_CLASSES
    }
    invalid = [r for r in rows if r["status"] not in VALID_STATUSES]
    return {
        "n": len(rows),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "n_decided": len(decided),
        "n_abstain": len(abstained),
        "abstention_rate": len(abstained) / len(valid) if valid else float("nan"),
        "accuracy": hits / len(decided) if decided else float("nan"),
        "accuracy_incl_abstain": hits / len(valid) if valid else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "decisions": dict(Counter(r["decision"] for r in valid)),
        "confidence": {str(k): v for k, v in sorted(Counter(r["confidence"] for r in valid).items())},
        "invalid_examples": [
            {
                "record_id": r["record_id"],
                "error_message": r.get("error_message"),
            }
            for r in invalid
        ],
        "per_class": per_class,
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Referencia features-only
# ---------------------------------------------------------------------------
def reference_metrics(
    dev: Sequence[RhythmRecord],
    test: Sequence[RhythmRecord],
    centers: Mapping[str, float],
    scales: Mapping[str, float],
) -> dict[str, Any]:
    """Regresión logística entrenada en desarrollo y evaluada en test."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def matrix(records: Sequence[RhythmRecord]) -> np.ndarray:
        rows = []
        for record in records:
            scores = z_scores(record.values, centers, scales)
            rows.append(
                [
                    float("nan") if scores[name] is None else scores[name]
                    for name in CANONICAL_FEATURES
                ]
            )
        return np.asarray(rows, dtype=float)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(matrix(dev), [r.label for r in dev])
    predictions = model.predict(matrix(test))

    truth = [r.label for r in test]
    hits = sum(1 for t, p in zip(truth, predictions) if t == p)
    per_class = {}
    for label in RHYTHM_CLASSES:
        in_class = [p for t, p in zip(truth, predictions) if t == label]
        per_class[label] = {
            "n": len(in_class),
            "recall": (
                sum(1 for p in in_class if p == label) / len(in_class)
                if in_class
                else float("nan")
            ),
            "precision": (
                sum(1 for p in in_class if p == label)
                / sum(1 for p in predictions if p == label)
                if any(p == label for p in predictions)
                else float("nan")
            ),
        }
    confusion = {
        label: {
            predicted: sum(
                1 for t, p in zip(truth, predictions) if t == label and p == predicted
            )
            for predicted in RHYTHM_CLASSES
        }
        for label in RHYTHM_CLASSES
    }
    recalls = [e["recall"] for e in per_class.values() if not np.isnan(e["recall"])]
    return {
        "n_train": len(dev),
        "n_test": len(test),
        "accuracy": hits / len(test),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Claims y certeza
# ---------------------------------------------------------------------------
def claims_by_class(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Respaldo de claims por clase verdadera y desglose por métrica."""
    grouped: dict[str, list[Any]] = {label: [] for label in RHYTHM_CLASSES}
    global_verdicts: list[Any] = []
    verdict_records: dict[int, str] = {}
    for record in rows:
        if record["status"] not in VALID_STATUSES or not record.get("claims"):
            continue
        verdicts = verify_claims(record["claims"], record["prompt"])
        grouped[labels[record["record_id"]]].extend(verdicts)
        global_verdicts.extend(verdicts)
        for verdict in verdicts:
            verdict_records[id(verdict)] = record["record_id"]

    by_class = {
        label: summarize_verdicts(verdicts) for label, verdicts in grouped.items()
    }

    per_metric: dict[str, Counter[str]] = {}
    for verdict in global_verdicts:
        per_metric.setdefault(verdict.metric, Counter())[verdict.category] += 1
    metric_block = {
        metric: {
            "n": sum(counts.values()),
            **{category: counts.get(category, 0) for category in
               ("supported", "contradicted", "unsupported", "indeterminate")},
        }
        for metric, counts in sorted(per_metric.items())
    }

    contradicted_direction = sum(
        1
        for v in global_verdicts
        if v.category == "contradicted" and v.direction_ok is False
    )
    contradicted_magnitude = sum(
        1
        for v in global_verdicts
        if v.category == "contradicted"
        and v.direction_ok is True
        and v.magnitude_ok is False
    )
    stable_claims = [v for v in global_verdicts if v.direction == "stable"]
    detailed = Counter()
    for verdict in global_verdicts:
        if verdict.category != "contradicted":
            continue
        observed = verdict.observed
        if verdict.direction == "stable":
            detailed["stable_on_deviation"] += 1
        elif observed is not None and abs(observed) < 0.2:
            detailed["directional_on_near_zero"] += 1
        elif verdict.direction == "increase" and observed is not None and observed < 0:
            detailed["reversed_direction"] += 1
        elif verdict.direction == "decrease" and observed is not None and observed > 0:
            detailed["reversed_direction"] += 1
        else:
            detailed["magnitude_only"] += 1
    breakdown = {
        "contradicted_direction": contradicted_direction,
        "contradicted_magnitude_only": contradicted_magnitude,
        "contradicted_detail": dict(detailed),
        "stable_claims": len(stable_claims),
        "stable_contradicted": sum(
            1 for v in stable_claims if v.category == "contradicted"
        ),
        "declared_directions": dict(Counter(v.direction for v in global_verdicts)),
    }

    examples = [
        {
            "record_id": verdict_records.get(id(verdict)),
            "metric": verdict.metric,
            "declared": f"{verdict.direction} ({verdict.magnitude})",
            "observed": verdict.observed,
            "category": verdict.category,
            "detail": verdict.detail,
        }
        for verdict in global_verdicts
        if verdict.category == "contradicted"
    ][:5]

    return {
        "global": summarize_verdicts(global_verdicts),
        "breakdown": breakdown,
        "by_class": by_class,
        "by_metric": metric_block,
        "contradicted_examples": examples,
    }


def certainty_by_class(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Índice de certeza lingüística por clase verdadera."""
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in RHYTHM_CLASSES}
    all_rows: list[dict[str, Any]] = []
    for record in rows:
        if record["status"] not in VALID_STATUSES:
            continue
        certainty = linguistic_certainty(record.get("reasoning") or "")
        if not certainty["n_words"]:
            continue
        grouped[labels[record["record_id"]]].append(certainty)
        all_rows.append(certainty)

    def mean_block(entries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        if not entries:
            return {
                "certainty_index_mean": float("nan"),
                "hedges_mean": float("nan"),
                "boosters_mean": float("nan"),
                "n": 0,
            }
        return {
            "certainty_index_mean": float(
                np.mean([e["certainty_index"] for e in entries])
            ),
            "hedges_mean": float(np.mean([e["hedges"] for e in entries])),
            "boosters_mean": float(np.mean([e["boosters"] for e in entries])),
            "n": len(entries),
        }

    return {
        "global": mean_block(all_rows),
        "by_class": {label: mean_block(entries) for label, entries in grouped.items()},
        "significance": _certainty_significance(grouped),
    }


def _certainty_significance(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Contraste de certeza entre ``noisy`` y el resto de clases."""
    from scipy.stats import kruskal, mannwhitneyu

    samples = {
        label: [e["certainty_index"] for e in entries]
        for label, entries in grouped.items()
        if entries
    }
    if len(samples) < 2:
        return {}
    kruskal_stat = kruskal(*samples.values())
    noisy = samples.get("noisy", [])
    clean = [value for label, values in samples.items() if label != "noisy" for value in values]
    result: dict[str, Any] = {
        "kruskal_h": float(kruskal_stat.statistic),
        "kruskal_p": float(kruskal_stat.pvalue),
        "n_classes": len(samples),
    }
    if noisy and clean:
        test = mannwhitneyu(noisy, clean, alternative="two-sided")
        wins = sum(1 for a in noisy for b in clean if a > b)
        ties = sum(1 for a in noisy for b in clean if a == b)
        cliffs = (wins + 0.5 * ties) / (len(noisy) * len(clean))
        result.update(
            {
                "noisy_vs_rest_u": float(test.statistic),
                "noisy_vs_rest_p": float(test.pvalue),
                "noisy_vs_rest_cliffs_delta": cliffs,
                "noisy_mean": float(np.mean(noisy)),
                "rest_mean": float(np.mean(clean)),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Señal de calidad y consistencia
# ---------------------------------------------------------------------------
def class_signal(test: Sequence[RhythmRecord]) -> dict[str, Any]:
    """AUROC univariante por clase y feature (¿cuánta señal hay en la entrada?).

    Se orienta cada AUROC para que ``>0.5`` indique que la feature separa la
    clase del resto en la dirección natural (mayor o menor valor).
    """
    block: dict[str, Any] = {}
    for label in RHYTHM_CLASSES:
        per_feature = {}
        for name in CANONICAL_FEATURES:
            positive = [
                float(r.values[name])
                for r in test
                if r.label == label and r.values.get(name) is not None
            ]
            negative = [
                float(r.values[name])
                for r in test
                if r.label != label and r.values.get(name) is not None
            ]
            value = auroc(positive, negative)
            per_feature[name] = {
                "auroc_signed": value,
                "auroc_oriented": max(value, 1.0 - value),
            }
        best = max(per_feature.items(), key=lambda item: item[1]["auroc_oriented"])
        block[label] = {
            "per_feature": per_feature,
            "best_feature": best[0],
            "best_auroc": best[1]["auroc_oriented"],
        }
    return block


def quality_signal(
    dev: Sequence[RhythmRecord],
    test: Sequence[RhythmRecord],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """¿Hay en la entrada una señal de ``noisy`` que el LLM no use?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    univariate = {}
    for name in CANONICAL_FEATURES:
        positive = [
            float(r.values[name])
            for r in test
            if r.label == "noisy" and r.values.get(name) is not None
        ]
        negative = [
            float(r.values[name])
            for r in test
            if r.label != "noisy" and r.values.get(name) is not None
        ]
        univariate[name] = {
            "auroc": auroc(positive, negative),
            "median_noisy": float(np.median(positive)) if positive else float("nan"),
            "median_clean": float(np.median(negative)) if negative else float("nan"),
        }

    def matrix(records: Sequence[RhythmRecord]) -> np.ndarray:
        return np.asarray(
            [
                [
                    float("nan") if r.values.get(name) is None else float(r.values[name])
                    for name in CANONICAL_FEATURES
                ]
                for r in records
            ],
            dtype=float,
        )

    from sklearn.impute import SimpleImputer

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(matrix(dev), [1 if r.label == "noisy" else 0 for r in dev])
    probabilities = model.predict_proba(matrix(test))[:, 1]

    positive_scores = [p for p, r in zip(probabilities, test) if r.label == "noisy"]
    negative_scores = [p for p, r in zip(probabilities, test) if r.label != "noisy"]

    row_by_id = {
        r["record_id"]: r for r in rows if r["status"] in VALID_STATUSES
    }
    gate_curve = []
    for target in (0.5, 0.8, 0.9):
        threshold = float(np.quantile(positive_scores, 1.0 - target))
        detected = sum(1 for p in positive_scores if p >= threshold)
        lost = sum(1 for p in negative_scores if p >= threshold)
        retained = [
            (p, record)
            for p, record in zip(probabilities, test)
            if p < threshold and row_by_id.get(record.record_id, {}).get("decision")
            not in (None, "abstain")
        ]
        correct = sum(
            1
            for _, record in retained
            if row_by_id[record.record_id]["decision"] == record.label
        )
        gate_curve.append(
            {
                "target_noisy_recall": target,
                "threshold": threshold,
                "noisy_detected": detected,
                "noisy_recall": detected / len(positive_scores),
                "clean_abstained": lost,
                "clean_coverage_loss": lost / len(negative_scores),
                "n_retained": len(retained),
                "coverage": len(retained) / len(test),
                "accuracy_retained": (
                    correct / len(retained) if retained else float("nan")
                ),
            }
        )

    return {
        "note": (
            "Detector entrenado solo con registros de desarrollo; las métricas "
            "se calculan en el conjunto de evaluación."
        ),
        "univariate_auroc": univariate,
        "multivariate": {
            "auroc": auroc(positive_scores, negative_scores),
            "average_precision": average_precision(positive_scores, negative_scores),
            "prevalence_noisy": len(positive_scores) / len(test),
        },
        "gate_curve": gate_curve,
    }


def consistency_block(test: Sequence[RhythmRecord]) -> dict[str, Any]:
    """Compuerta de consistencia de P1 aplicada a este dataset."""
    per_class: dict[str, Counter[str]] = {label: Counter() for label in RHYTHM_CLASSES}
    evaluated = Counter()
    vetoes = Counter()
    vetoes_key_features = Counter()
    failed = Counter()
    vetoed_ids: set[str] = set()
    for record in test:
        report = check_consistency(record.values, duration_s=record.duration_s)
        per_class[record.label]["n"] += 1
        per_class[record.label]["evaluated"] += len(report.evaluated)
        evaluated[len(report.evaluated)] += 1
        for check in report.failures:
            failed[check.name] += 1
        if should_abstain(report, require_key_features=False):
            vetoes[record.label] += 1
            per_class[record.label]["vetoed"] += 1
            vetoed_ids.add(record.record_id)
        if should_abstain(report, require_key_features=True):
            vetoes_key_features[record.label] += 1

    return {
        "n_checks_evaluable": {str(k): v for k, v in sorted(evaluated.items())},
        "failed_checks": dict(failed),
        "vetoes_by_class": dict(vetoes),
        "vetoes_by_class_require_key_features": dict(vetoes_key_features),
        "vetoed_ids": sorted(vetoed_ids),
        "per_class": {
            label: {
                "n": counts["n"],
                "checks_evaluated": counts["evaluated"],
                "vetoed": counts["vetoed"],
            }
            for label, counts in per_class.items()
        },
        "note": (
            "Con ``require_key_features=True`` (configuración de WESAD) la "
            "compuerta vetaría todo el dataset: sus features clave son de EDA y "
            "respiración y aquí solo hay ECG."
        ),
    }


def composite_system(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    vetoed_ids: set[str],
) -> dict[str, Any]:
    """LLM + compuerta de consistencia: abstención forzada antes del modelo."""
    valid = [r for r in rows if r["status"] in VALID_STATUSES]
    decided_ids = {
        r["record_id"]
        for r in valid
        if r["decision"] != "abstain" and r["record_id"] not in vetoed_ids
    }
    decided = [r for r in valid if r["record_id"] in decided_ids]
    abstained = [r for r in valid if r["record_id"] not in decided_ids]
    hits = sum(1 for r in decided if labels[r["record_id"]] == r["decision"])

    per_class = {}
    for label in RHYTHM_CLASSES:
        in_class = [r for r in valid if labels[r["record_id"]] == label]
        class_decided = [r for r in in_class if r["record_id"] in decided_ids]
        per_class[label] = {
            "n": len(in_class),
            "n_decided": len(class_decided),
            "abstention_rate": (
                (len(in_class) - len(class_decided)) / len(in_class)
                if in_class
                else float("nan")
            ),
            "recall": (
                sum(1 for r in class_decided if r["decision"] == label)
                / len(class_decided)
                if class_decided
                else float("nan")
            ),
        }

    recalls = [e["recall"] for e in per_class.values() if not np.isnan(e["recall"])]
    return {
        "n_decided": len(decided),
        "n_abstain": len(abstained),
        "abstention_rate": len(abstained) / len(valid) if valid else float("nan"),
        "accuracy": hits / len(decided) if decided else float("nan"),
        "accuracy_incl_abstain": hits / len(valid) if valid else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "per_class": per_class,
    }



# ---------------------------------------------------------------------------
# Ablación few-shot en desarrollo
# ---------------------------------------------------------------------------
def few_shot_ablation(
    base_dir: Path,
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Compara las dos ejecuciones de desarrollo (mismos registros)."""
    from scipy.stats import binomtest

    runs = {}
    for tag in ("dev_nofs", "dev_fs"):
        rows = load_jsonl(base_dir / f"physionet2017_{tag}.responses.jsonl")
        rows = [r for r in rows if r["record_id"] in labels]
        metrics = llm_metrics(rows, labels)
        metrics["claims"] = summarize_verdicts(
            [
                verdict
                for record in rows
                if record["status"] in VALID_STATUSES and record.get("claims")
                for verdict in verify_claims(record["claims"], record["prompt"])
            ]
        )
        certainties = [
            linguistic_certainty(record.get("reasoning") or "")
            for record in rows
            if record["status"] in VALID_STATUSES
        ]
        certainties = [c for c in certainties if c["n_words"]]
        metrics["certainty_index_mean"] = (
            float(np.mean([c["certainty_index"] for c in certainties]))
            if certainties
            else float("nan")
        )
        runs[tag] = {"rows": rows, "metrics": metrics}

    base = {r["record_id"]: r for r in runs["dev_nofs"]["rows"]}
    with_fs = {r["record_id"]: r for r in runs["dev_fs"]["rows"]}
    shared = sorted(set(base) & set(with_fs))
    agreement = sum(1 for rid in shared if base[rid]["decision"] == with_fs[rid]["decision"])

    correct_base = {rid: base[rid]["decision"] == labels[rid] for rid in shared}
    correct_fs = {rid: with_fs[rid]["decision"] == labels[rid] for rid in shared}
    only_base = sum(1 for rid in shared if correct_base[rid] and not correct_fs[rid])
    only_fs = sum(1 for rid in shared if correct_fs[rid] and not correct_base[rid])
    discordant = only_base + only_fs
    p_value = (
        binomtest(only_fs, discordant, 0.5).pvalue if discordant else float("nan")
    )

    return {
        "no_few_shot": {
            key: value
            for key, value in runs["dev_nofs"]["metrics"].items()
            if key not in ("confusion",)
        },
        "few_shot": {
            key: value
            for key, value in runs["dev_fs"]["metrics"].items()
            if key not in ("confusion",)
        },
        "paired": {
            "n_shared": len(shared),
            "decision_agreement": agreement / len(shared) if shared else float("nan"),
            "correct_only_no_few_shot": only_base,
            "correct_only_few_shot": only_fs,
            "mcnemar_exact_p": float(p_value),
        },
    }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def analyze(base_dir: Path) -> dict[str, Any]:
    """Ejecuta todos los bloques y devuelve el informe."""
    labels_test = load_labels(base_dir / "physionet2017.labels.jsonl")
    labels_dev = load_labels(base_dir / "physionet2017_dev_nofs.labels.jsonl")
    cache = json.loads((base_dir / "physionet2017.features.json").read_text())
    summary = json.loads((base_dir / "physionet2017_summary.json").read_text())

    raw_rows = load_jsonl(base_dir / "physionet2017.responses.jsonl")
    rows = dedupe_responses(raw_rows)
    dropped = [r["record_id"] for r in rows if r["record_id"] not in labels_test]
    rows = [r for r in rows if r["record_id"] in labels_test]

    dev = rhythm_records(cache, sorted(labels_dev), labels_dev)
    test = rhythm_records(cache, sorted(labels_test), labels_test)

    centers, scales = population_reference(dev)
    stored = summary["population_reference"]
    reference_matches_summary = all(
        abs(centers[name] - stored["centers"][name]) < 1e-6
        and abs(scales[name] - stored["scales"][name]) < 1e-6
        for name in CANONICAL_FEATURES
    )

    llm = llm_metrics(rows, labels_test)
    reference = reference_metrics(dev, test, centers, scales)
    consistency = consistency_block(test)

    majority_label = Counter(labels_test.values()).most_common(1)[0][0]
    majority_hits = sum(1 for label in labels_test.values() if label == majority_label)
    majority = {
        "label": majority_label,
        "accuracy_all": majority_hits / len(labels_test),
        "accuracy_decided": majority_hits / llm["n_decided"],
        "balanced_accuracy": 1.0 / len(RHYTHM_CLASSES),
    }

    noisy_n = llm["per_class"]["noisy"]["n"]
    noisy_abstain = llm["per_class"]["noisy"]["abstentions"]
    noisy_interval = wilson_interval(noisy_abstain, noisy_n)

    excess = {
        "accuracy": llm["accuracy"] - reference["accuracy"],
        "balanced_accuracy": llm["balanced_accuracy"] - reference["balanced_accuracy"],
        "per_class_recall": {
            label: llm["per_class"][label]["recall"] - reference["per_class"][label]["recall"]
            for label in RHYTHM_CLASSES
        },
    }

    # Recuperación tolerante (misma política que el brazo textual de WESAD).
    recovered_rows, n_recovered = apply_recovery(
        list(rows), decision_mode=DECISION_MODE_RHYTHM4
    )
    recovery = {
        "n_recovered": n_recovered,
        "metrics": llm_metrics(recovered_rows, labels_test),
    }

    return {
        "dataset": "PHYSIONET2017",
        "task": "4-class rhythm (normal/af/other-rhythm/noisy)",
        "generated_at": utc_now(),
        "inputs": {
            "n_response_lines": len(raw_rows),
            "n_unique_responses": len(dedupe_responses(raw_rows)),
            "dropped_unlabeled": dropped,
            "n_test_labels": len(labels_test),
            "n_dev_labels": len(labels_dev),
            "reference_matches_stored_summary": bool(reference_matches_summary),
        },
        "llm": llm,
        "recovery": {
            "n_recovered": recovery["n_recovered"],
            "accuracy": recovery["metrics"]["accuracy"],
            "balanced_accuracy": recovery["metrics"]["balanced_accuracy"],
            "n_decided": recovery["metrics"]["n_decided"],
            "decisions": recovery["metrics"]["decisions"],
        },
        "majority_baseline": majority,
        "reference": reference,
        "excess_cost": excess,
        "abstention_on_noisy": {
            "n": noisy_n,
            "n_abstain": noisy_abstain,
            "rate": noisy_abstain / noisy_n if noisy_n else float("nan"),
            "wilson_95": list(noisy_interval),
            "n_predicted_noisy": llm["decisions"].get("noisy", 0),
        },
        "claims": claims_by_class(rows, labels_test),
        "claims_vs_correctness": claims_vs_correctness(rows, labels_test),
        "certainty": certainty_by_class(rows, labels_test),
        "class_signal": class_signal(test),
        "quality_signal": quality_signal(dev, test, rows),
        "consistency": consistency,
        "composite_system": composite_system(
            rows, labels_test, set(consistency["vetoed_ids"])
        ),
        "few_shot_ablation": few_shot_ablation(base_dir, labels_dev),
    }


def print_report(report: Mapping[str, Any]) -> None:
    """Resumen legible por consola."""
    llm = report["llm"]
    reference = report["reference"]
    print("\n=== PhysioNet 2017 — análisis ===")
    print(
        f"Registros: {report['inputs']['n_test_labels']} test / "
        f"{report['inputs']['n_dev_labels']} desarrollo | "
        f"respuestas únicas: {report['inputs']['n_unique_responses']} | "
        f"descartadas sin etiqueta: {len(report['inputs']['dropped_unlabeled'])}"
    )
    print(
        f"Referencia poblacional reproducible: "
        f"{report['inputs']['reference_matches_stored_summary']}"
    )

    print("\n-- Corrección --")
    print(
        f"LLM: acc={llm['accuracy']:.3f} bal={llm['balanced_accuracy']:.3f} "
        f"decididas={llm['n_decided']} inválidas={llm['n_invalid']} "
        f"abstención={llm['abstention_rate']:.3f}"
    )
    print(
        f"Referencia: acc={reference['accuracy']:.3f} "
        f"bal={reference['balanced_accuracy']:.3f}"
    )
    print(
        f"Mayoría trivial ({report['majority_baseline']['label']}): "
        f"acc={report['majority_baseline']['accuracy_decided']:.3f} "
        f"bal={report['majority_baseline']['balanced_accuracy']:.3f}"
    )
    print(f"Exceso de coste (LLM − referencia): {report['excess_cost']['accuracy']:+.3f}")

    print("\n-- Por clase (LLM vs referencia) --")
    header = f"{'clase':<14}{'n':>5}{'recall':>8}{'ref':>8}{'Δ':>8}{'abst':>7}"
    print(header)
    for label, entry in llm["per_class"].items():
        delta = entry["recall"] - reference["per_class"][label]["recall"]
        print(
            f"{label:<14}{entry['n']:>5}{entry['recall']:>8.3f}"
            f"{reference['per_class'][label]['recall']:>8.3f}{delta:>+8.3f}"
            f"{entry['abstention_rate']:>7.3f}"
        )

    noisy = report["abstention_on_noisy"]
    print(
        f"\nAbstención en noisy: {noisy['n_abstain']}/{noisy['n']} "
        f"(IC95 {noisy['wilson_95'][0]:.3f}–{noisy['wilson_95'][1]:.3f}); "
        f"predicciones 'noisy': {noisy['n_predicted_noisy']}"
    )

    claims = report["claims"]["global"]
    print(
        f"\nClaims: n={claims['n_claims']} respaldados={claims['supported_rate']:.3f} "
        f"contradichos={claims['contradicted_rate']:.3f} "
        f"no respaldados={claims['unsupported_rate']:.3f} "
        f"dirección={claims['direction_accuracy']:.3f}"
    )
    print("Por clase:")
    for label, block in report["claims"]["by_class"].items():
        if not block["n_claims"]:
            continue
        print(
            f"  {label:<14} n={block['n_claims']:>4} "
            f"resp={block['supported_rate']:.3f} "
            f"contra={block['contradicted_rate']:.3f} "
            f"dirección={block['direction_accuracy']:.3f}"
        )
    print("\nPor métrica (contradichos/total):")
    for metric, block in report["claims"]["by_metric"].items():
        print(
            f"  {metric:<16} {block['contradicted']:>4}/{block['n']:<4} "
            f"resp={block['supported']}"
        )

    breakdown = report["claims"]["breakdown"]
    print(
        f"  contradichos por dirección: {breakdown['contradicted_direction']} | "
        f"por magnitud: {breakdown['contradicted_magnitude_only']} | "
        f"stable: {breakdown['stable_claims']} "
        f"(contradichos {breakdown['stable_contradicted']})"
    )

    support = report["claims_vs_correctness"]
    print(
        f"  respaldo vs corrección: aciertos {support['supported_rate_correct']:.3f} "
        f"(n={support['n_correct']}) vs fallos "
        f"{support['supported_rate_incorrect']:.3f} (n={support['n_incorrect']}) | "
        f"AUROC={support['auroc_support_for_correctness']:.3f}"
    )

    certainty = report["certainty"]
    print(f"\nCerteza global: {certainty['global']['certainty_index_mean']:+.2f}")
    for label, block in certainty["by_class"].items():
        print(f"  {label:<14} {block['certainty_index_mean']:+.2f} (n={block['n']})")
    significance = certainty.get("significance", {})
    if significance:
        print(
            f"  Kruskal-Wallis H={significance['kruskal_h']:.2f} "
            f"p={significance['kruskal_p']:.4f}"
        )
        if "noisy_vs_rest_p" in significance:
            print(
                f"  noisy vs resto: p={significance['noisy_vs_rest_p']:.4f} "
                f"Cliff δ={significance['noisy_vs_rest_cliffs_delta']:+.3f} "
                f"(medias {significance['noisy_mean']:+.2f} vs "
                f"{significance['rest_mean']:+.2f})"
            )

    print("\n-- Señal univariante por clase (mejor feature, AUROC orientado) --")
    for label, block in report["class_signal"].items():
        print(
            f"  {label:<14} {block['best_feature']:<16} "
            f"AUROC={block['best_auroc']:.3f}"
        )

    quality = report["quality_signal"]
    print("\n-- Señal de calidad en la entrada (noisy vs resto) --")
    for name, block in quality["univariate_auroc"].items():
        print(
            f"  {name:<16} AUROC={block['auroc']:.3f} "
            f"(mediana noisy {block['median_noisy']:.1f} vs "
            f"limpio {block['median_clean']:.1f})"
        )
    multivariate = quality["multivariate"]
    print(
        f"  multivariante: AUROC={multivariate['auroc']:.3f} "
        f"AP={multivariate['average_precision']:.3f} "
        f"(prevalencia {multivariate['prevalence_noisy']:.3f})"
    )
    print("  compuerta (si se abstuviera por ruido detectado):")
    for point in quality["gate_curve"]:
        print(
            f"    recall noisy objetivo {point['target_noisy_recall']:.1f} → "
            f"detectados {point['noisy_detected']} | "
            f"limpios perdidos {point['clean_abstained']} "
            f"({point['clean_coverage_loss']:.3f}) | cobertura "
            f"{point['coverage']:.3f} acc={point['accuracy_retained']:.3f}"
        )

    consistency = report["consistency"]
    print(
        f"\nConsistencia: vetos={consistency['vetoes_by_class']} "
        f"fallos={consistency['failed_checks'] or 'ninguno'}"
    )
    print(
        "  con features clave (config WESAD): "
        f"{consistency['vetoes_by_class_require_key_features']}"
    )
    composite = report["composite_system"]
    print(
        f"  sistema LLM+compuerta: decididas={composite['n_decided']} "
        f"abstención={composite['abstention_rate']:.3f} "
        f"acc={composite['accuracy']:.3f} bal={composite['balanced_accuracy']:.3f} "
        f"acc_incl_abstain={composite['accuracy_incl_abstain']:.3f}"
    )

    ablation = report["few_shot_ablation"]
    print("\n-- Ablación few-shot (desarrollo, mismos registros) --")
    for tag, key in (("sin few-shot", "no_few_shot"), ("con few-shot", "few_shot")):
        block = ablation[key]
        print(
            f"  {tag:<14} acc={block['accuracy']:.3f} bal={block['balanced_accuracy']:.3f} "
            f"decisiones={block['decisions']} inválidas={block['n_invalid']} "
            f"claims resp={block['claims']['supported_rate']:.3f} "
            f"certeza={block['certainty_index_mean']:+.2f}"
        )
    paired = ablation["paired"]
    print(
        f"  acuerdo={paired['decision_agreement']:.3f} | "
        f"solo acierta sin fs: {paired['correct_only_no_few_shot']} | "
        f"solo acierta con fs: {paired['correct_only_few_shot']} | "
        f"McNemar p={paired['mcnemar_exact_p']:.3f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="Directorio P3.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON de salida.")
    args = parser.parse_args(argv)

    base_dir = Path(args.dir)
    report = analyze(base_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print_report(report)
    print(f"\nInforme: {output}")


if __name__ == "__main__":
    main()
