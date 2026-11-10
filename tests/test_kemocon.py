"""Tests del adaptador de K-EmoCon (P3)."""

from __future__ import annotations

import math

import pytest

from sources.external.kemocon import (
    KEMOCON_FEATURES,
    aggregate_windows,
    build_payload,
    opaque_window_id,
    subject_centered,
)


def _rows(subject: str, n: int, base_hr: float = 70.0, base_eda: float = 3.0):
    rows = {}
    for win in range(n):
        rows[(subject, str(win))] = {
            "hr_mean": base_hr + win,
            "polar_hr_mean": None,
            "ibi_mean": 60000.0 / (base_hr + win),
            "ibi_std": 20.0 + win,
            "eda_mean": base_eda + 0.1 * win,
            "eda_std": 0.1,
            "temp_mean": 30.0,
            "temp_std": 0.01,
            "acc_std": 0.5,
        }
    return rows


def _assignments(subject: str, periods: dict[str, int]):
    assignments = {}
    counter = 0
    for period, n in periods.items():
        for win_in_period in range(n):
            assignments[(subject, str(counter))] = (period, win_in_period)
            counter += 1
    return assignments


def test_aggregate_builds_60s_windows():
    rows = _rows("1", 24)
    assignments = _assignments("1", {"pre": 12, "debate": 12})
    windows = aggregate_windows(rows, assignments)
    assert len(windows) == 2
    assert windows[0].period == "pre"
    assert windows[1].period == "debate"
    assert windows[0].n_subwindows == 12
    assert windows[0].duration_s == 60.0


def test_aggregate_means_and_rms():
    rows = _rows("1", 12)
    assignments = _assignments("1", {"pre": 12})
    window = aggregate_windows(rows, assignments)[0]
    # hr_mean = 70..81 => media 75,5
    assert window.values["ecg_hr_bpm"] == pytest.approx(75.5)
    # EDA media de 3,0 a 4,1
    assert window.values["eda_mean_us"] == pytest.approx(3.55)
    # ibi_std 20..31 => rms
    expected_rms = math.sqrt(sum((20 + i) ** 2 for i in range(12)) / 12)
    assert window.values["ecg_sdnn_ms"] == pytest.approx(expected_rms)


def test_aggregate_drops_incomplete_window():
    rows = _rows("1", 12)
    # Quitar la mitad de las subventanas de EDA => cobertura 0,5 exacta.
    for win in range(6):
        rows[("1", str(win))]["eda_mean"] = None
    assignments = _assignments("1", {"pre": 12})
    assert len(aggregate_windows(rows, assignments, min_coverage=0.6)) == 0
    assert len(aggregate_windows(rows, assignments, min_coverage=0.5)) == 1


def test_aggregate_filters_subjects():
    rows = {**_rows("1", 12), **_rows("2", 12)}
    assignments = {**_assignments("1", {"pre": 12}), **_assignments("2", {"pre": 12})}
    windows = aggregate_windows(rows, assignments, subjects=["2"])
    assert {w.subject for w in windows} == {"2"}


def test_periods_are_not_stress_labels():
    """K-EmoCon no tiene etiqueta de estrés: los periodos son solo periodos."""
    rows = _rows("1", 24)
    assignments = _assignments("1", {"pre": 12, "debate": 12})
    windows = aggregate_windows(rows, assignments)
    assert {w.period for w in windows} == {"pre", "debate"}
    assert all(w.affect_label is None for w in windows)


def _ratings_rows(subject: str, n: int, values):
    rows = _rows(subject, n)
    for index, value in enumerate(values):
        rows[(subject, str(index))]["self_arousal"] = value
        rows[(subject, str(index))]["external_arousal_mean"] = value
    return rows


def test_assign_binary_affect_label_within_subject_by_median():
    from sources.external.kemocon import assign_binary_affect_label

    rows = _ratings_rows("1", 144, list(range(144)))
    assignments = _assignments("1", {"debate": 144})
    windows = aggregate_windows(rows, assignments)
    assert len(windows) == 12
    labelled = assign_binary_affect_label(windows, source="self_arousal")
    assert labelled == 12
    assert all(w.affect_label in ("high-arousal", "low-arousal") for w in windows)
    assert sum(1 for w in windows if w.affect_label == "high-arousal") == 6


def test_assign_binary_affect_label_global_split():
    from sources.external.kemocon import assign_binary_affect_label

    rows = {**_ratings_rows("1", 12, list(range(12))),
            **_ratings_rows("2", 12, list(range(100, 112)))}
    assignments = {**_assignments("1", {"pre": 12}), **_assignments("2", {"pre": 12})}
    windows = aggregate_windows(rows, assignments)
    labelled = assign_binary_affect_label(
        windows, source="self_arousal", split="global", period="pre"
    )
    assert labelled == 2
    # La media de 100..111 está por encima de la mediana global.
    assert [w.affect_label for w in windows if w.subject == "2"] == ["high-arousal"]


def test_assign_binary_affect_label_invalid_source():
    from sources.external.kemocon import assign_binary_affect_label

    rows = _rows("1", 12)
    windows = aggregate_windows(rows, _assignments("1", {"pre": 12}))
    with pytest.raises(ValueError, match="Fuente afectiva"):
        assign_binary_affect_label(windows, source="cortisol")


def test_subject_centered_is_within_subject_and_clipped():
    rows = _rows("1", 24)
    assignments = _assignments("1", {"pre": 12, "debate": 12})
    windows = aggregate_windows(rows, assignments)
    subject_centered(windows, features=KEMOCON_FEATURES, method="robust")
    for name in KEMOCON_FEATURES:
        zs = [w.zscores[name] for w in windows if w.zscores[name] is not None]
        assert zs, name
        assert all(-3.0 <= value <= 3.0 for value in zs)


def test_build_payload_excludes_motion():
    rows = _rows("1", 12)
    assignments = _assignments("1", {"pre": 12})
    windows = aggregate_windows(rows, assignments)
    subject_centered(windows, features=KEMOCON_FEATURES)
    payload = build_payload(windows[0])
    assert set(payload["modalities"]) == {"ecg", "eda", "temp"}
    assert payload["window"]["duration_s"] == 60.0
    assert payload["window"]["window_id"] == opaque_window_id(windows[0].window_id)


def test_opaque_id_is_stable():
    assert opaque_window_id("KEMOCON:1:pre:w000") == opaque_window_id(
        "KEMOCON:1:pre:w000"
    )
    assert opaque_window_id("KEMOCON:1:pre:w000") != opaque_window_id(
        "KEMOCON:1:pre:w001"
    )
