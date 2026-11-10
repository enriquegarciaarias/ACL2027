"""Comprobación de consistencia fisiológica interna (P1 de Fase 2).

Las features de una ventana no son independientes: varias relaciones son
aritméticamente deterministas (``n_beats ≈ HR × duración``,
``SCR_rate ≈ SCR_count / duración``, ``EDA_mean ≈ EDA_tonic``,
``SCR_amplitude_mean ≤ SCR_amplitude_max``). Los controles negativos que
barajan valores dentro de una modalidad rompen esas relaciones aunque
conserven la distribución marginal.

Este módulo explota esas relaciones para dos usos:

1. **Compuerta determinista previa al LLM** (``should_abstain``): si la
   evidencia es internamente incoherente, el sistema se abstiene sin llamar
   al modelo. Es una comprobación de calidad *derivada de las propias
   features*, no una señal externa (no usa SQI ni cobertura).
2. **Diagnóstico** de una ventana para auditoría (``ConsistencyReport.to_dict``).

Se calcula sobre los valores **absolutos** (no sobre z-scores intra-sujeto):
la estandarización por sujeto destruye las relaciones deterministas, y por
eso la vista E12 centrada no permite verificarlas.

Todas las comprobaciones son no supervisadas y no usan etiquetas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: Tolerancia relativa de cada comprobación y su severidad.
#: - ``hard``: relación aritméticamente determinista; un fallo es una
#:   incoherencia fuerte (imposible en datos reales).
#: - ``soft``: relación empírica con margen amplio; un fallo aislado puede
#:   ser legítimo, varios fallos juntos indican corrupción.
CHECK_TOLERANCES: dict[str, float] = {
    "ecg_hr_vs_rr_mean": 0.05,
    "ecg_n_beats_vs_hr": 0.10,
    "eda_scr_rate_vs_count": 0.10,
    "eda_scr_amplitude_mean_vs_max": 0.01,
    "eda_mean_vs_tonic": 0.05,
    "resp_n_cycles_vs_rate": 0.60,
    "temp_range_vs_std": 2.00,          # rango >= 2 × DE (datos reales: >= 3.77)
    "eda_std_vs_phasic_std": 1.00,      # DE >= DE(fásica) (datos reales: >= 1.15)
}

CHECK_SEVERITY: dict[str, str] = {
    "ecg_hr_vs_rr_mean": "hard",
    "ecg_n_beats_vs_hr": "hard",
    "eda_scr_rate_vs_count": "hard",
    "eda_scr_amplitude_mean_vs_max": "hard",
    "eda_mean_vs_tonic": "hard",
    "resp_n_cycles_vs_rate": "soft",
    "temp_range_vs_std": "soft",
    "eda_std_vs_phasic_std": "soft",
}

#: Features cuya ausencia impide decidir con garantías (usadas por la
#: compuerta cuando ``require_key_features`` está activo).
KEY_FEATURES: tuple[str, ...] = (
    "ecg_hr_bpm",
    "ecg_n_beats",
    "eda_scr_count",
    "eda_scr_rate_per_min",
    "resp_rate_bpm",
    "resp_n_cycles",
)

#: Umbrales por defecto de la compuerta: ningún fallo ``hard`` y como mucho
#: un fallo ``soft`` (en los datos limpios del corpus no se observa ninguno).
DEFAULT_MAX_HARD_FAILURES: int = 0
DEFAULT_MAX_SOFT_FAILURES: int = 1


@dataclass(frozen=True)
class CheckOutcome:
    """Resultado de una comprobación de consistencia."""

    name: str
    severity: str
    kind: str  # "identity" (igualdad con tolerancia) | "inequality"
    reported: float | None
    expected: float | None
    relative_error: float | None
    tolerance: float
    ok: bool | None  # None = no evaluable (faltan features de entrada)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "kind": self.kind,
            "reported": self.reported,
            "expected": self.expected,
            "relative_error": self.relative_error,
            "tolerance": self.tolerance,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ConsistencyReport:
    """Conjunto de comprobaciones sobre una ventana."""

    checks: tuple[CheckOutcome, ...] = field(default_factory=tuple)
    missing_key_features: tuple[str, ...] = field(default_factory=tuple)

    @property
    def evaluated(self) -> tuple[CheckOutcome, ...]:
        return tuple(c for c in self.checks if c.ok is not None)

    @property
    def failures(self) -> tuple[CheckOutcome, ...]:
        return tuple(c for c in self.checks if c.ok is False)

    @property
    def n_hard_failures(self) -> int:
        return sum(1 for c in self.failures if c.severity == "hard")

    @property
    def n_soft_failures(self) -> int:
        return sum(1 for c in self.failures if c.severity == "soft")

    @property
    def consistency_score(self) -> float:
        """Fracción de comprobaciones evaluables que pasan (1.0 si ninguna)."""
        evaluated = self.evaluated
        if not evaluated:
            return 0.0
        passed = sum(1 for c in evaluated if c.ok)
        return passed / len(evaluated)

    def failed_names(self) -> list[str]:
        return [c.name for c in self.failures]

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistency_score": self.consistency_score,
            "n_evaluated": len(self.evaluated),
            "n_hard_failures": self.n_hard_failures,
            "n_soft_failures": self.n_soft_failures,
            "failed_checks": self.failed_names(),
            "missing_key_features": list(self.missing_key_features),
            "checks": [c.to_dict() for c in self.checks],
        }


def _number(values: Mapping[str, float | None], name: str) -> float | None:
    """Devuelve el valor numérico finito de ``name`` o ``None``."""
    value = values.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _identity_check(
    name: str,
    reported: float | None,
    expected: float | None,
    tolerance: float,
    abs_tolerance: float = 0.0,
) -> CheckOutcome:
    """Compara un valor reportado con su valor esperado (error relativo)."""
    if reported is None or expected is None:
        return CheckOutcome(
            name=name,
            severity=CHECK_SEVERITY[name],
            kind="identity",
            reported=reported,
            expected=expected,
            relative_error=None,
            tolerance=tolerance,
            ok=None,
        )
    if abs(expected) <= abs_tolerance:
        # Caso degenerado: esperado ~0 (p. ej. 0 SCR). Se usa error absoluto.
        denominator = abs_tolerance if abs_tolerance > 0 else 1e-9
        error = abs(reported - expected) / denominator
    else:
        error = abs(reported - expected) / abs(expected)
    return CheckOutcome(
        name=name,
        severity=CHECK_SEVERITY[name],
        kind="identity",
        reported=reported,
        expected=expected,
        relative_error=error,
        tolerance=tolerance,
        ok=error <= tolerance,
    )


def _missing_check(name: str, reported: float | None, bound: float | None) -> CheckOutcome:
    return CheckOutcome(
        name=name,
        severity=CHECK_SEVERITY[name],
        kind="inequality",
        reported=reported,
        expected=bound,
        relative_error=None,
        tolerance=CHECK_TOLERANCES[name],
        ok=None,
    )


def _lower_bound_check(
    name: str,
    reported: float | None,
    bound: float | None,
    factor: float,
    scale: float | None = None,
) -> CheckOutcome:
    """Comprueba ``reported >= factor × bound`` (p. ej. rango >= 2 × DE)."""
    if reported is None or bound is None:
        return _missing_check(name, reported, bound)
    threshold = factor * bound
    denominator = abs(scale) if scale else (abs(threshold) if threshold else 1.0)
    error = max(0.0, threshold - reported) / denominator
    return CheckOutcome(
        name=name,
        severity=CHECK_SEVERITY[name],
        kind="inequality",
        reported=reported,
        expected=bound,
        relative_error=error,
        tolerance=factor,
        ok=reported >= threshold,
    )


def _upper_bound_check(
    name: str,
    reported: float | None,
    bound: float | None,
    tolerance: float,
) -> CheckOutcome:
    """Comprueba ``reported <= bound × (1 + tolerance)``."""
    if reported is None or bound is None:
        return _missing_check(name, reported, bound)
    threshold = bound * (1.0 + tolerance)
    denominator = abs(bound) if bound else 1.0
    error = max(0.0, reported - threshold) / denominator
    return CheckOutcome(
        name=name,
        severity=CHECK_SEVERITY[name],
        kind="inequality",
        reported=reported,
        expected=bound,
        relative_error=error,
        tolerance=tolerance,
        ok=reported <= threshold,
    )


def check_consistency(
    values: Mapping[str, float | None],
    duration_s: float = 60.0,
    tolerance_scale: float = 1.0,
) -> ConsistencyReport:
    """Comprueba la coherencia interna de las features de una ventana.

    Args:
        values: mapping ``feature_name -> value`` en **unidades absolutas**
            (no z-scores). Las features ausentes o no finitas se tratan como
            ``None`` y su comprobación se marca como no evaluable.
        duration_s: duración de la ventana en segundos (necesaria para
            contrastar conteos con tasas).
        tolerance_scale: factor de laxitud aplicado a todas las tolerancias
            (>1 = más permisivo). Con 1.0 los datos limpios del corpus pasan
            el 100 % de las comprobaciones; con 2.0 el ruido de medición del
            ±10 % deja de disparar la compuerta, a costa de perder sensibilidad
            ante corrupciones suaves.

    Returns:
        ``ConsistencyReport`` con una entrada por comprobación.
    """
    duration_s = float(duration_s) if duration_s else 60.0
    duration_min = duration_s / 60.0
    scale = float(tolerance_scale) if tolerance_scale > 0 else 1.0

    def tol(name: str) -> float:
        """Tolerancia de una comprobación de igualdad, escalada por laxitud."""
        return CHECK_TOLERANCES[name] * scale

    def lower_factor(name: str) -> float:
        """Factor de una cota inferior (más laxitud = factor menor)."""
        return CHECK_TOLERANCES[name] / scale

    hr = _number(values, "ecg_hr_bpm")
    rr_mean = _number(values, "ecg_rr_mean_ms")
    n_beats = _number(values, "ecg_n_beats")
    scr_count = _number(values, "eda_scr_count")
    scr_rate = _number(values, "eda_scr_rate_per_min")
    scr_amp_mean = _number(values, "eda_scr_amplitude_mean_us")
    scr_amp_max = _number(values, "eda_scr_amplitude_max_us")
    eda_mean = _number(values, "eda_mean_us")
    eda_tonic = _number(values, "eda_tonic_mean_us")
    eda_std = _number(values, "eda_std_us")
    eda_phasic_std = _number(values, "eda_phasic_std_us")
    resp_rate = _number(values, "resp_rate_bpm")
    resp_cycles = _number(values, "resp_n_cycles")
    temp_range = _number(values, "temp_range_degc")
    temp_std = _number(values, "temp_std_degc")

    checks: list[CheckOutcome] = []

    # 1) HR media vs intervalo RR medio (identidad: HR = 60000 / RR).
    checks.append(
        _identity_check(
            "ecg_hr_vs_rr_mean",
            hr,
            60_000.0 / rr_mean if rr_mean else None,
            tol("ecg_hr_vs_rr_mean"),
        )
    )

    # 2) Número de latidos vs HR × duración (identidad).
    checks.append(
        _identity_check(
            "ecg_n_beats_vs_hr",
            n_beats,
            hr * duration_min if hr is not None else None,
            tol("ecg_n_beats_vs_hr"),
        )
    )

    # 3) Tasa de SCR vs conteo / duración (identidad).
    checks.append(
        _identity_check(
            "eda_scr_rate_vs_count",
            scr_rate,
            scr_count / duration_min if scr_count is not None else None,
            tol("eda_scr_rate_vs_count"),
            abs_tolerance=0.05,
        )
    )

    # 4) Amplitud media de SCR ≤ amplitud máxima (desigualdad estricta).
    checks.append(
        _upper_bound_check(
            "eda_scr_amplitude_mean_vs_max",
            scr_amp_mean,
            scr_amp_max,
            CHECK_TOLERANCES["eda_scr_amplitude_mean_vs_max"] * scale,
        )
    )

    # 5) EDA media vs componente tónica (identidad aproximada).
    checks.append(
        _identity_check(
            "eda_mean_vs_tonic",
            eda_mean,
            eda_tonic,
            tol("eda_mean_vs_tonic"),
        )
    )

    # 6) Ciclos respiratorios vs frecuencia respiratoria × duración (empírica).
    checks.append(
        _identity_check(
            "resp_n_cycles_vs_rate",
            resp_cycles,
            resp_rate * duration_min if resp_rate is not None else None,
            tol("resp_n_cycles_vs_rate"),
        )
    )

    # 7) Rango de temperatura ≥ 2 × desviación estándar (empírica).
    checks.append(
        _lower_bound_check(
            "temp_range_vs_std",
            temp_range,
            temp_std,
            lower_factor("temp_range_vs_std"),
            scale=temp_range,
        )
    )

    # 8) DE de EDA ≥ DE de la componente fásica (empírica).
    checks.append(
        _lower_bound_check(
            "eda_std_vs_phasic_std",
            eda_std,
            eda_phasic_std,
            lower_factor("eda_std_vs_phasic_std"),
            scale=eda_std,
        )
    )

    missing = tuple(name for name in KEY_FEATURES if _number(values, name) is None)
    return ConsistencyReport(checks=tuple(checks), missing_key_features=missing)


def should_abstain(
    report: ConsistencyReport,
    max_hard_failures: int = DEFAULT_MAX_HARD_FAILURES,
    max_soft_failures: int = DEFAULT_MAX_SOFT_FAILURES,
    require_key_features: bool = True,
) -> bool:
    """Decide si la compuerta de consistencia debe forzar la abstención.

    Args:
        report: resultado de ``check_consistency``.
        max_hard_failures: fallos ``hard`` tolerados antes de abstenerse.
        max_soft_failures: fallos ``soft`` tolerados antes de abstenerse.
        require_key_features: si True, la ausencia de cualquier feature clave
            (``KEY_FEATURES``) también fuerza la abstención.

    Returns:
        True si la ventana debe tratarse como evidencia insuficiente.
    """
    if require_key_features and report.missing_key_features:
        return True
    if report.n_hard_failures > max_hard_failures:
        return True
    if report.n_soft_failures > max_soft_failures:
        return True
    return False
