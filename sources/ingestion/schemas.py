"""
Contratos de datos (esquemas) del pipeline de ingesta.

Los artefactos se versionan con ``schema_version``. Cualquier cambio
incompatible debe incrementar esa versión y documentarse; los consumidores
(prompting, evaluación) no deben aceptar versiones que no conozcan.

Nota metodológica: las etiquetas viven en ``groundtruth/`` y nunca forman
parte de ``RepresentationRecord``. Esta separación se comprueba en tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "1.0"


@dataclass
class SubjectEntry:
    """Entrada de un sujeto en el manifiesto del dataset."""

    subject: str
    file: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    selected: bool = True
    readme: Optional[str] = None
    quest: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SubjectEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetManifest:
    """Manifiesto reproducible del dataset descubierto."""

    dataset: str
    root: str
    algorithm: str
    generated_at: str
    config_fingerprint: str
    subjects: list[SubjectEntry] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = asdict(self)
        data["subjects"] = [s.to_dict() if isinstance(s, SubjectEntry) else s for s in self.subjects]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetManifest":
        subjects = [SubjectEntry.from_dict(s) for s in data.get("subjects", [])]
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "subjects"}
        return cls(subjects=subjects, **payload)


@dataclass
class UnitReport:
    """Resultado de la verificación de unidades de un canal."""

    channel: str
    unit: str
    expected_range: list[float]
    observed_min: float
    observed_max: float
    violations: int
    fraction_violations: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CanonicalMeta:
    """Metadatos del array canónico de un sujeto."""

    subject: str
    source_file: str
    source_sha256: str
    canonical_file: str
    canonical_sha256: str
    schema_version: str
    created_at: str
    fingerprint: str
    shapes: dict
    dtypes: dict
    units: dict
    sampling_rates: dict
    n_samples: int
    duration_s: float
    label_counts: dict
    unit_reports: list[UnitReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["unit_reports"] = [
            u.to_dict() if isinstance(u, UnitReport) else u for u in self.unit_reports
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalMeta":
        reports = [UnitReport(**u) for u in data.get("unit_reports", [])]
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "unit_reports"}
        return cls(unit_reports=reports, **payload)


@dataclass
class WindowRecord:
    """Ventana temporal etiquetada (H3)."""

    window_id: str
    subject: str
    label: dict
    t_start_s: float
    t_end_s: float
    ranges: dict
    purity: float
    guard_applied_s: float
    source: dict
    config_fingerprint: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureValue:
    """Valor de una característica con su procedencia.

    ``kind`` distingue el origen metodológico: ``observed`` (estadístico
    descriptivo de la señal), ``derived`` (cálculo fisiológico), ``quality``
    (indicador de calidad) o ``error`` (no calculable).
    """

    value: Any
    unit: str
    method: str
    method_version: str
    params: dict = field(default_factory=dict)
    kind: str = "derived"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureRecord:
    """Características y calidad de una ventana (H4)."""

    window_id: str
    subject: str
    features: dict
    quality: dict
    config_fingerprint: str
    feature_version: str = "1.0"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepresentationRecord:
    """Representación de una ventana lista para el LLM (H5).

    No contiene etiquetas: la etiqueta se consulta en ``groundtruth/``.
    """

    window_id: str
    subject: str
    representation: str
    representation_version: str
    payload: Any
    config_fingerprint: str
    feature_version: str = "1.0"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)
