"""
Validaciones reutilizables para entradas y artefactos.

El objetivo es fallar pronto y con mensajes que digan *qué* falló, *dónde* y
*con qué valores*, en lugar de propagar ``KeyError``/``IndexError`` opacos.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


class ValidationError(ValueError):
    """Error de validación con contexto estructurado."""

    def __init__(self, message: str, *, where: Optional[str] = None):
        self.where = where
        super().__init__(f"[{where}] {message}" if where else message)


def require(condition: bool, message: str, *, where: Optional[str] = None) -> None:
    """Lanza ``ValidationError`` si ``condition`` es falsa."""
    if not condition:
        raise ValidationError(message, where=where)


def require_keys(
    mapping: Mapping[str, Any],
    required: Iterable[str],
    *,
    where: str,
) -> None:
    """Comprueba que ``mapping`` contiene todas las claves requeridas."""
    require(isinstance(mapping, Mapping), f"se esperaba un mapping, hay {type(mapping)!r}", where=where)
    missing = sorted(set(required) - set(mapping.keys()))
    if missing:
        raise ValidationError(
            f"faltan claves obligatorias: {missing}; presentes: {sorted(mapping.keys())}",
            where=where,
        )


def check_array(
    name: str,
    array: Any,
    *,
    ndim: Optional[int] = None,
    min_len: int = 1,
    finite: bool = True,
    allow_nan: bool = False,
) -> np.ndarray:
    """Valida forma y finitud de un array y lo devuelve como ``np.ndarray``."""
    arr = np.asarray(array)
    require(arr.size > 0, "array vacío", where=name)
    require(arr.shape[0] >= min_len, f"longitud {arr.shape[0]} < {min_len}", where=name)
    if ndim is not None and arr.ndim != ndim:
        raise ValidationError(f"se esperaba ndim={ndim}, hay ndim={arr.ndim} shape={arr.shape}", where=name)
    if finite:
        if allow_nan:
            bad = np.isinf(arr).sum()
        else:
            bad = (~np.isfinite(arr)).sum()
        require(int(bad) == 0, f"{int(bad)} valores no finitos", where=name)
    return arr


def check_same_length(
    arrays: Mapping[str, np.ndarray],
    *,
    where: str,
) -> int:
    """Comprueba que todos los arrays tienen la misma primera dimensión."""
    require(bool(arrays), "no hay arrays que comparar", where=where)
    lengths = {name: int(arr.shape[0]) for name, arr in arrays.items()}
    unique = set(lengths.values())
    if len(unique) != 1:
        raise ValidationError(f"longitudes inconsistentes: {lengths}", where=where)
    return unique.pop()


def range_report(
    name: str,
    array: np.ndarray,
    lo: float,
    hi: float,
    *,
    unit: str,
    strict: bool = False,
) -> dict:
    """Comprueba el rango plausible de un canal y devuelve un informe.

    Args:
        strict: si es ``True``, un valor fuera de rango lanza excepción; si no,
            se registra en el informe para que el QC decida.
    """
    arr = np.asarray(array, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValidationError("no hay valores finitos", where=name)
    vmin = float(finite.min())
    vmax = float(finite.max())
    violations = int(np.count_nonzero((finite < lo) | (finite > hi)))
    report = {
        "channel": name,
        "unit": unit,
        "expected_range": [float(lo), float(hi)],
        "observed_min": vmin,
        "observed_max": vmax,
        "violations": violations,
        "fraction_violations": float(violations / finite.size),
    }
    if strict and violations:
        raise ValidationError(
            f"{violations} valores fuera de [{lo}, {hi}] {unit} "
            f"(min={vmin:.4g}, max={vmax:.4g})",
            where=name,
        )
    return report


def iterable_to_list(value: Any, *, where: str) -> Sequence[Any]:
    """Normaliza un valor escalar o iterable a lista."""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if value is None:
        return []
    raise ValidationError(f"no se puede interpretar como lista: {value!r}", where=where)
