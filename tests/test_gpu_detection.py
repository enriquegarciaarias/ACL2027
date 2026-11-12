"""Tests de detección de GPU (`paramsManager.check_gpu`).

Motivación: en el servidor ``torch`` estaba instalado con un wheel CUDA 13.2 y
el driver solo soportaba 12.8, así que `torch.cuda.is_available()` devolvía
False y el log decía «No CUDA devices available» aunque la GPU existía. El
respaldo por ``nvidia-smi`` estaba declarado pero nunca se ejecutaba (llamaba a
una función inexistente).
"""

from __future__ import annotations

import os

import pytest

from sources.common import paramsManager
from sources.common.common import processControl

A5000 = {
    "index": 0,
    "name": "NVIDIA RTX A5000",
    "memory_gb": 24.0,
    "driver": "570.86.10",
}
TWO_GPUS = [
    {"index": 0, "name": "GPU pequeña", "memory_gb": 8.0, "driver": "570.1"},
    {"index": 1, "name": "GPU grande", "memory_gb": 24.0, "driver": "570.1"},
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(processControl, "defaults", {}, raising=False)


def test_torch_disponible_selecciona_la_mayor(monkeypatch):
    monkeypatch.setattr(
        paramsManager,
        "_torch_gpus",
        lambda: [{"index": 0, "name": "RTX 4060 Ti", "memory_gb": 8.0}],
    )
    paramsManager.check_gpu()
    assert processControl.defaults["device"] == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_torch_elegida_por_memoria(monkeypatch):
    monkeypatch.setattr(paramsManager, "_torch_gpus", lambda: list(TWO_GPUS))
    paramsManager.check_gpu()
    assert processControl.defaults["device"] == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_gpu_visible_solo_por_nvidia_smi_no_oculta_la_gpu(monkeypatch, caplog):
    """Caso del servidor: torch no puede usar CUDA pero la GPU existe."""
    monkeypatch.setattr(paramsManager, "_torch_gpus", lambda: [])
    monkeypatch.setattr(
        paramsManager, "_detect_all_gpus_nvidia_smi", lambda: [A5000]
    )
    paramsManager.check_gpu()

    # La GPU existe: no se pone CUDA_VISIBLE_DEVICES=-1 (rompería procesos hijos).
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is None
    # torch no puede usarla, así que el dispositivo declarado es CPU.
    assert processControl.defaults["device"] == "cpu"


def test_sin_gpu_se_usa_cpu_y_se_oculta(monkeypatch):
    monkeypatch.setattr(paramsManager, "_torch_gpus", lambda: [])
    monkeypatch.setattr(paramsManager, "_detect_all_gpus_nvidia_smi", lambda: [])
    paramsManager.check_gpu()
    assert processControl.defaults["device"] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_detect_all_gpus_parsea_nvidia_smi(monkeypatch):
    salida = "0, NVIDIA RTX A5000, 24564, 570.86.10\n1, NVIDIA RTX 4090, 24564, 570.86.10"
    monkeypatch.setattr(paramsManager, "_nvidia_smi", lambda _query: salida)
    gpus = paramsManager._detect_all_gpus_nvidia_smi()
    assert [gpu["index"] for gpu in gpus] == [0, 1]
    assert gpus[0]["name"] == "NVIDIA RTX A5000"
    assert gpus[0]["memory_gb"] == pytest.approx(24564 / 1024, rel=1e-6)
    assert gpus[0]["driver"] == "570.86.10"


def test_detect_all_gpus_sin_nvidia_smi(monkeypatch):
    def _falla(_query):
        raise RuntimeError("nvidia-smi no disponible")

    monkeypatch.setattr(paramsManager, "_nvidia_smi", _falla)
    assert paramsManager._detect_all_gpus_nvidia_smi() == []


def test_detect_gpus_filtra_por_memoria(monkeypatch):
    monkeypatch.setattr(
        paramsManager, "_detect_all_gpus_nvidia_smi", lambda: list(TWO_GPUS)
    )
    assert paramsManager._detect_gpus_nvidia_smi(10.0) == [TWO_GPUS[1]]


def test_torch_gpus_sin_torch(monkeypatch):
    monkeypatch.setattr(paramsManager, "_torch_module", lambda: None)
    assert paramsManager._torch_gpus() == []


def test_gpu_report_sin_gpu(monkeypatch):
    monkeypatch.setattr(paramsManager, "_torch_module", lambda: None)
    monkeypatch.setattr(paramsManager, "_detect_all_gpus_nvidia_smi", lambda: [])
    report = paramsManager.gpu_report()
    assert report["usable"] is None
    assert report["device"] == "cpu"
    assert report["mismatch"] is False


def test_gpu_report_mismatch_torch_driver(monkeypatch):
    monkeypatch.setattr(paramsManager, "_torch_gpus", lambda: [])
    monkeypatch.setattr(
        paramsManager, "_detect_all_gpus_nvidia_smi", lambda: [A5000]
    )
    report = paramsManager.gpu_report()
    assert report["usable"] == "nvidia-smi"
    assert report["mismatch"] is True
    assert report["device"] == "cpu"
    assert "Ollama" in report["reason"]


def test_gpu_report_con_torch(monkeypatch):
    monkeypatch.setattr(
        paramsManager,
        "_torch_gpus",
        lambda: [{"index": 0, "name": "RTX 4060 Ti", "memory_gb": 8.0}],
    )
    report = paramsManager.gpu_report()
    assert report["usable"] == "torch"
    assert report["mismatch"] is False
    assert report["device"] == "cuda"
