# P3 — Validación externa con K-EmoCon: qué se puede y qué no se puede validar

**Fecha:** 2026-11-09
**Adaptador:** `sources/external/kemocon.py`
**Script:** `scripts/run_p3_kemocon.py`
**Datos:** tabla de features de PSRI (`kemocon_feature_table.csv`) +
`kemocon_period_assignments.csv`

---

## 1. Corrección de enfoque: K-EmoCon no mide estrés

K-EmoCon **no tiene etiqueta de estrés**. Sus constructos son **valencia y
arousal** (autoinforme, pareja y observadores externos) y solo están disponibles
en el periodo `debate`.

Un primer enfoque (archivado en
`results/output/KEMOCON/P3/archive_pre_vs_debate/`) trató `pre` vs `debate`
como `non-stress` vs `stress`. **Se descarta como validación**: `pre` no es un
periodo «sin actividad» (es el registro continuo durante la preparación) y el
contraste `pre → debate` confunde habla, movimiento, carga social, anticipación
y orden temporal. Detectar `debate` no equivale a detectar estrés.

Este documento reporta lo que sí se puede medir con K-EmoCon.

## 2. Datos y adaptación

| Aspecto | WESAD | K-EmoCon |
|---|---|---|
| Sujetos | 15 | 23 |
| Sensores | ECG/EDA/resp/temp de pecho | EDA/BVP/temp/IBI/ACC de muñeca |
| Respiración | sí | no |
| SCR (conteo/amplitud) | sí | no |
| Constructo | condición de protocolo (Trier) | valencia/arousal |
| Ventana nativa | — | 5 s → agregada a 60 s |

Procesado (sin rehacer la ingesta, reutilizando la tabla de PSRI):

1. Unión con `kemocon_period_assignments.csv` (`pre` 2.582 / `debate` 2.946
   ventanas de 5 s).
2. Agregación a 60 s (12 subventanas): medias para medias y RMS para
   dispersiones; se descarta la ventana con cobertura de EDA < 50 %.
   Resultado: **445 ventanas** (207 `pre`, 238 `debate`).
3. Traducción a features canónicas: `ecg_hr_bpm` (BVP/Polar), `ecg_rr_mean_ms`
   (IBI), `ecg_sdnn_ms` (**proxy** de `ibi_std`), `eda_mean_us`, `eda_std_us`,
   `temp_mean_degc`, `temp_std_degc`. Nada se inventa: respiración, SCR y
   latidos quedan ausentes.
4. Estandarización robusta intra-participante (E12, mediana/IQR, recorte ±3).
5. Etiquetas afectivas binarizadas por **mediana por participante** (principal)
   o mediana global (sensibilidad), solo sobre `debate`.

## 3. Referencia features-only: la señal afectiva es casi inexistente

Regresión logística (`class_weight="balanced"`, imputación por mediana,
estandarización) con **CV agrupada por participante** (5 pliegues), usando la
misma representación (z intra-sujeto) que ve el LLM:

| Etiqueta | Split | n | Balanced accuracy | AUROC |
|---|---|---:|---:|---:|
| `external_arousal` | por participante | 237 | **0,563 ± 0,083** | **0,609** |
| `external_valence` | por participante | 237 | 0,576 ± 0,048 | 0,587 |
| `partner_arousal` | por participante | 237 | 0,527 ± 0,033 | 0,583 |
| `self_arousal` | por participante | 237 | 0,545 ± 0,079 | 0,527 |
| `self_arousal` | global | 237 | 0,528 ± 0,057 | 0,552 |
| `external_arousal` | global | 237 | 0,512 ± 0,141 | 0,512 |
| `self_valence` | por participante | 237 | 0,436 ± 0,040 | 0,417 |
| `external_valence` | global | 237 | 0,458 ± 0,101 | 0,430 |

**Ninguna etiqueta afectiva supera AUROC ≈ 0,61.** Con EDA/HR/IBI/temperatura
de muñeca en ventanas de 60 s, el arousal y la valencia autoinformados de
K-EmoCon **no son predecibles**: no hay señal que sirva de referencia para
validar al LLM.

### 3.1. Contraste con el contraste de protocolo (archivado)

El contraste `pre` vs `debate` **sí** es detectable con las mismas features
(balanced 0,708 ± 0,093; AUROC 0,785). La diferencia es informativa: la
fisiología de muñeca sigue la **tarea/actividad** (hablar, moverse, preparar),
no el estado afectivo autoinformado. Es exactamente el motivo por el que
`pre`/`debate` no debe presentarse como detección de estrés.

## 4. Piloto LLM (cordura, no validación)

Prompt de arousal (`high-arousal` / `low-arousal`) sobre `external_arousal`
(la etiqueta con más señal), 4 participantes, 44 ventanas de `debate`:

| Sistema | Balanced | Accuracy | Precision | Recall | AUROC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Referencia features-only (z, CV por participante, mismos 4 sujetos) | 0,507 | — | — | — | 0,628 | — | — |
| **LLM (arousal)** | **0,614** | 0,614 | 0,632 | 0,545 | **0,642** | 0,232 | 0,286 |

- 0 inválidas y 0 abstenciones; confianza 85 en 42/44 respuestas (saturación de
  P4 otra vez).
- Confusión: 12/22 `high-arousal` correctas, 15/22 `low-arousal` correctas.
- El LLM queda **al mismo nivel que la referencia débil** (AUROC 0,64 vs 0,63)
  y muy lejos de WESAD (0,90–0,96). Con n = 44 el intervalo de confianza es
  amplio: no se puede afirmar que aporte información.
- En el corpus completo (23 participantes) la referencia es 0,563 ± 0,083 y
  AUROC 0,609, así que el techo de la tarea con estas features es bajo.

Interpretación: **no hay evidencia de que el LLM lea el arousal autoinformado**
en K-EmoCon. Tampoco lo hace la referencia: la tarea no es resoluble con
EDA/HR/IBI/temperatura de muñeca a 60 s.

## 5. Conclusión y decisión

- **K-EmoCon no permite validar el endpoint de estrés** (no tiene etiqueta de
  estrés) **ni el afectivo** (la señal no es predecible con estas features).
  Queda como **limitación documentada**, no como resultado positivo.
- Lo que sí aporta, y es relevante:
  1. El contraste `pre` vs `debate` es detectable (AUROC 0,785) pero mide
     **tarea/actividad** (hablar, moverse, preparar), no estrés. Presentarlo
     como detección de estrés sería incorrecto.
  2. Las etiquetas afectivas reales (arousal/valencia) **no** se reflejan en
     EDA/HR/IBI/temperatura de muñeca a 60 s (AUROC ≤ 0,61).
  3. El LLM no supera a la referencia: no hay evidencia de lectura fisiológica
     fina, en línea con P2/P4.
- Para validez externa real del endpoint se necesita un dataset con
  **inducción de estrés** y sensores comparables (UBFC-Phys, SWELL-KW) o
  features más ricas (respiración, SCR) que K-EmoCon no tiene.

## 6. Limitaciones

- Confusor temporal y de actividad en `pre`/`debate` (por eso se descarta).
- Etiquetas afectivas solo en `debate`; no hay periodo de calma etiquetado.
- Autoinformes discretos (escala 1-5) y con varianza entre participantes alta.
- La agregación a 60 s puede diluir la respuesta afectiva (no se probó 30 s).
- `ecg_sdnn_ms` es un proxy (`ibi_std` agregado por RMS).
- Un solo dataset externo, con sensores distintos a WESAD.

## 7. Reproducción

```bash
# Barrido de referencia (sin LLM)
python scripts/run_p3_kemocon.py --task arousal --source external_arousal \
  --reference --reference-only

# Piloto LLM de cordura (4 participantes)
python scripts/run_p3_kemocon.py --subjects 1,5,8,10 --task arousal \
  --source external_arousal --reference
```

Salidas:

- `results/output/KEMOCON/P3/kemocon_arousal.responses.jsonl` (sin etiquetas)
- `results/output/KEMOCON/P3/kemocon_arousal.labels.jsonl`
- `results/output/KEMOCON/P3/kemocon_arousal_summary.json`
- `results/output/KEMOCON/P3/archive_pre_vs_debate/` (enfoque descartado)
