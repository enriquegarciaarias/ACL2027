# Verificación de afirmaciones (H5/RQ3) y certeza lingüística (H2b)

**Fecha:** 2026-11-09
**Módulos:** `sources/phase3/claims.py`, `sources/phase3/certainty.py`
**Script:** `scripts/analyze_degradation.py`
**Reporte:** `results/output/WESAD/PHASE2/degradation_report.json`

---

## 1. Qué se mide

H5 pregunta si las afirmaciones del LLM están respaldadas por la evidencia que
recibió, y cómo cambia esa proporción con la degradación. H2b pregunta si la
certeza del lenguaje (hedges/boosters) sigue a la evidencia o se desacopla.

Ambas medidas son **deterministas y auditables**, y no necesitan etiquetas:
se calculan contra el propio prompt (lo que el modelo vio) y contra el texto de
razonamiento.

## 2. Verificación de claims

Cada claim (`metric`, `channel`, `direction`, `magnitude`) se contrasta con la
feature correspondiente **parseada del prompt** (`- nombre: valor unidad`).

### 2.1. Categorías

| Categoría | Regla |
|---|---|
| `supported` | dirección compatible y magnitud dentro de ±1 nivel |
| `contradicted` | dirección incompatible, o magnitud a ≥2 niveles |
| `unsupported` | la métrica no aparece en la entrada de esa ventana/condición |
| `indeterminate` | la métrica aparece con `not available`, o la escala es absoluta sin referencia |

### 2.2. Intervalos de compatibilidad (z intra-sujeto)

| Dirección | Compatible si |
|---|---|
| `increase` | `z ≥ +0.2` |
| `decrease` | `z ≤ −0.2` |
| `stable` | `|z| ≤ 0.5` |

Los intervalos **se solapan a propósito**: una desviación de +0,3 SD es
compatible tanto con `increase` como con `stable`. Solo se marca
`contradicted` cuando la dirección declarada es incompatible con el valor
(p. ej. `increase` con `z = −0.8`, o `stable` con `z = 1.5`). Sin esta banda,
claims razonables como «slightly decreased» con `z = −0,26` se contarían como
alucinaciones.

### 2.3. Magnitud

`|z| < 1` → `low`; `1 ≤ |z| < 2` → `medium`; `|z| ≥ 2` → `high`. Para los
claims `stable` no se evalúa la magnitud (`magnitude_ok = None`), porque no
aporta información discriminante.

## 3. Certeza lingüística

`linguistic_certainty(reasoning)` cuenta marcadores cerrados:

- **Hedges** (cautela): *might, may, could, possibly, probably, suggests,
  appears, approximately, relatively, unclear, insufficient…*
- **Boosters** (asertividad): *clearly, obviously, certainly, strongly,
  markedly, consistently, proves, confirms, evident…*

Índice = `(boosters − hedges) / palabras × 100`. Positivo = lenguaje asertivo;
negativo = cauteloso.

## 4. Resultados preliminares (S5, barrido en curso)

Con ~50 % de las respuestas de S5 (una sola ventana de evaluación completada
por condición), el pipeline funciona y ya muestra las tendencias que H5/H2b
predicen. **No son resultados finales** (n = 20 por condición; el barrido tiene
567 ventanas × 5 condiciones).

| Condición | Claims | supported | contradicted | certainty index |
|---|---:|---:|---:|---:|
| `C0_clean` | 72 | 0,778 | 0,222 | +0,54 |
| `C1_partial_resp` | 80 | 0,787 | 0,212 | +0,04 |
| `C2_partial_eda` | 77 | 0,714 | 0,286 | +0,31 |
| `C4_total_eda` | 77 | 0,844 | 0,156 | **−1,10** |
| `C5_total_both` | 59 | 0,746 | 0,254 | **−1,70** |

Dos observaciones relevantes, a confirmar con el corpus completo:

1. **Respaldo y corrección son ejes distintos.** En `C4_total_eda` la accuracy
   cae a 0, pero el respaldo de claims **sube** (0,844): EDA está en el extremo
   bajo (z ≈ −3), así que los claims de «EDA disminuida» son literalmente
   ciertos aunque la inferencia del estado sea errónea. H5 debe reportarse
   junto a H1, no en su lugar.
2. **La certeza lingüística sí reacciona a la degradación de EDA** (índice
   +0,54 → −1,10 → −1,70), mientras que en respiración apenas cambia. Es la
   primera señal de que el modelo modula el lenguaje con la evidencia, aunque
   de forma desigual por modalidad.

## 5. Limitaciones

- La verificación usa la escala z intra-sujeto; con representaciones absolutas
  hace falta una referencia por sujeto (no implementado; se marca
  `indeterminate`).
- El clipping en ±3 de la estandarización hace que `C4`/`C5` muestren `−3.0 SD`
  (no el valor real, que sería más extremo). Documentado en el análisis.
- Las listas de hedges/boosters son cerradas y requieren validación con
  anotación humana (previsto en H2b).
- `supported` no implica que la decisión sea correcta; son métricas separadas.

## 6. Reproducción

```bash
# Análisis del barrido (con el barrido en marcha o terminado)
python scripts/analyze_degradation.py

# Solo condiciones concretas
python scripts/analyze_degradation.py --conditions C0_clean,C4_total_eda
```
