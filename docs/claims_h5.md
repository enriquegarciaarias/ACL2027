# Verificación de afirmaciones (H5/RQ3) y certeza lingüística (H2b)

**Fecha:** 2026-11-09 · **Actualizado:** 2026-11-19 (eje de relevancia
registrado en §5, sin ejecutar)
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

## 5. Relevancia y coherencia con la decisión (eje ausente, registrado 2026-11-19)

### 5.1. La objeción

«Como el prompt le da los valores de las señales, los claims son demasiado
evidentes: el modelo siempre va a acertar.» Es la crítica correcta al alcance
de H5, y conviene contestarla con tres comprobaciones, no con una opinión.

### 5.2. Comprobación 1: no acierta siempre

Recálculo sobre los JSONL del corpus histórico (`degradation/`, 12 sujetos,
2.835 registros; `qwen2.5:7b`, 4 clases, con temperatura), replicando
`verify_claims` claim a claim:

| Condición | Claims | `supported` | `contradicted` | Acierto de dirección |
|---|---:|---:|---:|---:|
| `C0_clean` | 1.853 | 0,781 | **0,219** | 0,858 |
| `C1_partial_resp` | 1.959 | 0,697 | **0,303** | 0,832 |
| `C2_partial_eda` | 2.022 | 0,734 | 0,266 | 0,849 |
| `C4_total_eda` | 1.914 | 0,828 | 0,172 | 0,894 |
| `C5_total_both` | 1.623 | 0,763 | 0,237 | 0,857 |

Uno de cada cinco claims es **falso en limpio**, con el valor a la vista. En
PhysioNet/CinC 2017 (otra tarea, ECG) la tasa de contradichos es **0,476**.

### 5.3. Comprobación 2: no lee el valor más llamativo

| Medida | Valor |
|---|---:|
| Claims por respuesta | 3,31 (máx. 11) |
| Features disponibles por ventana | 17,4 |
| Fracción de la evidencia citada | 19 % |
| Percentil medio de \|z\| de la feature citada | **61,6** |

El percentil se define como la fracción de features con |z| mayor que la
citada: 100 sería «cita siempre la más extrema», 50 es elección al azar. Con
61,6 el modelo **no** está copiando el número más visible; selecciona casi
arbitrariamente entre lo que tiene delante y aun así falla el 22 %.

### 5.4. Comprobación 3: los claims y la decisión se desacoplan

Cruce de cada claim con la decisión tomada, solo respuestas que decidieron
`stress`, usando la firma de estrés declarada en §5.5 (excluye claims sobre
métricas sin firma, p. ej. `resp_*` y `temp_*`):

| Condición | Respuestas `stress` | Claims con firma | Verdadero **y** a favor | Verdadero pero **en contra** |
|---|---:|---:|---:|---:|
| `C0_clean` | 152 | 398 | 0,804 | 0,098 |
| `C4_total_eda` | 404 | 902 | 0,147 | **0,681** |
| `C5_total_both` | 474 | 1.187 | 0,120 | **0,650** |

Con EDA destruida, el modelo decide `stress` y en dos de cada tres
afirmaciones cita, **correctamente**, evidencia que apunta a lo contrario
(«EDA muy disminuida», z ≈ −3). No alucina: narra bien y concluye mal. Es la
forma más nítida encontrada hasta ahora del hallazgo central de H5 («claims
localmente ciertos, inferencia global falsa»).

### 5.5. Por qué el eje falta y cómo se cerraría

Dar los valores explícitos es deliberado: hace el fallo **atribuible a la
inferencia** y no a la percepción (si el modelo viera la señal cruda, un claim
falso sería ambiguo: ¿no lo vio, lo leyó mal o razonó mal?). El precio es que
H5 mide una **cota inferior**: el claim es necesario pero no suficiente. El
verificador actual (`claims.py`) comprueba **verdad de la premisa**, no:

| Eje | Pregunta | Estado |
|---|---|---|
| Verdad | ¿el claim coincide con el valor mostrado? | **implementado** (`supported`/`contradicted`) |
| Relevancia | ¿la evidencia citada favorece la clase predicha? | **implementado** (`claim_coherence`; preregistro en [`h5_relevancia.md`](h5_relevancia.md)) |
| Suficiencia | ¿bastan esos claims para sostener la decisión? | **ausente** |
| Selección | ¿cita lo que apoya y omite lo que contradice? | **ausente** |

La implementación (2026-11-21) está en `sources/phase3/signatures.py`
(tabla congelada) y `sources/phase3/claims.py` (`claim_coherence`,
`response_coherence`, `summarize_coherence`); el analizador expone el bloque
con `--coherence-set {primary,compact,sensitivity}`. Lo que sigue se conserva
como registro de la decisión:

1. **`claims_coherent_with_decision`** en `summarize_verdicts`: tabla de firma
   por clase (dirección esperada de cada feature), aplicada a la decisión
   predicha. Salida: `n_coherent`, `n_incoherent`, `coherence_rate`, y la
   misma tabla por condición y por modelo. La firma es
   [`h5_relevancia.md`](h5_relevancia.md) §3: **12 features primarias**
   (8 EDA, 3 ECG, `resp_rate_bpm`), 10 sin firma y 1 en sensibilidad.
2. **Contraprueba de suficiencia**: repetir la llamada retirando del prompt las
   features citadas por el propio modelo; si la decisión no cambia, los claims
   eran decorativos. Coste: 1 llamada por respuesta, factible sobre S5–S7.
3. **Sesgo de selección**: comparar el percentil de |z| de los claims citados
   contra una línea base de citación aleatoria (ya medido: 61,6 vs 50).

### 5.6. Estado y advertencias

- **Implementado el 2026-11-21**: el eje de relevancia es código con tests
  (`tests/test_signatures.py`, `tests/test_claims.py`) y bloque propio en el
  informe (`coherence`, `signature_check`). Los resultados del **diseño
  primario** siguen pendientes de ejecutar en el servidor; lo de §5.7 es un
  pre-chequeo sobre el corpus histórico y tiene estatus **exploratorio**.
- El corpus usado en §5.2–§5.4 y §5.7 es el **histórico de 4 clases**
  (`degradation/`), no el diseño primario
  (`degradation_binary_schmidt_notemp/`). Esas cifras no deben citarse como
  resultado del experimento primario.
- Pequeña discrepancia de auditoría: `degradation_report.json` declara 1.862
  claims en `C0` y el recálculo da 1.853 (Δ 9; −8/−8/0/−8/−6 en las cinco
  condiciones, todas < 0,5 %). La tasa de respaldo coincide a tres decimales
  (0,7814 vs 0,7809), así que las conclusiones no cambian, pero el informe no
  es reproducible byte a byte desde el corpus actual.

### 5.7. Pre-chequeo P0 y primera lectura con el verificador implementado (exploratorio)

Ejecutado el 2026-11-21 sobre el **corpus histórico de 4 clases** (12 sujetos,
`qwen2.5:7b`, con temperatura, 2.835 respuestas por brazo), no sobre el diseño
primario. Sirve para validar la implementación, no para concluir.

**P0 (¿la firma describe estos datos?).** Mediana de `z` en `C0` por clase:
**11/12** features cumplen el signo preregistrado y P0 **pasa**. La única que
falla es `resp_rate_bpm` (z = −0,16 en `stress` frente a +0,07 en el resto):
con `meditation` (≈9 rpm) dentro del grupo de comparación, el estrés respira
*más lento* que la media. Es coherente con la jerarquía Schmidt: en el diseño
primario `meditation` no forma parte de `non-stress`. No se ha tocado la tabla.

**Coherencia del claim con la decisión** (conjunto primario, 12 features,
bootstrap agrupado por sujeto):

| Condición | Pares | Tasa [IC 95 %] | `a_favor` | Sin confirmación (stress) |
|---|---:|---:|---:|---:|
| `C0_clean` | 1.545 | 0,677 [0,641; 0,713] | 0,270 | 3,9 % |
| `C1_partial_resp` | 1.343 | 0,712 [0,679; 0,736] | 0,219 | 4,0 % |
| `C2_partial_eda` | 1.340 | 0,634 [0,593; 0,678] | 0,317 | 2,2 % |
| `C4_total_eda` | 1.133 | 0,315 [0,285; 0,354] | 0,737 | 33,2 % |
| `C5_total_both` | 1.191 | 0,299 [0,267; 0,332] | 0,825 | 38,8 % |

- En limpio la coherencia no es perfecta (0,68): una de cada tres afirmaciones
  verdaderas no apoya la conclusión.
- Al destruir EDA la coherencia del lado `stress` cae a ≈0,31 y `a_favor` sube
  a 0,74–0,83: el modelo decide `stress` y cita, correctamente, evidencia que
  apunta a lo contrario.
- El lado `non-stress` en `C4`/`C5` tiene < 30 claims decididos (celdas
  `insufficient`): no se publica su tasa.
- El brazo textual (mismo corpus) da 0,652 en `C0` y 0,41 en `C4`/`C5`: misma
  dirección, magnitud parecida.

Estas cifras **no** son el resultado de H5 del experimento primario y no se
citan como tal; la tabla de §5.4 se reproduce aquí con el código nuevo, que es
lo que había que comprobar.

## 6. Limitaciones

- La verificación usa la escala z intra-sujeto; con representaciones absolutas
  hace falta una referencia por sujeto (no implementado; se marca
  `indeterminate`).
- El clipping en ±3 de la estandarización hace que `C4`/`C5` muestren `−3.0 SD`
  (no el valor real, que sería más extremo). Documentado en el análisis.
- Las listas de hedges/boosters son cerradas y requieren validación con
  anotación humana (previsto en H2b).
- `supported` no implica que la decisión sea correcta; son métricas separadas.
- La verificación mide **verdad**, no **relevancia ni suficiencia**: un claim
  verdadero e irrelevante (o contrario a la conclusión) cuenta como
  `supported`. El eje que falta está especificado en §5.

## 7. Reproducción

```bash
# Análisis del barrido (con el barrido en marcha o terminado)
python scripts/analyze_degradation.py

# Solo condiciones concretas
python scripts/analyze_degradation.py --conditions C0_clean,C4_total_eda
```

### 7.1. Corrección de medición (2026-11-23)

`verify_claims` normalizaba el prompt con el vocabulario del dataset pero la
métrica de **cada claim** sin él: todo claim que citara la etiqueta legible que
el modelo había leído («Movement fraction», «Mean conductance») se contaba como
`unsupported` —«métrica ausente»— aunque estuviera en la entrada. Afectaba solo
a los brazos que pasan `extra_aliases` (DREAMT); los análisis de WESAD no los
pasan, así que **ninguna cifra publicada en este documento cambia**.

Para que los dos significados de `unsupported` no vuelvan a mezclarse, el bloque
de claims de DREAMT incluye ahora `unrecognized_metric_names`: cuántos claims
citan un nombre que el verificador no conoce. Un `unsupported` con nombre
reconocido es **evidencia ausente**; sin nombre reconocido es **vocabulario**.
Residual tras ampliar la tabla: 1 claim de 960 llamadas. Detalle en
[`piloto_dreamt.md`](piloto_dreamt.md) §19.2.
