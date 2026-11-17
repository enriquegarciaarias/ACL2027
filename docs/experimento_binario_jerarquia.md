# Experimento binario: jerarquía de clases y temperatura

**Fecha de fijación:** 2026-11-16 (antes de re-ejecutar).
**Estado:** especificación cerrada; implementación completa
(`sources/phase3/binary_hierarchy.py`, `scripts/analyze_binary_hierarchy.py`,
flags `--nonstress-variant` y `--temperature`). **Fase 2 completa** (2.835
respuestas por modelo, `qwen2.5:7b` y `qwen3:14b`) y **clasificación evaluada**:
resultados en [`resultados_jerarquia.md`](resultados_jerarquia.md). El análisis
del resto de hipótesis (H1, H2a, H2b, H4, H5, H6) se ejecuta con
`scripts/analyze_degradation.py`, que ya resuelve esta variante y la ausencia de
temperatura.

---

## 1. Pregunta

El endpoint primario de la tesis es binario: `stress` vs `non-stress`. El paper
original de WESAD (Schmidt et al., 2018) define `non-stress` como
`baseline` + `amusement` y **no** usa `meditation` como clase afectiva (es una
fase de de-excitación). La implementación anterior del prompt incluía
`meditation` en `non-stress`, lo que **no** era la tarea de Schmidt.

Además, la temperatura de pecho es un confusor verificado del orden del
protocolo (r = 0,801) y la **única modalidad de pecho que no se degrada**. Si el
modelo la usa, puede amortiguar C1/C2/C4/C5 y atenuar el efecto de degradación
(H1). Este documento fija, antes de re-ejecutar:

1. Qué variantes binarias se reportan y cuál es el primario.
2. Por qué el primario excluye la temperatura.
3. Cómo se calculan las métricas y la incertidumbre.
4. Qué reglas de decisión se aplican y qué desviaciones se declaran.

---

## 2. Estimando y variantes

El estimando es **P(stress | evidencia)** con la etiqueta real de WESAD. Las
variantes cambian únicamente qué clases forman `non-stress`:

| Variante | `non-stress` | n (evaluación) | Prevalencia stress | Tasa base |
|---|---|---:|---:|---:|
| `baseline` | baseline | 355 | 0,358 | 0,642 |
| **`schmidt`** | **baseline + amusement** | **424** | **0,300** | **0,700** |
| `no_amusement` | baseline + meditation | 498 | 0,255 | 0,745 |
| `all` | baseline + amusement + meditation | 567 | 0,224 | 0,776 |

Datos de evaluación: 12 sujetos (S5–S17 salvo S12); 228 `baseline`, 127
`stress`, 69 `amusement`, 143 `meditation`. Restringido a las tres clases de
Schmidt, la distribución es 53,8 / 29,8 / 16,3 %, prácticamente la del paper
(53 / 30 / 17).

**Tasa base** = accuracy de un clasificador constante que siempre predice
`non-stress`. Se reporta porque accuracy, ECE y Brier dependen de la prevalencia
y cambian entre variantes. `balanced accuracy` y AUROC son los ejes robustos.

---

## 3. Jerarquía declarada

1. **Primario:** `schmidt` (424 ventanas), **sin temperatura**.
   - Excluye `meditation` por **constructo** (Schmidt no la usa como clase
     afectiva), no porque el modelo falle en ella.
   - Conserva `amusement`: es parte de Schmidt y ocupa posiciones distintas del
     protocolo (antes del stress en la versión A, después en la B), lo que
     introduce `non-stress` frío y caliente y **debilita parcialmente** la regla
     "temperatura alta = stress".
2. **Sensibilidades:** `baseline` (355), `no_amusement` (498) y `all` (567).
   - No se eligen por acierto; se reportan todas.
   - Si coinciden en dirección y significación con el primario, la conclusión es
     robusta a la composición de clases.
   - Si divergen, la divergencia es un hallazgo sobre la heterogeneidad de
     `non-stress` y se reporta como tal.
3. **Sensibilidad de temperatura:** mismo `schmidt`, mismo prompt, **con**
   temperatura. Cuantifica cuánto influye el confusor. No es candidata a
   primario.

La comparación con Schmidt es de **composición de clases**, no de rendimiento:
su 93 % es de clasificadores supervisados con LOSO y otras ventanas; el LLM
zero-shot no replica ese benchmark.

---

## 4. Temperatura: por qué fuera del primario

1. **Confusor verificado:** r = 0,801 entre posición del protocolo y
   temperatura; 14/15 sujetos con |r| > 0,5.
2. **Única modalidad no degradada:** C1 degrada respiración, C2/C4 degradan EDA,
   C5 ambas. `temp` queda intacta. Si codifica la fase, el modelo puede seguir
   infiriendo estrés cuando EDA/resp están destruidas y **atenuar H1**.
3. **No pertenece al constructo:** la temperatura de pecho refleja la
   trayectoria térmica de la sesión, no la respuesta aguda de estrés.
4. **El few-shot ya la omitía**, pero la evidencia la incluía: la inconsistencia
   se resuelve excluyéndola de la evidencia primaria.

**Qué se conserva:** `temp` sigue en el corpus canónico y en las features, para
que `scripts/resolvability_analysis.py` pueda **cuantificar** el confusor. La
separación es: el LLM primario ve `ecg`, `eda` y `resp`; el análisis conserva
`temp`.

**Confusor residual:** quitar `temp` es necesario pero no suficiente. La
clasificación absoluta sigue teniendo confusor de orden; se controla con el
contraste within-window C0→C5, la residualización por fase y el bootstrap
agrupado por sujeto. Se declara.

---

## 5. Métricas e incertidumbre

Por condición (C0, C1, C2, C4, C5; C3 ≡ C1) y por variante:

- n, prevalencia de stress, tasa base.
- accuracy, balanced accuracy, precisión/recall/F1 de `stress`, especificidad.
- AUROC de `stress` (score derivado de la confianza).
- ECE y Brier (dependen de la prevalencia; se reportan con la tasa base).
- Cobertura y tasa de abstención.

**Incertidumbre:** IC 95 % por **bootstrap agrupado por sujeto** (2.000
remuestreos por defecto; el sujeto es la unidad de generalización, no la
ventana). Las ventanas de un mismo sujeto están correlacionadas; remuestrear
ventanas subestimaría los intervalos. No se afirma que "la potencia no es un
problema": el n efectivo para generalizar es ≈ 12 sujetos.

**Claims y certeza (H2b/H5):** se calculan con las mismas funciones del corpus
(`claims.py`, `certainty.py`) y se reportan por variante y condición.

---

## 6. Reglas de decisión

- La conclusión principal se basa en **`schmidt` sin temperatura**.
- La robustez se evalúa con las sensibilidades. Si todas coinciden en dirección
  y los IC se solapan sustancialmente, se concluye que el efecto no depende de
  la composición de clases. Si no, se reporta la divergencia.
- La diferencia entre `schmidt` sin y con temperatura **es** la estimación del
  confusor; se reporta como limitación.
- **Nunca** se cambia el primario porque una sensibilidad dé resultados más
  favorables.
- Las cifras de 4 clases y las del binario `all` de 567 se mantienen como
  secundarias/exploratorias y **no se mezclan** con el primario.

---

## 7. Desviaciones declaradas

1. El **endpoint binario** (2026-11-09) fue post hoc (se tomó tras ver el
   colapso del modelo en 4 clases).
2. La **jerarquía y la exclusión de temperatura** (2026-11-16) son una segunda
   desviación post hoc. Se fijan **antes** de re-ejecutar y se reportan las
   cuatro variantes.
3. La exclusión de `meditation` se justifica por constructo; la de temperatura,
   por confusor de orden y modalidad no degradada. Ninguna se justifica por el
   resultado del modelo.

---

## 8. Artefactos y comandos

### 8.1 Fase 1 (local, ya implementada, sin LLM)

```bash
# Análisis de la jerarquía sobre respuestas existentes (si las hay)
python scripts/analyze_binary_hierarchy.py --model qwen3:14b

# Sobre el corpus de 4 clases (exploratorio): reinterpreta decisiones
python scripts/analyze_binary_hierarchy.py \
  --dir results/output/WESAD/PHASE2/degradation --from-affective
```

Salida: `results/output/WESAD/PHASE2/evaluation_binary_hierarchy.json` y tablas
por condición y variante.

### 8.2 Fase 2 (servidor, LLM)

Primario sin temperatura, todos los modelos:

```bash
~/data/launch_lsi.sh --proc run_ACL.sh --time 12:00:00 -- \
  --proc GENERATE --experiment degradation --subject WESAD \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --model qwen3:14b
```

Repetir con `--model qwen2.5:7b`. El barrido ejecuta **567 ventanas × 5
condiciones = 2.835 llamadas por modelo** (5.670 en total). Se ejecutan las 567
(no solo las 424) porque las sensibilidades 498 y 567 necesitan las ventanas de
`meditation`. A ~4,7 s/llamada (14b) ≈ 7,4 h para los dos modelos; el 7b puede
ser más lento. Es reanudable por `(ventana, condición)`.

Salida: `results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp/{modelo}/`.

Sensibilidad de temperatura (mismo prompt, un modelo):

```bash
~/data/launch_lsi.sh --proc run_ACL.sh --time 12:00:00 -- \
  --proc GENERATE --experiment degradation --subject WESAD \
  --decision-mode binary --nonstress-variant schmidt --temperature include \
  --model qwen3:14b
```

Salida: `results/output/WESAD/PHASE2/degradation_binary_schmidt/{modelo}/`.

### 8.3 Fase 3 (evaluación)

```bash
python main.py --proc EVALUATE --experiment degradation --model qwen3:14b \
  --hierarchy --bootstrap 2000 \
  --output results/output/WESAD/PHASE2/evaluation_binary_hierarchy_qwen3_14b.json
```

Para la sensibilidad con temperatura, indicar el directorio explícito:

```bash
python scripts/analyze_binary_hierarchy.py \
  --dir results/output/WESAD/PHASE2/degradation_binary_schmidt/qwen3_14b \
  --model qwen3:14b --bootstrap 2000 \
  --output results/output/WESAD/PHASE2/evaluation_binary_hierarchy_temp_qwen3_14b.json
```

---

## 9. Qué se reportará en el manuscrito

1. Tabla de las cuatro variantes por condición (n, prevalencia, tasa base,
   accuracy, balanced, AUROC, ECE, Brier, IC por sujeto).
2. Primario `schmidt` sin temperatura; sensibilidades en el apéndice o en la
   misma tabla con una marca.
3. Comparación sin/con temperatura como estimación del confusor.
4. Declaración explícita de que la comparación con Schmidt es de composición de
   clases, no de rendimiento, y de que H1–H5 se recalculan en el primario.
