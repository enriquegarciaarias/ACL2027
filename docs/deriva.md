# Deriva de la decisión bajo degradación

Capa de **mecanismo** del proyecto: no añade una hipótesis nueva (H1–H6 siguen
siendo el *qué*), sino que explica el *porqué* del desplazamiento de la
respuesta cuando la evidencia se degrada o desaparece.

Los informes de degradación (`analyze_degradation.py`) miden **cada condición
por separado**: dicen que en `C4` la accuracy cae de 0,638 a 0,315 y que se
predice mucho más `stress`. Eso deja tres preguntas abiertas que solo se
responden **por ventana**:

1. ¿Hacia qué clase se desplaza la respuesta?
2. ¿Cuánto del error final lo ha **inducido** la degradación y cuánto ya
   existía en limpio?
3. ¿El desplazamiento se explica por falta de información o por la **política de
   decisión**?

La tercera se contesta con un control: el **mismo análisis sobre la referencia
supervisada**, que ve exactamente las mismas features y la misma
estandarización intra-sujeto.

- Módulo: `sources/phase3/drift.py` (puro: sin LLM, sin sklearn).
- Informe: `scripts/analyze_drift.py`.
- CLI: `python main.py --proc EVALUATE --experiment degradation --drift …`.
- Tests: `tests/test_drift.py` (24).

---

## 1. Los seis indicadores y por qué no se usa «Δ respecto al equilibrio»

| Indicador | Qué añade | Dónde se calcula |
|---|---|---|
| **Marginal predicha + tasa de reclamación** `P(predicha)/P(real)` | Mide sesgo en la escala correcta | `prediction_marginal` |
| **Matriz pareada `C0 → Cx`** por ventana | Qué ventanas cambian y hacia dónde | `transition_analysis` |
| **Descomposición 2×2** bien/mal antes × después | Separa error inducido de preexistente | `_decomposition` |
| **Destino por clase real** | A qué clase van las que aciertan en limpio y fallan | `_by_truth_class` |
| **Flujos de abstención** | Decidir↔abstenerse al degradar | `abstain_flow` |
| **Control con la referencia** | ¿Deriva la información o la política? | `table_from_predictions` |

### Por qué se descarta `Δ_r = desviación respecto al balance de clases`

El borrador inicial proponía medir la deriva como distancia de la marginal
predicha al reparto uniforme. Es **mis-especificado**: en el diseño primario el
prior real es `stress` 0,2995 / `non-stress` 0,7005, así que un clasificador
perfecto **no** predice 50/50. La métrica castigaría al modelo correcto y, de
hecho, `C0` (la mejor condición, accuracy 0,915) sería la más «desviada».

La lectura correcta es la **tasa de reclamación**:

```
claim_rate(clase) = P(predicha = clase | decididas) / P(real = clase)
```

`> 1` sobrerreclama, `< 1` infra-reclama, `= 1` respeta el prior. Va siempre
junto a **balanced accuracy** y **cobertura**, porque la accuracy bruta premia
al clasificador que colapsa a la clase mayoritaria.

Tampoco se usa G-Mean (redundante con balanced) ni conformal por clase: con
≈10 ventanas de estrés por sujeto un umbral por clase no es estimable. El
conformal sigue siendo **global** con validación cruzada por sujeto
(`docs/p4_calibracion.md`).

---

## 2. Método

### 2.1 Tabla de decisiones

`build_decision_table(records, labels, …)` produce, por ventana:

```python
{"truth": "stress", "subject": "S5",
 "decisions": {"C0_clean": "stress", "C4_total_eda": "non-stress"},
 "confidence": {...}, "n_reps": {...}}
```

Reglas:

- La ventana se identifica por su **id opaco** (`win_…`); la etiqueta real se
  une desde `INGEST/groundtruth/labels.jsonl` y nunca entra en el prompt.
- **Tarea**: `affective` (4 clases) o `stress_binary` (`stress`/`non-stress`).
  En binario, las etiquetas fuera de la variante (`meditation` con `schmidt`)
  se **descartan y se cuentan**, no se reetiquetan.
- **Estado**: por defecto solo `valid_response` y `valid_abstention`, es decir,
  la decisión **del modelo**. La compuerta de consistencia
  (`gate_abstention`) se excluye salvo `include_gate=True`, porque decide por
  regla y no por el LLM.
- **Repeticiones**: si hay varias filas por (ventana, condición) se agrega por
  **voto mayoritario** (empate → `abstain`) y se guarda `n_reps`. La varianza
  entre repeticiones es otro experimento (`docs/repetitions.md`).

### 2.2 Matriz pareada y descomposición

Para cada condición `Cx` se toman los pares con decisión en `C0` y en `Cx`:

| | acierta en `Cx` | falla en `Cx` |
|---|---|---|
| **acierta en `C0`** | `correct_correct` | **`correct_wrong`** (error inducido) |
| **falla en `C0`** | `wrong_correct` (reparado) | `wrong_wrong` (error preexistente) |

- `induced_error_share = correct_wrong / (correct_wrong + wrong_wrong)`.
- `preexisting_error_share = wrong_wrong / (…)`.
- `repaired_share = wrong_correct / (wrong_correct + wrong_wrong)`.

El **error inducido** es la parte que la degradación crea; el **preexistente**,
la que el modelo ya tenía sin degradación. Sin esta separación, una caída de
accuracy puede atribuirse a la degradación cuando en realidad es una
incapacidad previa que simplemente se mantiene.

### 2.3 La referencia como control

`table_from_predictions` convierte las predicciones **fuera de pliegue**
(regresión logística con CV agrupada por sujeto, sobre las mismas features y la
misma referencia de estandarización que ve el LLM) en una tabla idéntica. Las
mismas funciones producen su marginal, su matriz y su descomposición.

Lectura:

- Si la referencia **no** deriva y el LLM **sí** → la deriva es de la **política
  de decisión**, no de la información disponible.
- Si la referencia deriva en la misma dirección → falta información (o las
  features no contienen lo necesario).

La referencia no es «el techo del LLM en la misma tarea»: es un clasificador
supervisado que ve la misma evidencia. Sirve para distinguir «no hay
información» de «el modelo no la usa».

### 2.4 Cruce con claims (RQ3)

En la tarea binaria, `scripts/analyze_drift.py` (con el cruce de claims **activo
por defecto**; `--no-claims` lo desactiva) agrupa las respuestas
en **derivan** / **estables correctas** / **estables erróneas** y agrega:

- `coherence_rate = confirmaciones / (confirmaciones + contrarios)`;
- `against_supported_share`: claims **verdaderos** que apuntan **contra** la
  decisión (la forma más nítida del hallazgo de H5);
- `zero_confirmation_share`: respuestas sin ningún claim que confirme la clase
  predicha;
- `evidence_conflict_rate`.

Se apoya en `sources/phase3/signatures.py` (tabla de firma congelada) y
`claims.response_coherence`. Solo se define para `stress`/`non-stress`: no se
inventa una firma de 4 clases.

### 2.5 Incertidumbre

El **sujeto** es la unidad de generalización. `transition_analysis` guarda
`by_subject` con los recuentos por sujeto y el script calcula el IC 95 % del
error inducido con **bootstrap agrupado** (2.000 remuestreos por defecto). Una
tasa sobre 424 ventanas de 12 personas no es un intervalo de 424 observaciones.

---

## 3. Cómo se lee: patrón → causa → prueba discriminante

| Patrón observado | Causa compatible | Prueba que lo discrimina |
|---|---|---|
| Todo se mueve hacia `stress`; las ventanas de `stress` no se mueven | Umbral desplazado: degradar EDA baja la evidencia de calma y basta para cruzar | `by_truth`: `stress` con 0 errores y `non-stress` con casi todos |
| La referencia deriva en direcciones **difusas**, el LLM en **una** | Política de decisión, no información | Misma matriz sobre referencia (`--no-reference` la quita) |
| `abstain` en limpio → `stress` al degradar | La duda se resuelve hacia la clase positiva | `abstain_flow`: `abstain_to_stress` alto |
| Deriva solo cuando se destruye **EDA** | Canal crítico | Comparar `C1`/`C2` (respiración/parcial) con `C4`/`C5` |
| Deriva con claims **contra** la decisión | Evidencia local correcta, inferencia global errónea | `against_supported_share` en el grupo `drifted` |
| Cobertura que cae a la vez que el error | Retirada, no reasignación | `abstain_flow` + `changed_share` |

Los dos modos de fallo ya documentados en `docs/conclusiones.md` (resultado 2)
se distinguen aquí con una sola tabla: **reasignar** (7b: balanced 0,72 pero
accuracy 0,55) frente a **retirarse** (14b: cobertura 0,856 pero balanced 0,52).

---

## 4. Preliminar exploratorio (corpus legado, 4 clases)

> **No es el diseño primario.** Corpus `results/output/WESAD/PHASE2/degradation/`
> (12 sujetos, 567 ventanas, `qwen2.5:7b`, **4 clases y temperatura incluida**),
> anterior a la jerarquía `schmidt` y al split congelado. Sirve para validar el
> instrumento, no para citar resultados.

Ejecución:

```bash
python scripts/analyze_drift.py \
  --dir results/output/WESAD/PHASE2/degradation \
  --decision-mode affective --temperature include \
  --bootstrap 500 --output /tmp/drift_legacy.json
```

### 4.1 Marginal y tasa de reclamación

Prior real: `baseline` 0,404 · `stress` 0,225 · `amusement` 0,119 ·
`meditation` 0,252.

| Condición | `baseline` | `stress` | tasa `stress` |
|---|---:|---:|---:|
| `C0_clean` | 0,692 | 0,308 | 1,37 |
| `C2_partial_eda` | 0,694 | 0,306 | 1,37 |
| `C4_total_eda` | **0,010** | **0,990** | **4,41** |
| `C5_total_both` | **0,002** | **0,998** | **4,45** |

### 4.2 Matriz y descomposición (`C0 → C4`)

359 pares decididos; **220 cambian** (61,3 %). Los 220 son
`baseline → stress`: no hay ni un `stress → baseline`.

| | acierta en `C4` | falla en `C4` |
|---|---:|---:|
| **acierta en `C0`** | 109 | **120** |
| **falla en `C0`** | 4 | 126 |

- Accuracy 0,638 → 0,315; error inducido **0,488** [0,424; 0,549].
- Por clase real: `stress` 111 ventanas, **0 errores**; `non-stress` 129, **127
  errores, los 127 a `stress`** — de ellas, 120 acertaban en `C0`.
- Abstención: 116 `non-stress → abstain`, 47 `abstain → stress`,
  24 `abstain → abstain`, 17 `stress → abstain`.

### 4.3 La referencia sobre las mismas features

| | LLM | Referencia |
|---|---:|---:|
| Accuracy `C0 → C4` | 0,638 → **0,315** | 0,817 → **0,746** |
| Pares que cambian | **220** (61,3 %) | 112 (19,8 %) |
| Dirección dominante | `baseline → stress` (**220**) | `baseline → amusement` (29), `baseline → meditation` (17) |
| `baseline → stress` | **120** ventanas | **4** ventanas |
| Error inducido | 0,488 [0,424; 0,549] | 0,479 [0,285; 0,624] |

**Lo discriminante no es el error inducido** (parecido) sino **la dirección y el
volumen**: la referencia pierde 7 puntos de accuracy moviéndose entre clases de
forma difusa; el LLM pierde 32 puntos moviendo *todas* las ventanas de calma a
`stress`. Con la misma evidencia. Eso es política de decisión.

### 4.4 Un aviso de lectura obligatorio

En este corpus el LLM **nunca predice `amusement` ni `meditation`**: su espacio
efectivo es `baseline`/`stress`/`abstain`. Su deriva *solo puede* ir hacia
`stress` o a la abstención, así que comparar su matriz con la de una referencia
de 4 clases no es simétrico. El informe lo declara
(`label_space_note` / `predicted_classes_ever`).

Es exactamente el argumento a favor de la **jerarquía `schmidt`**: con la tarea
binaria, LLM y referencia comparten espacio de decisión y la comparación es
limpia. Por eso el análisis definitivo se hace sobre
`degradation_binary_schmidt_notemp/{modelo}`.

---

## 5. Ejecución definitiva (servidor)

El nodo de login `sirius` aborta al importar numpy (`Illegal instruction`), así
que los análisis van en un nodo de cálculo:

```bash
srun --pty bash
cd ~/data/proyectos/ACL2027 && source .venv/bin/activate

# WESAD primario, cuatro brazos
for model in qwen2.5:7b qwen3:14b; do
  for rep in numeric text; do
    python main.py --proc EVALUATE --experiment degradation --drift \
      --model "$model" --representation "$rep" \
      --decision-mode binary --nonstress-variant schmidt --temperature exclude \
      --bootstrap 2000 \
      --output "results/output/WESAD/PHASE2/drift_${model//[:.]/_}_${rep}.json"
  done
done
```

Sin `--drift` el despachador sigue usando `analyze_degradation.py`; con
`--drift --dry-run` el plan muestra `runner: scripts/analyze_drift.py`, la
condición base y las llamadas estimadas (0: **no hace ninguna llamada al LLM**).

### 5.1 Preflight antes de citar cifras

```bash
# 1) el plan resuelve el runner y la condición base
python main.py --proc EVALUATE --experiment degradation --drift \
  --model qwen3:14b --representation numeric --decision-mode binary \
  --nonstress-variant schmidt --temperature exclude --dry-run

# 2) existe el corpus de ese brazo y con qué motor se generó
ls results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp/qwen3_14b
cat results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp/qwen3_14b/run_provenance.json
```

El directorio se resuelve con `resolve_degradation_dir` (misma función que el
barrido y los demás analizadores): **no** se escribe la ruta a mano. Sin
`--decision-mode` prefiere el corpus binario y cae al de 4 clases; el modo
efectivo se deduce después de la procedencia de cada respuesta.

### 5.2 Qué leer en el informe

| Campo | Interpretación |
|---|---|
| `marginal.conditions.<Cx>.claim_rate` | `> 1` sobrerreclama la clase; se lee con `predicted_decided`, `n_decided` y `abstain_share` (y con `accuracy_balanced`/`coverage` del mismo corpus en `analyze_degradation.py`) |
| `transitions.conditions.<Cx>.matrix_decided` | qué ventanas cambian y hacia dónde (pares con decisión en base y en `Cx`) |
| `…decomposition.induced_error_share` + `induced_error_ci.<Cx>` | qué parte del error final crea la degradación (IC bootstrap por sujeto) |
| `…by_truth.<clase>.drifted_destination` | destino de las que acertaban en limpio |
| `…abstain_flow` | retirada (`→ abstain`) frente a reasignación (`abstain → clase`) |
| `reference.*` | las mismas tablas para la referencia; si no deriva y el LLM sí, la causa es la **política** |
| `claims.<Cx>.<drifted\|stable_correct\|stable_wrong>` | `coherence_rate`, `against_supported_share`, `zero_confirmation_share` (solo binario; `--no-claims` lo omite) |
| `label_space_note` / `predicted_classes_ever` | aviso cuando el modelo nunca predice alguna clase del espacio |

Regla de lectura: ninguna conclusión se afirma con `induced_error_share` solo;
se cita junto a `accuracy_after`, `changed_share` y el tamaño del brazo.

Después, DREAMT (segunda pasada). **Pendiente de implementación**: `drift.py`
soporta hoy `affective` y `stress_binary`; la tarea de sueño (`sleep4`) requiere
añadir su espacio de decisión, su tabla de firma (ya existe `SLEEP_SIGNATURES` en
`sources/phase3/signatures.py`) y su mapeo de etiquetas. El script lo rechaza con
un `ValueError` explícito en lugar de producir tablas silenciosamente vacías.

---

## 6. Limitaciones

1. **Necesita corpus con condición base y degradada del mismo modelo, motor y
   prompt.** Mezclar motores (Ollama 0.40.6 en servidor, 0.30.7 en local)
   cambia las decisiones: ya obligó a una guardia en el barrido.
2. **El espacio efectivo de etiquetas condiciona la lectura** (§4.4). Se
   declara por corpus.
3. **El cruce de claims solo está definido en binario.** Una firma de 4 clases
   para WESAD no está validada; no se inventa.
4. **La referencia es un control, no un oráculo**: un logreg con 23 features no
   agota la información de la señal; si el LLM deriva y la referencia no, la
   conclusión es sobre política, no sobre optimalidad.
5. **Bootstrap por sujeto con 12 sujetos**: el IC es honesto pero ancho. No se
   afirma significación con n=12 sin mirar el intervalo.
6. **Composición ≠ significación**: `induced_error_share` alto puede convivir
   con pocos errores. Se lee junto a `accuracy_after` y `changed_share`.

---

## 7. Relación con los demás documentos

- `docs/experimentos.md` — inventario E0–E12; el mecanismo se cita en E1.
- `docs/resultados_jerarquia.md` — resultados H1–H6 del diseño primario; aquí se
  explica el *porqué* de H1 y H5.
- `docs/conclusiones.md` — resultado 2 («varias formas de fallar») y resultado 3
  (claims locales verdaderos); la deriva los cuantifica.
- `docs/h5_relevancia.md` — eje de relevancia de H5.
- `docs/claims_h5.md` — verificación de claims.
- `docs/repetitions.md` — la varianza de muestreo es un eje **distinto** de la
  deriva: aquí se agrega la respuesta por ventana, allí se mide cuánto cambia
  consigo misma.
