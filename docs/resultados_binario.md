# Endpoint binario bajo degradación: `qwen2.5:7b` vs `qwen3:14b`

Resultados del barrido con **prompt binario** (`stress` / `non-stress`) sobre las
mismas 567 ventanas de evaluación (S5–S17 salvo S12) y las cinco condiciones
efectivas de degradación. Cierra la matriz **2 modelos × 2 tareas × 5
condiciones** y contiene los resultados más citables de la línea.

- Artefactos: `results/output/WESAD/PHASE2/degradation_binary/{modelo}/`
- Informes: `degradation_report.json` (14b) y `degradation_report_legacy7b.json`
  (7b, 4 clases) en `results/output/WESAD/PHASE2/`
- Comparador: `results/output/WESAD/PHASE2/model_comparison.json`
- Reproducción: §8.

> **Nota de jerarquía (2026-11-16).** Las cifras de este documento se
> calcularon sobre **567 ventanas** con la variante `all` (`non-stress` =
> baseline + amusement + meditation) y con **temperatura** en la evidencia. No
> son la tarea binaria de Schmidt (`baseline` + `amusement`): se reportan como
> **"stress vs all non-stress"**. El experimento primario es **424 ventanas,
> variante `schmidt`, sin temperatura**; sus resultados se añadirán tras la
> re-ejecución. La jerarquía y las sensibilidades se analizan con
> `scripts/analyze_binary_hierarchy.py` (bootstrap agrupado por sujeto). Ver
> [`experimento_binario_jerarquia.md`](experimento_binario_jerarquia.md).

---

## 1. Qué se midió

| | 7b | 14b |
|---|---|---|
| Prompt | binario (`stress`/`non-stress`/`abstain`) | binario (idéntico) |
| Respuestas | 2.835 (567 × 5) | 2.835 |
| Errores de llamada | 0 | 0 |
| Inválidas de formato | 52 (1,8 %) → **52 recuperadas** | 0 |
| Condiciones | C0, C1, C2, C4, C5 | C0, C1, C2, C4, C5 |

Diseño congelado: split de evaluación, referencia de estandarización calculada
**solo con C0** de cada sujeto, few-shot de S2–S4, misma semilla y mismos
parámetros de generación. Lo único que cambia entre columnas es el modelo.

> Corrección aplicada durante el análisis: las 52 inválidas del 7b eran
> vocabulario de claims (`claim[3] falta campo 'magnitude'` en 50 casos). El
> validador acepta ya `magnitude` opcional y la recuperación usa el modo de
> decisión correcto (antes intentaba recuperar `stress`/`non-stress` contra el
> vocabulario de 4 clases). Sin esa corrección, el 7b aparecía con 521 decididas
> en C0 en vez de 567.

---

## 2. Resultados por condición

| Cond. | Modelo | cob | acc | bal | prec | rec | ECE | conf | #conf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 7b | 1,000 | 0,914 | 0,897 | 0,775 | 0,866 | 0,066 | 85,0 | 2 |
| C0 | 14b | 1,000 | **0,928** | 0,881 | **0,871** | 0,795 | 0,113 | 84,3 | 7 |
| C1 | 7b | 1,000 | 0,910 | **0,908** | 0,747 | 0,906 | 0,063 | 85,0 | 3 |
| C1 | 14b | 1,000 | 0,926 | 0,882 | 0,857 | 0,803 | 0,122 | 84,8 | 7 |
| C2 | 7b | 1,000 | 0,894 | **0,867** | 0,738 | 0,819 | 0,056 | 84,9 | 2 |
| C2 | 14b | 0,824 | 0,955 | 0,756 | **1,000** | 0,512 | 0,127 | 75,6 | 7 |
| C4 | 7b | 1,000 | 0,547 | **0,630** | 0,302 | **0,780** | 0,303 | 84,9 | 2 |
| C4 | 14b | 0,945 | 0,821 | 0,500 | 0,000 | 0,000 | 0,099 | 84,2 | 5 |
| C5 | 7b | 0,998 | 0,495 | **0,613** | 0,285 | **0,827** | 0,355 | 84,9 | 3 |
| C5 | 14b | 0,919 | 0,845 | 0,500 | 0,000 | 0,000 | 0,094 | 83,9 | 6 |

Referencia features-only (mismas 23 features, logreg con CV por sujeto):
`ref_bin` = 0,923 / 0,907 / 0,932 / 0,877 / 0,849. **Todas las condiciones son
resolubles**: incluso en C5 queda información (HR del ECG).

Δ accuracy del 14b sobre el 7b (bootstrap agrupado por sujeto, IC 95 %):
C0 +0,014 [−0,012; +0,040]; C1 +0,016 [−0,018; +0,048]; C2 +0,061
[+0,041; +0,080]; C4 +0,274 [+0,189; +0,361]; C5 +0,350 [+0,235; +0,463].

---

## 3. Hallazgo 1 — en limpio están empatados

Los intervalos de C0 y C1 **cruzan cero**. La ventaja del 14b que se había
observado en la tarea de 4 clases (0,703 vs 0,611) era, sobre todo, la
abstención del 7b (12,5 %), no una mejor discriminación.

Se reparten el perfil: el 7b tiene más recall (0,866 vs 0,795) y el 14b más
precisión (0,871 vs 0,775). **El tamaño no compra acierto en limpio; compra
calibración** (§6).

---

## 4. Hallazgo 2 — bajo destrucción fallan en direcciones opuestas

Con la EDA a cero:

- **7b**: sigue decidiendo y sobrerreclama estrés → recall 0,780/0,827,
  precisión 0,302/0,285, balanced **0,630/0,613**, accuracy 0,547/0,495.
- **14b**: dice `non-stress` para todo → precisión y recall 0, balanced
  **0,500/0,500**, accuracy 0,821/0,845.

Ninguno es robusto: los dos están en la vecindad del azar. La diferencia es
**en qué prior caen**. Y ese prior depende de la tarea, no solo del modelo:

| Tarea | 7b | 14b |
|---|---|---|
| 4 clases, EDA destruida | colapso a `stress` (baseline recall 0,01) | abstención 96–100 % |
| binario, EDA destruida | sobrerreclama `stress` (rec 0,78–0,83) | colapso a `non-stress` (rec 0) |

Mecanismo propuesto y comprobado en los dos sentidos: **la abstención es el
residuo del espacio de etiquetas**. En 4 clases no hay clase comodín y el modelo
usa `abstain`; en binario, `non-stress` absorbe toda la ambigüedad y la
abstención desaparece (0 % en C0/C1 del 7b; 0–17,6 % en el 14b). El prompt
binario **también** contiene la regla «if the evidence is insufficient, set
decision to abstain»: la diferencia no es la instrucción, es que existe un
destino para la duda.

---

## 5. Hallazgo 3 — la accuracy premia al clasificador degenerado

Accuracy a **cobertura comparable** (`compare_models.py`):

| Cond. | cob | 7b | 14b |
|---|---:|---:|---:|
| C0 | 1,000 | 0,914 (567) | 0,928 (567) |
| C1 | 1,000 | 0,910 (567) | 0,926 (567) |
| C2 | 0,824 | 0,921 (467) | 0,955 (467) |
| C4 | 0,945 | 0,556 (536) | **0,821** (536) |
| C5 | 0,919 | 0,512 (521) | **0,845** (521) |

En C4/C5 el 14b gana por accuracy incluso igualando cobertura… porque 0,845 es
exactamente la proporción de `non-stress` del test. **Es un clasificador
constante.** El balanced lo desenmascara: 0,500 = azar, mientras el 7b, con peor
accuracy (0,512), conserva balanced 0,613.

Consecuencia metodológica: en un test desbalanceado, **la accuracy de un sistema
que ha dejado de detectar la clase de interés sube**. Cualquier evaluación de
robustez a la degradación debe reportar balanced y recall por clase junto a la
cobertura; la accuracy sola invierte la conclusión.

---

## 6. Hallazgo 4 — dos sabores de mala calibración (H2a)

| | 7b | 14b |
|---|---|---|
| Confianza media (C4/C5) | 84,9 / 84,9 | 84,2 / 83,9 |
| Niveles distintos | 2–3 | 5–7 |
| AUROC confianza (C0 → C5) | 0,510 → 0,507 | 0,792 → 0,748 |
| Sobreconfianza (C4/C5) | **+0,303 / +0,354** | +0,021 / −0,006 |
| ECE (C4/C5) | 0,303 / 0,355 | 0,099 / 0,094 |

- 7b: **sobreconfiado y equivocado**. Confianza prácticamente constante (85),
  inútil como score (AUROC ≈ 0,50).
- 14b: **calibrado y degenerado**. ECE 0,094 con un clasificador que responde
  0,845 porque esa es la tasa base.

`ECE` y `Brier` miden la calibración de la probabilidad declarada, no la utilidad
de la decisión. Reportarlos sin balanced y cobertura invita a la conclusión
contraria a los datos.

---

## 7. Hallazgo 5 — H6: el indicador de calidad **sustituye** a la confianza

ΔAUROC al añadir el SQI sobre la confianza del modelo:

| Cond. | 7b (conf. constante) | 14b (conf. informativa) |
|---|---:|---:|
| C0 | **+0,163** | +0,043 |
| C1 | **+0,171** | −0,054 |
| C2 | +0,071 | — |
| C4 | **+0,243** | — |
| C5 | +0,082 | — |

La compuerta de consistencia (P1) no aporta en ninguna condición (Δgate = 0):
mide integridad interna, no información.

Formulación para la tesis: **el SQI aporta en la medida en que la confianza del
modelo no informa**. Con el 7b (AUROC 0,50) recupera hasta +0,24; con el 14b
(AUROC 0,79 en limpio) casi nada. No es «el SQI mejora al modelo», es «el SQI
sustituye una señal de confianza ausente».

---

## 8. H2b y H5 en la matriz completa

**Certeza lingüística (índice hedges−boosters por 100 palabras, solo decididas):**

| | C0 | C1 | C2 | C4 | C5 | Lectura |
|---|---:|---:|---:|---:|---:|---|
| 7b, 4 clases | +0,11 | −0,22 | −0,34 | −0,67 | −0,99 | monótono |
| 14b, 4 clases | −0,59 | −0,63 | — | −0,84 | — | plano (veto asertivo lo enmascaraba) |
| 7b, binario | −0,40 | −1,04 | −0,65 | −0,55 | −0,62 | **plano** |
| 14b, binario | −1,10 | −1,67 | −1,14 | −2,59 | −3,04 | **monótono** |

**H2b queda en «depende de modelo × tarea»**: aparece en el 7b con 4 clases y en
el 14b con binario, y no en las otras dos combinaciones. No es una propiedad
general del lector, y la abstención la contamina cuando domina el texto
(14b, 4 clases: `índ_abs` +0,51 en C4 frente a `índ_dec` −0,84).

**Respaldo de claims (H5) sí replica en los cuatro brazos:**

| | C0 | C4 | C5 | dir_acc |
|---|---:|---:|---:|---:|
| 7b, binario | 0,771 | 0,820 | 0,767 | 0,83–0,89 |
| 14b, binario | **0,864** | **0,915** | **0,918** | 0,95–0,96 |
| 7b, 4 clases | 0,781 | 0,828 | 0,762 | 0,86–0,89 |
| 14b, 4 clases | 0,799 | 0,879 | 0,860 | 0,93–0,96 |

El respaldo **sube** cuando la corrección se hunde, y el 14b acierta
sistemáticamente más la dirección de sus afirmaciones. En C4/C5 binario se da la
disociación triple: claims correctos y mejor respaldados (0,915/0,918), lenguaje
más cauto (índice −2,6/−3,0) y decisión al azar (balanced 0,500).

---

## 9. Limitaciones y desviaciones

1. **La tarea de 4 clases es la preregistrada**; la binaria es una desviación
   declarada (ya lo era en P2/P4). El barrido binario se añadió después de ver
   que el modelo nunca predice `amusement` ni `meditation`.
2. **Dos modelos, uno por familia**: no se puede separar efecto de tamaño de
   efecto de familia. `qwen3:8b` está instalado y permitiría aislar el tamaño
   (experimento pendiente, 2 condiciones ≈ 2 h).
3. **C2 es el único punto con abstención informativa** en binario (14b: 17,6 %
   con precisión 1,000 y recall 0,512). Con una sola condición no se puede
   trazar una curva riesgo–cobertura fiable.
4. **La calibración se mide sobre decididas**; con coberturas distintas entre
   modelos (C2: 1,000 vs 0,824) el ECE no es estrictamente comparable entre
   filas.
5. **Un solo prompt por tarea**: no se ha hecho ablación de la regla de
   abstención en binario (el prompt la incluye en ambos casos).

---

## 10. Reproducción

```bash
# Barrido binario (GENERATE necesita GPU y Ollama: siempre con el lanzador)
~/data/launch_lsi.sh --proc run_ACL.sh --time 12:00:00 -- --proc GENERATE \
    --experiment degradation --subject WESAD --decision-mode binary --model qwen3:14b
~/data/launch_lsi.sh --proc run_ACL.sh --time 12:00:00 -- --proc GENERATE \
    --experiment degradation --subject WESAD --decision-mode binary --model qwen2.5:7b

# Análisis (sin GPU, terminal SSH)
python main.py --proc EVALUATE --experiment degradation --subject WESAD \
    --model qwen3:14b --decision-mode binary
python main.py --proc EVALUATE --experiment degradation --subject WESAD \
    --model qwen2.5:7b --decision-mode binary
python scripts/compare_models.py --models qwen2.5:7b,qwen3:14b --decision-mode binary
```

Los artefactos binarios viven en `degradation_binary/{modelo}/` y **no se
mezclan** con el corpus de 4 clases (`degradation/`). Para un smoke barato:
`--subjects S5 --conditions C0_clean,C4_total_eda --max-windows 1` (2 llamadas;
`--max-windows` es por sujeto, no global).
