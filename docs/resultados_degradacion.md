# Resultados del barrido de degradación (H1, H2, H4, H5, H6)

**Fecha:** 2026-11-10
**Corpus:** 12 sujetos de **evaluación** (S5–S17 salvo S12), 567 ventanas ×
5 condiciones = **2.835 respuestas** (7 inválidas, 0,25 %).
**Representación:** numérica E12 (23 features de pecho), referencia de
estandarización calculada solo con C0 limpio.
**Tarea:** clasificación afectiva de 4 clases (preregistrada).
**Modelo:** `qwen2.5:7b`, T=0,3, seed 42, few-shot de S2–S4.
**Reporte:** `results/output/WESAD/PHASE2/degradation_report.json`

---

## 1. Corrección y calibración (H1, H2a)

| Condición | Ref. 4 clases | Ref. binaria | LLM accuracy | LLM balanced | Abstención | ECE | Conf. media | Sobreconfianza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0_clean` | 0,828 | 0,923 | 0,611 | 0,478 | 12,5 % | 0,190 | 76,7 | +0,156 |
| `C1_partial_resp` | 0,723 | 0,907 | 0,571 | 0,462 | 9,5 % | 0,250 | 79,4 | +0,223 |
| `C2_partial_eda` | 0,826 | 0,932 | 0,601 | 0,464 | 20,5 % | 0,201 | 74,5 | +0,144 |
| `C4_total_eda` | 0,702 | 0,877 | **0,289** | 0,253 | 28,0 % | **0,489** | 71,0 | **+0,421** |
| `C5_total_both` | 0,625 | 0,849 | **0,263** | 0,251 | 16,2 % | **0,512** | 73,4 | **+0,471** |

- **Coste en exceso** (Δ LLM − Δ referencia 4 clases): la LLM cae 0,322 puntos
  de C0 a C4 mientras la referencia cae 0,126 → **exceso de 0,20**. De C0 a C5:
  0,348 vs 0,203 → **exceso de 0,15**. La degradación de EDA destruye al LLM
  mucho más de lo que destruye la información (H1 apoyada).
- **La pérdida de respiración es casi gratuita para el LLM** (0,611 → 0,571,
  −0,040) y más costosa para la referencia (0,828 → 0,723). El modelo no usa la
  respiración de forma apreciable.
- **Calibración**: ECE se multiplica por 2,6 con EDA degradada (0,190 → 0,489) y
  la sobreconfianza pasa de +0,16 a +0,47.
- **H2a**: la confianza apenas discrimina aciertos en limpio (AUROC 0,511) y
  **sube** con degradación severa (0,733 en C4, 0,755 en C5). Es decir, la
  confianza detecta que la entrada está degradada, no qué ventanas concretas se
  responden bien.

## 2. Abstención y resolubilidad (H4)

Todas las condiciones son **resolubles** para la referencia (4 clases: 0,625–0,828,
muy por encima de 0,55), incluida C5.

| Condición | Resoluble | Error del LLM (1 − acc) | Abstención | Brecha |
|---|---|---:|---:|---:|
| `C0_clean` | sí | 0,389 | 0,125 | +0,264 |
| `C1_partial_resp` | sí | 0,429 | 0,095 | +0,334 |
| `C2_partial_eda` | sí | 0,399 | 0,205 | +0,194 |
| `C4_total_eda` | sí | 0,711 | 0,280 | **+0,431** |
| `C5_total_both` | sí | 0,737 | 0,162 | **+0,575** |

- **Infra-abstención** clara en las condiciones degradadas: en C5 el modelo
  falla el 74 % de las veces y solo se abstiene el 16 %.
- **Sobre-abstención** en limpio: 12,6 % de abstenciones cuando la tarea es
  resoluble (0,828 de referencia).
- La abstención **no sigue la resolubilidad**: sigue (parcialmente) la severidad
  de la degradación, que es justo el criterio que el marco prohíbe usar.

## 3. Respaldo de afirmaciones (H5)

| Condición | Claims | Respaldadas | Contradichas | No respaldadas | Indeterminadas | Acierto dirección |
|---|---:|---:|---:|---:|---:|---:|
| `C0_clean` | 1.862 | 0,781 | 0,219 | 0,000 | 0,000 | 0,858 |
| `C1_partial_resp` | 1.967 | 0,697 | 0,303 | 0,000 | 0,000 | 0,832 |
| `C2_partial_eda` | 2.022 | 0,734 | 0,266 | 0,000 | 0,000 | 0,849 |
| `C4_total_eda` | 1.922 | **0,828** | 0,172 | 0,000 | 0,000 | 0,894 |
| `C5_total_both` | 1.629 | 0,762 | 0,238 | 0,000 | 0,000 | 0,857 |

**Hallazgo central**: el respaldo de los claims **no cae** con la degradación;
incluso sube en C4 (0,828) mientras la accuracy se hunde (0,289). El motivo es
que los claims son **localmente ciertos** — «EDA muy disminuida» es verdad con
EDA = 0 (z ≈ −3) — pero la **inferencia global** es incorrecta. Además, no hay
claims sobre canales ausentes (`unsupported` = 0): el modelo **deja de citar** la
respiración y el SCR cuando desaparecen.

Implicación para RQ3: la fidelidad de las afirmaciones no es una medida de
corrección. Es un eje independiente y debe reportarse junto a H1.

## 4. Certeza lingüística (H2b)

| Condición | Hedges | Boosters | Índice | Palabras |
|---|---:|---:|---:|---:|
| `C0_clean` | 0,39 | 0,43 | +0,13 | 30,0 |
| `C1_partial_resp` | 0,41 | 0,29 | −0,27 | 30,5 |
| `C2_partial_eda` | 0,53 | 0,37 | −0,51 | 30,7 |
| `C4_total_eda` | 0,58 | 0,41 | −0,74 | 30,3 |
| `C5_total_both` | 0,71 | 0,41 | −1,12 | 29,8 |

- El índice **desciende de forma monótona** con la severidad de la degradación:
  +0,13 → −1,12 (H2b apoyada en dirección).
- Los **hedges suben** (0,39 → 0,71) y los boosters se mantienen (0,43 → 0,41):
  el modelo añade cautela sin retirar asertividad, exactamente el
  desacoplamiento que predice H2b.
- La respuesta es **asimétrica por modalidad**: la degradación de respiración
  apenas mueve el índice (−0,27) frente a EDA (−0,74).

## 5. Indictador externo de calidad (H6)

AUROC de la confianza para predecir corrección, y AUROC al añadir un indicador
externo (CV agrupada por sujeto):

| Condición | Confianza | + Gate (P1) | Δ Gate | + SQI | Δ SQI |
|---|---:|---:|---:|---:|---:|
| `C0_clean` | 0,477 | 0,480 | +0,003 | 0,540 | **+0,063** |
| `C1_partial_resp` | 0,566 | 0,566 | +0,000 | 0,621 | **+0,055** |
| `C2_partial_eda` | 0,439 | 0,439 | +0,000 | 0,491 | **+0,052** |
| `C4_total_eda` | 0,744 | 0,746 | +0,002 | 0,831 | **+0,087** |
| `C5_total_both` | 0,763 | 0,763 | +0,000 | 0,809 | **+0,046** |

- **La compuerta de consistencia (P1) no aporta nada aquí** (Δ ≈ 0): detecta
  incoherencia estructural, y los datos degradados son coherentes (EDA = 0 es un
  valor internamente consistente). Es un indicador de **integridad**, no de
  **información**.
- El **SQI** sí aporta de forma consistente (+0,05 a +0,09 AUROC), y más donde
  más falta hace (C4). Es el indicador externo que H6 necesita.

## 6. H3 — Efecto de la representación (numérica vs textual)

Mismos 23 valores z y misma tarea, en dos formatos: JSON numérico
(`numeric_llm`) y resumen textual con etiquetas legibles (`text_llm`).
12 sujetos, 567 ventanas × 5 condiciones por brazo.

**Recuperación de respuestas inválidas:** el brazo textual produjo 66 respuestas
inválidas (2,3 %) frente a 7 en el numérico, todas por vocabulario de claims
(`channel: "not available"`, `magnitude: "none"`, `direction: "slight increase"`).
La recuperación tolerante (`sources/phase3/recovery.py`) conserva decisión,
confianza y claims bien formados; afecta al 0,25 % y 2,3 % de las respuestas.

### 6.1. Corrección (diferencia texto − numérico, IC bootstrap por sujeto)

| Condición | Δ accuracy | IC 95 % |
|---|---:|---|
| `C0_clean` | **−0,091** | [−0,131; −0,056] |
| `C1_partial_resp` | **−0,080** | [−0,112; −0,050] |
| `C2_partial_eda` | −0,023 | [−0,063; +0,016] |
| `C4_total_eda` | **−0,054** | [−0,101; −0,021] |
| `C5_total_both` | −0,026 | [−0,056; +0,000] |

- **Efecto principal:** el texto acierta menos que el numérico en todas las
  condiciones (−0,02 a −0,09), de forma significativa en limpio, C1 y C4.
- **Interacción (H3):** la caída C0 → C5 es **−0,337 (numérico)** y
  **−0,272 (texto)**; interacción **+0,065 [+0,025; +0,103]**. El texto parte
  más bajo pero **se degrada menos** en accuracy. H3 se apoya en la dirección,
  con un tamaño de efecto modesto.

### 6.2. Respaldo de claims (texto − numérico)

| Condición | Δ respaldo | IC 95 % |
|---|---:|---|
| `C0_clean` | −0,130 | [−0,157; −0,105] |
| `C1_partial_resp` | −0,071 | [−0,101; −0,040] |
| `C2_partial_eda` | −0,064 | [−0,085; −0,045] |
| `C4_total_eda` | −0,143 | [−0,172; −0,114] |
| `C5_total_both` | −0,115 | [−0,144; −0,086] |

El texto produce claims **sistemáticamente menos respaldados** (−0,06 a −0,14).
Parte del efecto es de vocabulario (etiquetas frente a nombres canónicos), pero
la diferencia persiste tras normalizar alias.

### 6.3. Certeza lingüística (texto − numérico)

| Condición | Δ índice | IC 95 % |
|---|---:|---|
| `C0_clean` | −0,17 | [−0,47; +0,11] |
| `C2_partial_eda` | −0,10 | [−0,41; +0,19] |
| `C4_total_eda` | +0,33 | [−0,10; +0,74] |
| `C5_total_both` | **+1,36** | [+0,94; +1,88] |

En la degradación severa las representaciones **divergen**: el numérico añade
cautela (índice −1,12) y el textual se vuelve más asertivo (+1,36 en C5). La
incertidumbre lingüística no solo depende de la evidencia, sino de cómo se
presenta.

### 6.4. Lectura

- H3 **apoyada**: el efecto de la degradación sobre la corrección difiere entre
  representaciones (interacción +0,065, IC sin cero), y el efecto sobre la
  certeza lingüística difiere aún más (+1,36 en C5).
- El texto no es "mejor" ni "peor" de forma uniforme: acierta menos en todo el
  rango, pero su caída relativa es menor y su lenguaje se vuelve más asertivo
  justo cuando la evidencia es peor — un patrón de riesgo epistémico.
- El respaldo de claims es menor en texto en todas las condiciones, lo que
  conviene separar de la corrección (sección 3).

## 7. Síntesis

| Hipótesis | Resultado |
|---|---|
| H1 corrección bajo degradación | **Apoyada**: caída 0,611 → 0,263; exceso de coste 0,15–0,20 sobre la referencia |
| H2a confianza y corrección | **Matizada**: la confianza no discrimina aciertos en limpio (0,51) y sube con la degradación (0,76), mientras la sobreconfianza se dispara (+0,47) |
| H2b certeza lingüística | **Apoyada en dirección**: índice +0,13 → −1,12; hedges al alza, boosters estables |
| H3 representación × degradación | **Apoyada**: interacción en accuracy +0,065 [0,025; 0,103] y divergencia fuerte en certeza (+1,36 en C5) |
| H4 adecuación de la abstención | **Refutada en su forma fuerte**: infra-abstención en degradado (brecha hasta +0,58) y sobre-abstención en limpio |
| H5 respaldo de afirmaciones | **Hallazgo principal**: el respaldo no cae (0,78 → 0,83) aunque la corrección se hunda; claims localmente ciertos, inferencia global errónea |
| H6 indicador externo | **Apoyada con SQI** (+0,05/+0,09); **no** con la compuerta de consistencia (Δ≈0) |

## 7. Reproducción

```bash
python scripts/run_degradation_sweep.py        # 2.835 llamadas (ya ejecutado)
python scripts/analyze_degradation.py          # reporte y tablas
```
