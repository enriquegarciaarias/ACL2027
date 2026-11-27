# Piloto DREAMT: sensibilidad a la evidencia en una segunda tarea (estadios de sueño, wearable)

**Estado:** preregistro congelado el **2026-11-22**, *antes* de generar ninguna
llamada al LLM. Cualquier cambio posterior a la primera llamada es una
desviación y se registra en §15.

**Naturaleza:** piloto **exploratorio de viabilidad**, no experimento. n = 4
sujetos y 80 ventanas: sirve para comprobar que la maquinaria funciona y para
leer la **dirección** de los efectos. No produce resultados citables ni
intervalos de confianza por sujeto.

---

## 1. Por qué este dataset y por qué ahora

### 1.1. Las hipótesis no dependen de la etiqueta

H1–H6 ([`hipotesis.md`](hipotesis.md) §3) se formulan sobre **corrección,
confianza numérica, certeza lingüística, respaldo de claims y abstención bajo
degradación controlada de la entrada**. Ninguna menciona estrés: estrés vs
baseline es una propiedad de las etiquetas de WESAD, no del constructo.

El estrés aparece solo en:

- §7 de [`hipotesis.md`](hipotesis.md): el **confusor del movimiento**, que es
  un sesgo propio del protocolo de WESAD;
- §8.1: la elección de tarea del piloto (WESAD).

Por tanto, cambiar de fenómeno **no toca §3**. Lo único que hay que
re-instanciar por dataset es la sección de confusores (§7), que aquí se
reemplaza por §14.2.

### 1.2. Lo que DREAMT aporta

| Aporte | Detalle |
|---|---|
| Segunda tarea, otro fenómeno | Estadios de sueño en vez de estados afectivos |
| Etiqueta de experto | Puntuación de PSG (EEG/EOG/EMG), no autoinforme |
| Sensor comercial | Empatica E4, el mismo tipo de dispositivo que se desplegaría |
| Volumen | 100 sujetos (una noche cada uno), frente a 12 sujetos de evaluación en WESAD |
| Confusor ausente | Ver §1.3 |

### 1.3. Un confusor de WESAD desaparece (argumento a favor)

En WESAD la calidad de la señal **correlaciona con la etiqueta** por un motivo
espurio: hablar en público y calcular mueven el cuerpo, así que ACC y EMG
discriminan estrés con AUC 0,928 — el modelo podía acertar «por movimiento» y
no por fisiología ([`hipotesis.md`](hipotesis.md) §7.5). En DREAMT la relación
es la contraria:

| | WESAD | DREAMT |
|---|---|---|
| ACC ↔ etiqueta | **espuria** | **definicional** (vigilia = movimiento) |
| Degradar ACC | destruye un atajo ilegítimo | destruye evidencia legítima |

Es decir: aquí la degradación **sí** mide pérdida de evidencia, y el hallazgo de
WESAD sobre el atajo de movimiento se puede contrastar en un dataset donde el
movimiento es evidencia válida. Es un control de generalidad mejor que el de
PhysioNet 2017.

---

## 2. Datos verificados (medidos, no supuestos)

Comprobaciones ejecutadas sobre `results/input/dreamt/` el 2026-11-22.

| Elemento | Valor verificado |
|---|---|
| Sujetos | 100 (`S002`–`S101`), **una noche** por sujeto |
| `data_64Hz/` | 14 GB, solo señales E4 + etiquetas de sueño y apnea |
| `data_100Hz/` | 101 GB, PSG (EEG, EOG, EMG mentón, ECG, flujo, tórax, abdomen, SpO2) + E4 |
| Duración E4 | 8,6 h (incluye «P» de preparación) |
| Duración PSG | 6,7 h (solo la noche puntuada) |
| Etiquetas | `Sleep_Stage` (P/W/N1/N2/N3/R) + `Obstructive_Apnea`, `Central_Apnea`, `Hypopnea`, `Multiple_Events` |
| Metadatos | `participant_info.csv`: edad, sexo, IMC, AHI, OAHI, SaO2 media, índice de arousal, historia clínica |
| **Estrés / afecto** | **no existe**: el ZIP solo contiene los 200 CSV + 3 documentos; no hay cuestionarios |
| Licencia | PhysioNet **Restricted** Health Data 1.5.0 (no redistribuible) |

### 2.1. Distribución de clases (13 sujetos muestreados)

| Clase | % de filas | Sujetos con la clase (de 13) |
|---|---:|---:|
| N2 | 40,7 | 13 |
| **P** (preparación) | 21,7 | 13 |
| W | 13,1 | 13 |
| R | 11,0 | 13 |
| N1 | 9,2 | 13 |
| N3 | 4,4 | **6** |

«P» es el 22 % del registro y hay que excluirlo, igual que WESAD excluye los
huecos de etiqueta 0.

### 2.2. Cuatro hallazgos de manejo de datos (costaron una comprobación cada uno)

1. **La rejilla de 30 s está anclada al inicio de la noche puntuada**, no a la
   primera fila del CSV. En `S004` el desfase es de 704 muestras (11 s).
   Anclando en la primera fila salen **27 % de épocas «mixtas»** que no existen;
   anclando en la primera muestra con etiqueta distinta de «P» salen **0,0000**
   en los 13 sujetos comprobados.
2. **Las etapas son constantes por época**: las rachas de etiqueta son múltiplos
   de 1920 muestras (30 s a 64 Hz) en el 99,3 % de los casos.
3. **`IBI` está en segundos** (0,469–1,375), aunque `Metadata.txt` declare
   milisegundos.
4. **`ACC` viene en cuentas de 1/64 g** (rango ±128): hay que dividir por 64,
   la misma convención que el ACC de muñeca de WESAD.

### 2.3. Extractores del proyecto probados sobre DREAMT

Ejecutados sobre ventanas de 60 s reales (no simuladas):

| Modalidad | Origen | Resultado |
|---|---|---|
| ECG | PSG `ECG` 100 Hz | 93 latidos/60 s → HR 93,7 bpm, SDNN 24,0 ms, RMSSD 9,0 ms (solo con PSG; **no en la rama wearable**, D7) |
| EDA | E4 4 Hz | media 9,21 µS, std 2,06 |
| TEMP | E4 4 Hz | 32,04 °C, std 0,196 |
| ACC | E4 32 Hz (÷64) | std 0,232 g |
| BVP | E4 64 Hz | HR 68,6 bpm (picos válidos; **RR no**, D7) |
| RESP | PSG `THORAX` 100 Hz | 25,4 rpm, 21 ciclos/60 s (solo PSG) |
| HR | E4 1 Hz | 0 NaN en las ventanas muestreadas; mediana por clase: W 68,1 · N1 64,9 · N2 65,1 · R 66,6 bpm |

Los seis extractores corren sin cambios. La capa de features es reutilizable
tal cual; lo que la rama wearable **no** puede dar es HRV (D7).

---

## 3. Decisiones del piloto (fijadas por el autor)

| # | Decisión | Valor |
|---|---|---|
| 1 | Rama de datos | **solo wearable** (`data_64Hz/`, 14 GB). El PSG se usa únicamente como fuente de etiquetas, y esas etiquetas vienen ya en los ficheros de 64 Hz |
| 2 | Ventana | **60 s, pureza 1,0** (comparable con WESAD y suficiente para HRV) |
| 3 | Clases | **W / N1 / N2 / R** + `abstain`. **N3 excluido** |
| 4 | Degradación | **5 condiciones + `NC_absent`** (§8) |
| 5 | Representaciones | **numérica y textual** (brazo H3) |

Ninguna de estas cinco decisiones se cambia tras la primera llamada sin
registrarla en §15.

---

## 4. Sujetos y split

| Rol | Sujetos | Criterio |
|---|---|---|
| **Desarrollo** (fijo) | `S002`, `S010`, `S026` | Reservados para few-shot y comprobación de la tabla de firma. **Nunca** entran en evaluación, en este piloto ni en la línea DREAMT |
| **Piloto** | `S034`, `S051`, `S076`, `S084` | AHI 5 / 1 / 2 / 1 (sin confusor de apnea), las 4 clases presentes, dos con N3 por si se activa el bloque opcional de §5.3 |

| SID | AHI | Arousal | Épocas 30 s | W | N1 | N2 | R | N3 | Ventanas puras de 60 s (W/N1/N2/R/N3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S034 | 5 | 6 | 716 | 96 | 52 | 382 | 34 | 152 | 90/36/359/32/141 |
| S051 | 1 | 25 | 768 | 117 | 72 | 469 | 110 | 0 | 91/27/419/104/0 |
| S076 | 2 | 10 | 841 | 75 | 28 | 439 | 158 | 141 | 58/10/412/151/135 |
| S084 | 1 | 11 | 820 | 177 | 104 | 346 | 193 | 0 | 162/73/324/186/0 |

La reserva de `S002`, `S010` y `S026` se declara ahora y es permanente: es el
mismo mecanismo que el split congelado de WESAD
([`split_dev_test.md`](split_dev_test.md)), y se hace antes de medir para no
repetir la desviación que ya costó una auditoría.

---

## 5. Ventanas y muestreo

### 5.1. Construcción

1. Leer `Sleep_Stage` del fichero de 64 Hz del sujeto.
2. Localizar la primera muestra con etiqueta distinta de `P` → **ancla**.
3. Cortar en épocas de 30 s de 1920 muestras desde el ancla (la última época
   incompleta se descarta).
4. Una ventana de 60 s = dos épocas consecutivas con **la misma** etiqueta
   (pureza 1,0). Sin solape.
5. Descartar `P`, `Missing` y **`N3`**.

### 5.2. Muestreo

- **5 ventanas por clase y sujeto** → 20 por sujeto → **80 ventanas**.
- Selección **sistemática sobre los tercios de la noche** (se eligen ventanas
  repartidas por el registro, no las primeras): evita que una clase quede
  concentrada en un tramo y confunda la firma con el orden de la noche.
- Semilla fija (`42`) y registro de los `window_id` elegidos en el manifiesto.

### 5.3. Bloque opcional de techo (desactivado por defecto)

`S034` y `S076` tienen N3 (141 y 135 ventanas puras). Existe la posibilidad de
un bloque diagnóstico de 20 ventanas N3 × 5 condiciones = 100 llamadas con
prompt de 5 clases, para responder a *¿detecta el modelo el estado más
distinguible del conjunto wearable, incluso en limpio?* **No forma parte del
piloto** y no entra en ninguna métrica; se activa con bandera si el resultado
limpio de 4 clases resulta ambiguo.

---

## 6. Espacio de decisión

| Clase | Etiqueta | Nota |
|---|---|---|
| `wake` | W | Vigilia intra-registro (no la «P» de preparación) |
| `n1` | N1 | Transición |
| `n2` | N2 | NREM ligero |
| `rem` | R | Sueño REM |
| `abstain` | — | Abstención de primera clase, como en WESAD |

**N3 se excluye del espacio de clases.** Motivos, con cifras:

1. Es el 5,3 % de las épocas y **falta en la mitad de los sujetos** (6 de 13
   comprobados): en 80 ventanas tendría ~10 casos cuando existe y 0 en los
   demás, así que cualquier tasa por clase sería ruido.
2. **Fusionarlo con N2 sería peor**: N3 tiene HR más baja, HRV más alta y
   movimiento casi nulo. Etiquetar eso como N2 inyecta ruido de etiqueta
   precisamente en el eje de **H5**, donde el modelo citaría evidencia verdadera
   que la tabla de firma marcaría como contradictoria, y se leería como fallo
   del modelo.

---

## 7. Evidencia disponible y mapa de canales

La rama wearable **no tiene respiración** (el E4 no la mide) y **no tiene onda
de ECG** (solo `HR` a 1 Hz e `IBI` por latido). Eso define el conjunto de
evidencia:

| Canal DREAMT | Ranura canónica | Extractor | Features resultantes |
|---|---|---|---|
| `EDA` (4 Hz) | `chest_EDA` | `eda.extract` | 9 de EDA |
| `TEMP` (4 Hz) | `chest_Temp` | `temp.extract` | 4 de temperatura |
| `ACC_X/Y/Z` (32 Hz, ÷64) | `chest_ACC` | `acc.extract` | 5 de movimiento |
| `HR` (1 Hz, del dispositivo) | bloque `ecg` | **derivación desde HR** | `ecg_hr_bpm`, `ecg_rr_mean_ms` |
| — | `resp` | — | **ausente** |
| — | `emg` | — | **ausente** |

Dos decisiones de implementación que se registran aquí:

1. **El E4 se mapea a las ranuras de pecho** (`chest_EDA`, `chest_Temp`,
   `chest_ACC`) para que lo recoja la vista LLM existente, que filtra por
   modalidad y excluye las de muñeca (`_llm_view_features`). Físicamente es
   muñeca; se documenta en cada respuesta.
2. **El bloque `ecg` se deriva de la columna `HR` del dispositivo** (1 Hz):
   `ecg_hr_bpm` = mediana de HR dentro de la ventana y `ecg_rr_mean_ms` =
   60.000 / HR. Se probaron las otras dos fuentes y **ambas resultaron
   inutilizables para HRV** (desviación D7): `IBI` está cuantizado a 1/64 s con
   *dither* (tras deduplicar, la mayoría de ventanas dan 0–21 intervalos) y los
   picos de `BVP` producen RR de 328 a 5.641 ms (SDNN 100–300 ms). Por eso la
   tarea **no incluye HRV** (D8). No se sintetiza ninguna onda.

Consecuencia declarada: **el modelo debe resolver la tarea sin canal
respiratorio** (informativo en la literatura de sueño) **y sin HRV**. Es una
limitación del piloto (§14.1) y, a la vez, el resultado que el piloto puede dar
antes de invertir en la rama de 100 Hz.

---

## 8. Matriz de degradación

La matriz C0–C5 de WESAD no se puede reutilizar: degrada EDA y **respiración**,
y en la rama wearable la respiración no existe. Además, un hallazgo operativo de
WESAD obliga a cambiar el eje: **con ≥50 % de ausencia el extractor devuelve
`null` para toda la modalidad**, así que «parcial» y «total» colapsan
(`C3_total_resp` ≡ `C1_partial_resp`).

Por eso el eje del piloto es el que de verdad interesa a la tesis:
**ausencia honesta** (`not available`) frente a **degeneración adversarial**
(valor presente pero fisiológicamente imposible).

| Condición | Operación | Qué mide |
|---|---|---|
| `C0_clean` | — | Referencia |
| `C1_absent_eda` | EDA 100 % NaN | ¿Acepta que falta el canal sudomotor? |
| `C2_degenerate_eda` | EDA a ceros | ¿Distingue «no hay dato» de «dato imposible»? (EDA medida: mínimo 0,103 µS) |
| `C3_absent_hrv` | `IBI` y `HR` 100 % NaN | Pierde todo lo autonómico; conserva EDA, temperatura y ACC |
| `C4_absent_core` | EDA + `IBI`/`HR` + ACC 100 % NaN | Ausencia del núcleo de la evidencia |
| `NC_absent` | Todas las features a `not available` | ¿Se abstiene cuando no hay evidencia alguna? |

Las dos condiciones de EDA son el contraste central: **ausencia** frente a
**valor imposible**. En WESAD ese contraste se midió con `NC_nan`/`NC_zero`,
pero `NC_zero` no es válido en escala centrada (0 = media del sujeto); aquí se
aplica antes de estandarizar, sobre el array crudo, así que **sí** es válido.

`NC_zero` de la implementación genérica **no se usa**: su semántica queda
cubierta por `C2_degenerate_eda`, que solo toca EDA y mantiene el resto como
evidencia.

---

## 9. Tabla de firma por clase (congelada)

La coherencia de H5 necesita saber qué dirección de cada feature **favorece**
cada clase. Para WESAD se congeló `STRESS_SIGNATURE` (12 features); aquí se
congela el equivalente para sueño: **3 features, 3 mecanismos**, medidas
respecto de la **referencia intra-sujeto de la noche** (z robusto sobre todas las
épocas puras del sujeto, no solo las muestreadas).

| Feature | Mecanismo | `wake` | `n2` | `rem` |
|---|---|:---:|:---:|:---:|
| `acc_std_g` | Movimiento / atonía | **+** | **−** | **−** |
| `ecg_hr_bpm` | Cronotropismo | **+** | **−** | **+** |
| `eda_std_us` | Actividad sudomotora | **+** | **−** | **+** |

Fundamento: la atonía muscular en REM y el movimiento en vigilia son los hechos
mejor establecidos de la polisomnografía; la frecuencia cardiaca baja al dormir
y sube en vigilia; la actividad fásica de la piel es mayor en vigilia y en REM
que en NREM estable.

**`n1` se declara SIN firma.** Sus direcciones coinciden con `n2`, así que no
sería discriminativa: el modelo no puede apoyar «N1» con una evidencia que
apoya igual «N2». Se conserva como clase de decisión y se excluye del cálculo de
coherencia (sus celdas salen `insufficient`, mecanismo que ya existe).

**`rem` no firma `ecg_hr_bpm`.** Se midió en las noches de desarrollo que la HR
mediana de REM está a −0,2 SD de la referencia, prácticamente el centro; la
tabla deja el signo en 0 y REM se apoya en movimiento y EDA (desviación D8).

### 9.1. Features excluidas y por qué

| Feature | Motivo |
|---|---|
| `temp_mean_degc`, `temp_std_degc`, `temp_range_degc`, `temp_slope_degc_per_s` | **Confusor de orden de la noche**: la temperatura cutánea sigue el ciclo y el orden de los estadios; es el equivalente del confusor térmico de WESAD (r = 0,801 con la posición del protocolo) |
| `ecg_sdnn_ms`, `ecg_rmssd_ms`, `ecg_n_beats` | **No calculables** en la rama wearable (D7): `IBI` y `BVP` resultaron inutilizables para HRV |
| `eda_slope_us_per_s` | Deriva/adaptación dentro de la ventana; el signo no es identificable |
| `ecg_rr_mean_ms` | Inversa aritmética de `ecg_hr_bpm`; duplicaría el peso de HR |
| `acc_mean_g` | Incluye la gravedad; no es direccional |
| `acc_p95_g`, `acc_energy_g2` | Redundantes con `acc_std_g` |
| `eda_tonic_mean_us`, `eda_phasic_std_us`, `eda_scr_amplitude_*` | Redundantes con `eda_std_us` o dependientes del umbral SCR |
| `resp_*`, `emg_*` | No disponibles en la rama wearable |

### 9.2. Conjunto de sensibilidad

`SENSITIVITY` = tabla primaria + `eda_scr_rate_per_min` + `acc_motion_fraction`.
Motivo: `eda_scr_rate_per_min` depende del umbral SCR, calibrado para EDA de
pecho en WESAD (0,05 µS) y **saturado en EDA de muñeca** (57 detecciones en
60 s en la ventana de preparación). Si el umbral se recalibra, entra en el
primario; mientras tanto, no contamina la métrica principal.

### 9.3. Comprobación previa obligatoria (P0)

Antes de interpretar ninguna coherencia se calcula, sobre `C0_clean` y con las
etiquetas reales, la mediana de `z` de las 3 features primarias por clase:

- **P0 se cumple** si las 3 direcciones coinciden con la tabla en `wake`, `n2` y
  `rem`.
- Si falla alguna, **no se edita la tabla**: se publica como pre-chequeo fallido
  y la feature afectada pasa al conjunto de sensibilidad. Es el mismo mecanismo
  que se usó en WESAD (P0 11/12; falló `resp_rate_bpm` por la presencia de
  meditación en el grupo de comparación).

**Resultado medido (2026-11-22, S002/S010/S026, 976 ventanas):** P0 **8/8**
(tres features × tres clases, menos la celda `rem`/HR que no se firma). Las
medianas son pequeñas salvo el movimiento en vigilia:

| Clase | `acc_std_g` | `ecg_hr_bpm` | `eda_std_us` |
|---|---:|---:|---:|
| `wake` | **+1,12** | +0,34 | +0,02 |
| `n2` | −0,08 | −0,10 | −0,02 |
| `rem` | −0,05 | (no firmada) | +0,07 |

La comprobación **pasa por signo**, pero el tamaño de efecto fuera de `wake` es
de centésimas de SD. Es el hallazgo de viabilidad que hay que reportar: con estas
18 features y ventanas de 60 s, **la separación N2/REM es débil por
construcción** y la referencia supervisada lo confirma (§12 del informe de
evaluación). El piloto no mide «¿puede el LLM hacer polisomnografía?» sino
«¿cómo responde su evidencia y su lenguaje cuando se retira evidencia?».

---

## 10. Prompt y esquema de salida

- `decision_mode = "sleep4"`; valores canónicos
  `wake | n1 | n2 | rem | abstain`; alias aceptados (`w`, `nrem1`, `r`,
  `rem_sleep`); etiquetas de otros datasets (`stress`, `baseline`) son
  inválidas.
- Nota de tarea y reglas **en inglés**, con la misma estructura que el prompt
  binario: decisión primero, luego `confidence` (entero 0–100), `claims`
  (métrica, canal, dirección, magnitud) y `reasoning`.
- Sin fugas: sin etiqueta, sin `SID`, sin hora absoluta, sin orden de la noche,
  sin SQI ni cobertura. IDs opacos (`win_XXXXXXXX`).
- Representación primaria: **z intra-sujeto** (misma decisión que E12, y por
  tanto misma limitación declarada en
  [`split_dev_test.md`](split_dev_test.md) desviación 7: la referencia se fija
  con las épocas del propio sujeto).
- Few-shot: medianas por clase de los **sujetos de desarrollo** (`S002`,
  `S010`, `S026`), con procedencia registrada (`subjects`, `n_windows`,
  `class_counts`, `statistic`). Nunca del sujeto evaluado.
- Regla de no truncamiento: si el prompt excede `num_ctx`, la ventana se excluye
  y se registra.

---

## 11. Métricas y reglas de lectura

Por condición y representación:

| Bloque | Métrica |
|---|---|
| Corrección (H1) | accuracy sobre decididas, `accuracy incl. abstención`, balanced (macro), precisión/recall por clase, matriz de confusión 4×4 |
| Cobertura (H4) | abstención por condición, y **por clase real** (la abstención dependiente de etiqueta fue el hallazgo del 14b en WESAD) |
| Confianza (H2a) | AUROC de confianza para acierto; sobreconfianza (confianza media − accuracy) |
| Certeza (H2b) | índice de `certainty.py` sobre `reasoning`, separando decididas y abstenidas |
| Claims (H5) | tasa de respaldo (`supported`), `contradicted`, `unsupported`, y **coherencia con la decisión** por clase predicha |
| Calidad (H6) | AUROC incremental del SQI sobre la confianza, con CV agrupada por sujeto |
| Representación (H3) | Δ texto − numérico por condición, con cobertura de los dos brazos |

**Reglas de lectura, obligatorias:**

1. **Nunca se publica accuracy sin balanced y cobertura.** Con prevalencia
   desigual, la accuracy premia al clasificador degenerado; ha ocurrido tres
   veces en esta línea.
2. Se declara la **tasa base de la muestra** (25 % por clase por construcción,
   pero se reportará la del corpus de origen).
3. Con n = 4 sujetos **no se calculan intervalos de confianza por sujeto**: el
   piloto describe dirección y magnitud aproximada, no inferencia.

---

## 12. Coste

| Bloque | Llamadas |
|---|---:|
| 80 ventanas × 5 condiciones (numérico) | 400 |
| `NC_absent` (numérico) | 80 |
| Ídem textual | 480 |
| **Total** | **960** |

En la práctica son 4 sujetos × 5 ventanas/clase × 4 clases = **80 ventanas**
(siempre que cada sujeto tenga las cuatro clases; los recuentos medidos son
W 31–82, N1 7–43, N2 168–218, R 17–94, así que se cumplen). A ~4 s por llamada
con `qwen2.5:7b` en la RTX 4060 Ti local: **60–70 min**. La Fase 0 (features,
degradación y prompts, sin LLM) tarda ~40 s en total (5 s por noche).

Disco: se leen 4 CSV (~150 MB cada uno, solo las columnas necesarias); **no se
canonicaliza nada** y no se toca `INGEST`.

---

## 13. Falsadores del piloto (qué haría parar)

| # | Observación | Lectura |
|---|---|---|
| F1 | P0 falla en ≥2 de las 3 features | La firma no es medible con esta evidencia: el eje H5 no se puede evaluar aquí. Parar y rediseñar antes de gastar LLM |
| F2 | `C0_clean` con balanced ≈ 0,25 y confusión repartida | La evidencia wearable no basta para la tarea: no hay nada que degradar. Reportar como resultado negativo |
| F3 | Abstención igual en `C3`/`C4` que en `C0` | El modelo no percibe la ausencia: H4 no es evaluable en esta tarea |
| F4 | Numérico y textual idénticos en todas las condiciones | H3 no se puede estudiar aquí (el corpus de WESAD sí mostró interacción) |
| F5 | `C1_absent_eda` y `C2_degenerate_eda` responden igual | El modelo no distingue ausencia de valor imposible: hallazgo por sí mismo, y el contraste central del piloto queda anulado |

---

## 14. Limitaciones y confusores declarados

### 14.1. Limitaciones

1. **Sin respiración** (el E4 no la mide): el canal más informativo del sueño
   después del EEG queda fuera. Los resultados no se comparan con la literatura
   de staging con respiración.
2. **Sin onda de ECG**: HRV derivada de `IBI` de Empatica, no de picos R
   detectados por nosotros.
3. **Temperatura cuantizada**: 171 niveles distintos en 8,6 h; las features de
   temperatura son de baja resolución.
4. **EDA de muñeca a 4 Hz**, no de pecho; el umbral SCR no está calibrado para
   este sensor.
5. **~40–50 % de ventanas perdidas** al exigir pureza 1,0 en 60 s (dos épocas
   iguales consecutivas).
6. **N3 fuera** (§6).
7. **n = 4 sujetos, 80 ventanas**: piloto, no experimento.
8. **Licencia restringida**: solo se publican features derivadas y estadísticos;
   nunca señales crudas ni respuestas que permitan reconstruirlas.

### 14.2. Confusor de orden de la noche (sustituye a §7 de `hipotesis.md`)

Los estadios siguen un orden biológico: al principio N2/N3, al final REM y
vigilia. Por tanto **cualquier feature que derive con la noche** —y la
temperatura es la candidata obvia— correlaciona con la clase sin ser su causa.
Por eso:

- la temperatura queda **fuera de la tabla de firma** (§9.1);
- el muestreo se reparte por los tercios de la noche (§5.2);
- la referencia de z es **intra-sujeto**, que absorbe nivel pero no deriva
  temporal;
- si el piloto muestra que el modelo acierta sin ninguna feature fisiológica
  (F2 invertido: acierta en `C4_absent_core`), la lectura será «usa el orden de
  la noche», y habrá que controlarlo antes de cualquier conclusión.

---

## 15. Desviaciones declaradas

| # | Desviación | Estado |
|---|---|---|
| D1 | Se usa el **wearable** y no el PSG como evidencia | Decisión del autor; cubierta por §1.1 (las hipótesis no dependen de la modalidad) |
| D2 | **N3 fuera** del espacio de clases | Decisión previa a medir (§6); no post hoc |
| D3 | La coherencia de H5 se calcula solo para `wake`, `n2`, `rem` (`n1` sin firma) | Previsto en §9; declarado antes de medir |
| D4 | La tabla de firma es **nueva** y no se ha validado con datos de sueño | Mitigado con P0 (§9.3) |
| D5 | `S002`, `S010` y `S026` se reservan como desarrollo para toda la línea DREAMT | Declarado el 2026-11-22, antes de medir |
| D6 | El piloto **no** es comparable en accuracy con WESAD: distinta tarea, distinta etiqueta, distinto sensor | Se comparan **hipótesis**, no cifras |
| D7 | **Sin HRV**: `IBI` (cuantizado a 1/64 s) y `BVP` (RR 328–5.641 ms) no permiten SDNN/RMSSD; el bloque cardiaco se deriva de la columna `HR` del dispositivo | Medido el 2026-11-22 sobre 7 sujetos, antes de generar respuestas; documentado en §7 |
| D8 | La tabla de firma queda en **3 features** y `rem` no firma HR (HR de REM ≈ mediana de la noche) | Medido en las noches de desarrollo; P0 8/8 (§9.3) |

Corolario de D7: la tarea de sueño **no** usa `ecg_rmssd_ms`, la feature con más
peso teórico en la literatura. Si el piloto muestra un techo bajo, la rama de
100 Hz (PSG, HRV real) es la siguiente iteración, no un cambio de diseño.

---

## 16. Reproducción

```bash
source .venv/bin/activate

# Fase 0 (local, sin LLM): features, degradación y prompts; escribe el manifiesto
python scripts/run_p4_dreamt.py --phase prepare --windows-per-class 5

# Piloto (local, Ollama con qwen2.5:7b): 960 llamadas, ~1 h
# (--representation both = vista numérica + textual sobre la misma evidencia)
python scripts/run_p4_dreamt.py --phase generate --representation both

# Evaluación (métricas, claims, coherencia y referencia features-only)
python scripts/run_p4_dreamt.py --phase evaluate --reference
```

El mismo flujo por el CLI de fases del proyecto (recomendado: valida rutas,
entorno y coste antes de llamar al modelo):

```bash
python main.py --proc GENERATE --experiment dreamt --dry-run    # plan + 960 llamadas
python main.py --proc GENERATE --experiment dreamt
python main.py --proc EVALUATE --experiment dreamt --reference
```

Salidas en `results/output/DREAMT/PHASE2/pilot/`:

```
manifest.json                     # ventanas elegidas, clases, tercios, hashes
p0.json                           # pre-chequeo de la tabla de firma (§9.3)
few_shot.json                     # ejemplares y su procedencia (S002/S010/S026)
reference/{sid}.json              # medianas e IQR de la noche (referencia intra-sujeto)
features/{sid}.night.jsonl         # features de todas las épocas puras de la noche
prompts/{sid}.prompts.jsonl       # prompt exacto enviado (numérico y texto) + etiqueta
labels/{sid}.labels.jsonl         # ground truth separado del prompt
responses/{slug}/{sid}.{repr}.jsonl  # decisión, confianza, claims, procedencia
evaluation_{slug}.json            # métricas por condición y representación
```

---

## 17. Qué se reportará en el manuscrito

1. Que el diseño de sensibilidad a la evidencia **no depende de la etiqueta**:
   se replica en una segunda tarea con etiqueta de experto y otro sensor.
2. H1 (coste de la degradación) y H5 (respaldo y coherencia) en sueño frente a
   afecto, con la advertencia de que son **dos tareas, no dos versiones de la
   misma**.
3. El contraste **ausencia vs degeneración** (`C1` vs `C2`), que en WESAD quedó
   contaminado por la invalidez de `NC_zero` en escala centrada.
4. Que en DREAMT el movimiento **es** evidencia legítima, lo que convierte el
   hallazgo del confusor de WESAD (§7.5) en un contraste entre dos datasets, no
   en una limitación.

---

## 18. Resultados del piloto (ejecutado 2026-11-22)

Ejecutado en la máquina local (RTX 4060 Ti 8 GB, Ollama 0.30.7,
`qwen2.5:7b` Q4_K_M, digest `845dbda0…`): **960/960 llamadas**, 34 min,
**3 inválidas** (0,3 %; todas del brazo textual). P0 **8/8** antes de generar.

Artefactos en `results/output/DREAMT/PHASE2/pilot/` (`manifest.json`, `p0.json`,
`evaluation_qwen2_5_7b.json`, `run_provenance.json`).

### 18.1. Techo de la tarea: la referencia supervisada apenas supera el azar

Referencia *features-only* con **las mismas 18 features y la misma
estandarización intra-sujeto** que ve el LLM, validación cruzada dejando un
sujeto fuera (4 pliegues, 1.309 ventanas):

| Pliegue | n | accuracy | balanced |
|---|---:|---:|---:|
| S034 | 267 | 0,603 | 0,446 |
| S051 | 339 | 0,531 | 0,440 |
| S076 | 326 | 0,543 | 0,484 |
| S084 | 377 | 0,342 | 0,323 |
| **Agregado** | **1.309** | **0,494** | **0,418** |

Azar = 0,25 (accuracy y balanced). **La tarea es débilmente resoluble con esta
evidencia**: 20 puntos de accuracy sobre el azar, pero un balanced de 0,42. Es
el dato que ordena la lectura de todo lo demás: sin el bloque de 100 Hz
(HRV real, EEG/EOG/EMG) y sin respiración, las features de muñeca de 60 s no
alcanzan la polisomnografía de la literatura.

### 18.2. Corrección y cobertura (H1, H4)

| Vista | Condición | n | inv | abstención | cobertura | accuracy | balanced |
|---|---|---:|---:|---:|---:|---:|---:|
| numérico | `C0_clean` | 80 | 0 | 0,0 % | 100 % | 0,300 | 0,300 |
| numérico | `C1_absent_eda` | 80 | 0 | 25,0 % | 75,0 % | 0,417 | 0,402 |
| numérico | `C2_degenerate_eda` | 80 | 0 | 0,0 % | 100 % | 0,338 | 0,338 |
| numérico | `C3_absent_hrv` | 80 | 0 | 0,0 % | 100 % | 0,300 | 0,300 |
| numérico | `C4_absent_core` | 80 | 0 | **100 %** | 0 % | — | — |
| numérico | `NC_absent` | 80 | 0 | **100 %** | 0 % | — | — |
| texto | `C0_clean` | 80 | 0 | 0,0 % | 100 % | 0,275 | 0,275 |
| texto | `C1_absent_eda` | 80 | 0 | **98,8 %** | 1,2 % | 1,000 † | 1,000 † |
| texto | `C2_degenerate_eda` | 80 | 1 | 0,0 % | 98,8 % | 0,304 | 0,304 |
| texto | `C3_absent_hrv` | 80 | 2 | 12,8 % | 85,0 % | 0,235 | 0,231 |
| texto | `C4_absent_core` | 80 | 0 | **100 %** | 0 % | — | — |
| texto | `NC_absent` | 80 | 0 | **100 %** | 0 % | — | — |

† Una sola ventana decidida de 80: la accuracy no es interpretable y por eso se
publica con cobertura.

> **Convención de los artefactos (2026-11-23).** Las cifras de §18.2 son de la
> ejecución original, sin recuperación. Los `evaluation_*.json` actuales se
> generan con `--recovery`, que conserva la decisión de las respuestas con claims
> mal formados (3 en el brazo textual de `qwen2.5:7b`). La diferencia es de una a
> tres ventanas: `C2` textual pasa a cobertura 100 % / accuracy 0,312 y `C3` a
> 87,5 % / 0,241. No cambia ninguna lectura de §18.

**H1 en limpio.** El LLM está **por debajo** de la referencia: exceso de coste
−0,194 (numérico) y −0,219 (texto) en accuracy; −0,118 y −0,143 en balanced. No
es un fallo de información (la referencia resuelve 0,49); es un fallo del lector.
El recall en `C0` lo explica: numérico colapsa a la clase mayoritaria
(`n2` 0,70) y abandona `n1` (0,10) y `rem` (0,05); el texto reparte más
(`n1` 0,45, `n2` 0,30, `rem` 0,00).

**La accuracy sube al destruir EDA** (`C1` numérico 0,417 > `C0` 0,300) sin que
el balanced acompañe (0,402). No es una mejora: al retirar EDA el modelo se
apoya en movimiento y responde `wake` (32/80), un clasificador degenerado que la
accuracy premia. Es la cuarta vez que aparece este patrón en la tesis.

### 18.3. H4: la representación decide si la ausencia se percibe

Este es el hallazgo del piloto, y no estaba garantizado por el diseño:

- **Ausencia del núcleo** (`C4`, `NC_absent`): **100 % de abstención en las dos
  vistas**. El modelo percibe que no hay evidencia (falsador **F3 no activado**).
- **Ausencia de EDA** (`C1`): **25,0 % numérico frente a 98,8 % textual**. La
  misma evidencia ausente, dos vistas, dos comportamientos. La lista numérica de
  `not available` se pasa por alto; el texto «EDA is not available» no.
- **Degeneración de EDA** (`C2`, ceros): **0 % de abstención en las dos vistas**.
  El valor imposible se acepta como medida real y `eda_scr_count` nulo en señal
  plana no alerta. **Falsador F5 no activado**: el modelo distingue ausencia de
  valor imposible, pero solo cuando la ausencia se verbaliza.
- **Ausencia de HR** (`C3`): 0 % numérico, 12,8 % textual. Es el canal con menos
  peso en la firma del sueño, y así se comporta.

Consecuencia: **H4 no es una propiedad del modelo, es una propiedad del par
(modelo, representación)**. En WESAD el contraste equivalente no se pudo medir
porque `NC_zero` era inválido en escala centrada (§14, y `docs/e12_resultados.md`
§7.4); aquí, con `C1`/`C2` aplicados antes de estandarizar, sí.

### 18.4. H5: claims verdaderos, inferencia débil

Tabla recalculada el 2026-11-23 con el verificador corregido (§19.2): el
vocabulario del dataset llega ahora también a la normalización de la métrica de
cada claim, no solo al parseo del prompt.

| Vista | Condición | claims | respaldados | contradichos | `unsupported` | coherencia | contrarios verdaderos |
|---|---|---:|---:|---:|---:|---:|---:|
| numérico | `C0_clean` | 247 | 0,684 | 0,316 | 0 | 0,719 | 0,318 |
| numérico | `C1_absent_eda` | 143 | 0,769 | 0,231 | 0 | 0,800 | 0,194 |
| numérico | `C2_degenerate_eda` | 231 | 0,632 | 0,368 | 0 | 0,803 | 0,209 |
| numérico | `C3_absent_hrv` | 229 | 0,651 | 0,341 | 0 | **0,491** | **0,537** |
| texto | `C0_clean` | 236 | 0,564 | 0,436 | 0 | 0,881 | 0,113 |
| texto | `C1_absent_eda` | 2 | 0,500 | 0,500 | 0 | 1,000 | 0,000 |
| texto | `C2_degenerate_eda` | 234 | 0,637 | 0,363 | 0 | 0,794 | 0,291 |
| texto | `C3_absent_hrv` | 177 | 0,492 | 0,277 | 0 | 0,914 | 0,179 |

- **`unsupported = 0` en las doce celdas con claims**: el modelo no cita canales
  que no están en la entrada. Deja de hablar de HR cuando HR desaparece. Es el
  mismo comportamiento que en WESAD y PhysioNet. En `C3` lo que aparece son 41
  claims `indeterminate` (23,1 %, textual): la métrica sigue nombrada en el
  prompt con valor `not available`, que es una categoría distinta de «no
  existe».
- **C3 numérico es el caso nítido**: sin HR, la decisión se mantiene (53/80
  `n2`) pero los claims que quedan —movimiento y EDA— apoyan otras clases. La
  coherencia cae de 0,719 a **0,491** y **53,7 %** de los claims verdaderos van
  **contra** la conclusión.
- **El texto es algo menos respaldado** (0,564 vs 0,684) y **más coherente**
  (0,881 vs 0,719): prosa con menos claims, mejor alineados con la decisión. Es
  la misma disociación verdad/coherencia que en el eje afectivo, con una brecha
  menor de lo que sugería la medición defectuosa.
- En `C1` textual solo quedan **2 claims** (98,8 % de abstención): abstenerse
  silencia las afirmaciones. Ausencia de evidencia → ausencia de claim → el
  sistema no puede alucinar sobre lo que no dice.

### 18.5. H2a y H2b: la confianza no informa; la certeza reacciona a la abstención

- La confianza toma **dos valores** (75 y 85) en todas las condiciones
  decididas. AUROC de confianza para el acierto: 0,59 (`C0` numérico), 0,61
  (`C2`), 0,48 (`C3`), 0,51 (`C0` texto) → **no ordena aciertos**.
- **Sobreconfianza** de +0,53 en limpio (confianza media 0,83 frente a accuracy
  0,30). El modelo está seguro en el peor punto posible.
- Índice de certeza lingüística **0,00** en las condiciones decididas y
  **−16,7 / −28,6** en las de abstención: el lenguaje se vuelve cauteloso
  **cuando se abstiene**, no cuando la evidencia se degrada.
- **Repetir no arreglaría H2a aquí**: la incertidumbre no está en el número.

### 18.6. Falsadores: qué se activó

| Falsador | Resultado |
|---|---|
| F1 (P0 falla en ≥2 features) | **No**: P0 8/8 (§9.3) |
| F2 (balanced ≈ 0,25) | **Parcialmente**: 0,300/0,275 en limpio, cerca del azar; la referencia (0,418) demuestra que hay señal, así que es límite del lector, no de la evidencia |
| F3 (abstención igual en `C3`/`C4` que en `C0`) | **No**: `C4`=100 %, `C3`=0–12,8 %, `C0`=0 % |
| F4 (numérico ≡ textual) | **No**: divergen hasta 74 pp en `C1` (25,0 % vs 98,8 % de abstención) |
| F5 (`C1` ≡ `C2`) | **No**: 25,0 % vs 0 % (numérico) y 98,8 % vs 0 % (texto) |

### 18.7. Lectura para la tesis

1. **El mecanismo se replica**: abstención ante ausencia total, `unsupported = 0`,
   claims localmente verdaderos con inferencia global errada, confianza que no
   informa. Nada de esto dependía de la etiqueta de estrés.
2. **Aparece un hallazgo que WESAD no podía dar**: la detección de ausencia
   depende de la representación (25 % vs 98,8 %). Refuerza H3 con un mecanismo
   concreto, no solo con una diferencia de precisión.
3. **Techo declarado**: con la rama *wearable* la tarea está en 0,49 de
   accuracy supervisada. **Este piloto no sirve para afirmar nada sobre la
   precisión del LLM en sueño**; sirve para las hipótesis de comportamiento.
4. **H6 no se evalúa aquí**: no hay indicador externo de calidad implementado
   para DREAMT (`SQI` y compuerta se definieron sobre canales de WESAD). Queda
   como trabajo antes de escalar a la rama de 100 Hz.
5. **Sin intervalos de confianza** (§11, regla 3): 4 sujetos, dirección y
   magnitud aproximada. Los 4 pliegues de la referencia están a la vista para
   que el lector juzgue la heterogeneidad.

### 18.8. ¿Es peor el modelo aquí que en WESAD? Sí, y hay que descomponerlo

La comparación directa de accuracy entre los dos ejes **no es válida** (otra
tarea, otro espacio de etiquetas, otra prevalencia), pero la pregunta se puede
responder si se normaliza contra la referencia supervisada del mismo eje:

| Eje | Clases | Azar (balanced) | Referencia supervisada | LLM | LLM / referencia |
|---|---:|---:|---:|---:|---:|
| WESAD `schmidt` (binario) | 2 | 0,500 | **0,905** | **0,907** | **100 %** |
| WESAD 4 clases (histórico) | 4 | 0,250 | 0,828 (acc) | 0,611 acc / 0,478 bal | ≈ 74 % |
| DREAMT (este piloto) | 4 | 0,250 | 0,494 acc / **0,418 bal** | 0,300 / **0,300** | ≈ 61 % |

Las tres causas están apiladas y **solo una es del modelo**:

1. **La evidencia es más débil** (no es culpa del lector): la referencia baja de
   0,828 a 0,494 de accuracy. Muñeca a 60 s, sin HRV y sin respiración no
   alcanza la polisomnografía de la literatura. Este es el techo del dataset,
   no del LLM.
2. **El espacio de decisión es más difícil**: 4 clases balanceadas (azar 0,25)
   frente a 2 clases con prevalencia 0,30 en WESAD (azar balanced 0,50). No es
   comparable en términos absolutos; por eso se publica balanced y cobertura.
3. **El lector pierde más de lo que pierde la referencia** (esto sí es del
   modelo): la ratio LLM/referencia cae de ≈0,74 a ≈0,61, y el fallo es del
   mismo tipo que en WESAD: **colapso a la clase mayoritaria**. Aquí `rem`
   queda en recall 0,05 (numérico) y 0,00 (texto), igual que en WESAD el modelo
   nunca predijo `amusement` ni `meditation`.

**Advertencia de motor.** El piloto DREAMT se ejecutó con **Ollama 0.30.7
local**, mientras que el corpus de WESAD se generó con **0.40.6 en el
servidor**. Está documentado que el mismo prompt y la misma semilla cambian el
comportamiento entre motores (WESAD `C5`: accuracy 0,682 en servidor frente a
0,614 en local, y 0 vs 28 abstenciones; ver `docs/repetitions.md` §8). Parte de
la diferencia de la fila 3 podría ser del motor y no del modelo, así que **la
comparación cross-eje no se cita como resultado** hasta repetir DREAMT en el
servidor (960 llamadas, ~1 h) o WESAD en local.

Lo que sí se sostiene sin depender de esa comparación: el **mecanismo**
(abstención ante ausencia total, `unsupported = 0`, claims verdaderos con
inferencia errada, confianza que no informa) y el hallazgo de que **H4 depende
de la representación** (25,0 % vs 98,8 %).


---

## 19. Tres modelos: `medgemma:4b` y `llama3.1:8b` (local, Ollama 0.30.7 — solo desarrollo)

> **Aviso de procedencia.** Toda esta sección se ejecutó en **local con Ollama
> 0.30.7**. El motor forma parte del experimento: la re-ejecución en el servidor
> (0.40.6) con la cohorte de 12 sujetos cambia el **nivel** de abstención en las
> condiciones parciales y **elimina** el atajo de la temperatura de `llama3.1:8b`
> en `C4`. Las cifras citables son las de
> [`escala_dreamt.md`](escala_dreamt.md) §7; esta sección se conserva como
> registro de desarrollo y de las correcciones de medición que se descubrieron
> aquí (§19.2).

El piloto de §18 se hizo con un solo modelo. Si todo el resultado dependiera de
`qwen2.5:7b`, un revisor podría atribuir el mecanismo a ese modelo y no a la
tarea. Esta sección repite el piloto **con los mismos cuatro sujetos, las mismas
ventanas y los mismos prompts** (960 llamadas por modelo) con dos modelos más de
familias distintas.

**Qué se puede y qué no se puede leer aquí.**

- **Sí**: si el comportamiento (abstención, respaldo de claims, certeza,
  colapso de clases) es una propiedad de la tarea o del modelo.
- **No**: una clasificación de modelos. `medgemma:4b` es un modelo de 4,3 B
  (gemma3, Q4_K_M, contexto 131 k), `llama3.1:8b` es de 8 B con plantilla
  instruct nativa y `qwen2.5:7b` de 7 B; la diferencia de tamaño y de familia
  impide ordenarlos por calidad.

### 19.1. Ejecución y fidelidad de formato

| Modelo | Brazo | Respuestas | Inválidas | Abstención válida | Ritmo |
|---|---|---:|---:|---:|---:|
| `qwen2.5:7b` | numérico | 480 | **0** (0,0 %) | 180 | ≈2 s/llamada |
| `qwen2.5:7b` | texto | 480 | 3 (0,6 %) | 249 | |
| `medgemma:4b` | numérico | 480 | **45** (9,4 %) | 183 | ≈1,1 s/llamada |
| `medgemma:4b` | texto | 480 | **66** (13,8 %) | 130 | |
| `llama3.1:8b` | numérico | 480 | **0** (0,0 %) | 79 | ≈3,5 s/llamada |
| `llama3.1:8b` | texto | 480 | **0** (0,0 %) | 60 | |

Las 111 inválidas de `medgemma` no son fallos de contenido sino de **forma**:

| Defecto | Ocurrencias | Ejemplo |
|---|---:|---|
| `direction: "high"` / `"low"` (magnitud escrita en el campo de dirección) | 97 | `{"direction": "high", "magnitude": "high"}` |
| `channel: "fraction"` (parte de un nombre compuesto) | 14 | `{"metric": "Movement fraction", "channel": "fraction"}` |
| `channel: "activity"` (ídem, en `qwen`) | 3 | `{"channel": "activity"}` |

Se **evalúa con `--recovery`**: la decisión y la confianza se conservan, los
claims mal formados se descartan y el registro queda marcado (`recovered`). En
`llama3.1:8b` no hubo nada que recuperar. No se añade un alias post hoc para
`direction: "high"`, por el mismo criterio que en el brazo textual de WESAD: si
se quiere medir «¿detecta el modelo una magnitud anómala?», merece un tipo de
claim diseñado antes de recoger datos. **Coste declarado**: el análisis de claims
de `medgemma` cubre solo los registros válidos (86–91 % según el brazo).

### 19.2. Dos correcciones de medición descubiertas aquí

Al comparar los brazos textuales apareció un `unsupported` de 55 % en
`medgemma` que **no era alucinación**. Dos defectos del verificador:

1. **`verify_claims` normalizaba el prompt con el vocabulario del dataset y el
   claim sin él.** Todo claim que citara la etiqueta legible que el modelo había
   leído («Movement fraction», «Mean conductance») se contaba como
   `unsupported` —«métrica ausente»— aunque estuviera en la entrada. Afectaba
   solo a los brazos que pasan `extra_aliases` (DREAMT): los análisis de WESAD
   no los pasan, así que **ninguna cifra publicada de WESAD cambia**.
   Efecto en `qwen` textual `C0`: respaldo 0,496 → **0,564**.
2. **Alias incompletos**: `eda conductance`, `conductance`, `skin conductance`,
   `mean_hr`, `acc_variability`, `eda_tonic_level`, `acc_95th_percentile`… y la
   forma con guion bajo de una etiqueta (`acc_mean`). Se añadieron a
   `SLEEP_METRIC_ALIASES` y `normalize_metric` reintenta con espacio ↔ guion
   bajo **solo** en vocabularios con `extra_aliases`.

Para que no vuelva a ocurrir, cada bloque de claims incluye ahora
`unrecognized_metric_names`: cuántos claims citan un nombre que el verificador
no conoce. Un `unsupported` con nombre reconocido es **evidencia ausente**; sin
nombre reconocido es **vocabulario**. Tras la corrección el residual es **1
claim de 2.880** («Skin conductance activity»); `llama3.1:8b` no produjo
ninguno.

### 19.3. Corrección y cobertura (4 sujetos × 20 ventanas por condición)

Media entre los cuatro sujetos de la tasa por sujeto. `acc*` = accuracy con la
abstención contada como error.

**Vista numérica**

| Condición | `qwen2.5:7b` acc / acc\* / cob. | `medgemma:4b` acc / acc\* / cob. | `llama3.1:8b` acc / acc\* / cob. |
|---|---|---|---|
| `C0_clean` | 0,300 / 0,300 / 100 % | **0,412** / 0,412 / 100 % | 0,350 / 0,350 / 100 % |
| `C1_absent_eda` | 0,415 / 0,312 / 75,0 % | 0,438 / 0,425 / 97,5 % | **0,450** / **0,450** / 100 % |
| `C2_degenerate_eda` | 0,338 / 0,338 / 100 % | 0,423 / 0,375 / 88,7 % | 0,438 / 0,438 / 100 % |
| `C3_absent_hrv` | 0,300 / 0,300 / 100 % | 0,288 / 0,212 / 73,8 % | 0,312 / 0,312 / 100 % |
| `C4_absent_core` | — / — / 0 % | — / — / 0 % | 0,353 / 0,175 / **48,8 %** |
| `NC_absent` | — / — / 0 % | — / — / 0 % | — / — / 0 % |

**Vista textual**

| Condición | `qwen2.5:7b` acc / acc\* / cob. | `medgemma:4b` acc / acc\* / cob. | `llama3.1:8b` acc / acc\* / cob. |
|---|---|---|---|
| `C0_clean` | 0,275 / 0,275 / 100 % | 0,283 / 0,275 / 96,3 % | **0,362** / **0,362** / 100 % |
| `C1_absent_eda` | 0,250 / 0,013 / 1,3 % | 0,302 / 0,263 / 87,5 % | 0,312 / 0,312 / 100 % |
| `C2_degenerate_eda` | 0,312 / 0,312 / 100 % | 0,301 / 0,287 / 95,0 % | 0,350 / 0,350 / 100 % |
| `C3_absent_hrv` | 0,241 / 0,212 / 87,5 % | 0,225 / 0,175 / 78,7 % | 0,263 / 0,263 / 100 % |
| `C4_absent_core` | — / — / 0 % | — / — / 0 % | 0,250 / 0,250 / **100 %** |
| `NC_absent` | — / — / 0 % | — / — / 0 % | — / — / 0 % |

**Abstención: la tabla que decide H4**

| Ausencia | `qwen` num. | `qwen` txt. | `medgemma` num. | `medgemma` txt. | `llama` num. | `llama` txt. |
|---|---:|---:|---:|---:|---:|---:|
| EDA (`C1`) | 25,0 % | **98,8 %** | 2,5 % | 12,5 % | 0,0 % | 0,0 % |
| Cardíaco (`C3`) | 0,0 % | 12,5 % | **26,3 %** | 21,2 % | 0,0 % | 0,0 % |
| Núcleo (`C4`) | 100 % | 100 % | 100 % | 100 % | **51,2 %** | **0,0 %** |
| Todo (`NC_absent`) | 100 % | 100 % | 100 % | 100 % | **100 %** | **100 %** |

### 19.4. Lo que no depende del modelo: nada disponible

Solo hay **un invariante** en los 6 brazos (3 modelos × 2 vistas): cuando
**todas** las features están a `not available` (`NC_absent`), los tres modelos
se abstienen el **100 %** de las veces. Cobertura 0, cero claims.

El resultado de dos modelos que se adelantó en la primera versión de esta
sección —«con el núcleo ausente (`C4`) también es 100 %»— **no se sostiene con
el tercero** en el motor 0.30.7: `llama3.1:8b` decide en `C4` usando la única
señal que queda.

> **Corrección (servidor, 0.40.6).** Con el motor del servidor `llama3.1:8b`
> **también se retira** en `C4`: abstención 1,000 en numérico y 0,952 en texto.
> El atajo de la temperatura era del motor, no del modelo. `C4` deja de depender
> del modelo y pasa a ser parte del invariante compartido junto con
> `NC_absent`; lo que discrimina entre modelos es la **ausencia parcial**
> (`C1`/`C2`/`C3`). Ver [`escala_dreamt.md`](escala_dreamt.md) §7.4 y §7.6.

| `C4` (EDA + cardíaco + ACC a `NaN`, temperatura intacta) | numérico | texto |
|---|---:|---:|
| `qwen2.5:7b` / `medgemma:4b` | 100 % abstención | 100 % abstención |
| `llama3.1:8b` | 51,2 % abstención, resto `n2` con confianza 80 | **0 % abstención**, mayoritariamente `n1` |

Su razonamiento, textual: *«Minimal movement is not directly available, but the
high skin temperature standard deviation suggests stable sleep. Heart rate and
other features are unavailable.»* Es decir: **convierte la ausencia de evidencia
en una decisión a partir del canal que queda**, y ese canal es justamente el
confusor de orden de la noche que §14.2 excluyó de la tabla de firma.

- **Accuracy 0,353** en `C4` numérico: la más alta de toda la matriz numérica.
  Su **balanced es 0,250 = azar exacto** (predice `n2` a todo lo que decide).
  Tercera vez en el proyecto que la accuracy premia a un clasificador degenerado.
- En texto predice `n1` (recall 0,75) y acierta 0,250: azar.
- La certeza lingüística baja a −10,5 en esas respuestas, así que **el lenguaje
  sí refleja la duda** aunque la decisión no la respete: es H2b otra vez.

### 19.5. Lo que sí depende: qué canal se echa en falta

- `qwen` percibe la ausencia de EDA **solo en texto** (25,0 % → 98,8 %).
- `medgemma` apenas la percibe (2,5 %) pero **sí** nota el bloque cardíaco
  ausente (26,3 % numérico, 21,2 % textual), que `qwen` numérico ignora (0 %).
- `llama3.1:8b` **no percibe ninguna ausencia parcial** (0,0 % en `C1`, `C2` y
  `C3`, en las dos vistas) y en `C4` depende de la representación (51,2 %
  numérico frente a 0,0 % textual).

Cada modelo reacciona al canal que **no** usa para decidir: `qwen` textual
colapsa a `n2`; `medgemma` numérico decide con movimiento (recall `wake` 0,65) y
nota cuando falta el cardíaco; `llama3.1:8b` acepta cualquier valor presente
como evidencia. **Conclusión: la detección de ausencia es una propiedad del par
(modelo, canal, representación), y el único umbral compartido es «no hay
ninguna feature».** H4 no se puede enunciar como «el LLM se abstiene cuando
falta evidencia» sin especificar qué evidencia, qué formato y qué modelo.

La re-ejecución en el servidor conserva el fondo y lo precisa: los umbrales
compartidos son **dos** condiciones (`C4` y `NC_absent`) y lo que depende del
par son las ausencias **parciales**.

### 19.6. Claims: verdad, coherencia y una excepción

Respaldo (media por sujeto) y coherencia con la decisión:

| Vista | Condición | `qwen` resp./coh. | `medgemma` resp./coh. | `llama` resp./coh. |
|---|---|---|---|---|
| numérico | `C0_clean` | 0,679 / 0,719 | 0,690 / **0,969** | 0,585 / 0,963 |
| numérico | `C1_absent_eda` | **0,765** / 0,803 | 0,503 / 0,961 | 0,589 / 0,933 |
| numérico | `C2_degenerate_eda` | 0,631 / 0,799 | 0,711 / 0,967 | 0,525 / 0,949 |
| numérico | `C3_absent_hrv` | 0,649 / **0,508** | 0,693 / 0,942 | 0,623 / 0,972 |
| numérico | `C4_absent_core` | sin claims | 0,388 | **0,335** (27/42 contradichos) |
| texto | `C0_clean` | 0,565 / 0,879 | 0,598 / **0,980** | 0,482 / 0,921 |
| texto | `C1_absent_eda` | 0,500 (2 claims) | 0,436 / 1,000 | 0,539 / 0,977 |
| texto | `C2_degenerate_eda` | 0,637 / 0,793 | 0,537 / 0,956 | 0,446 / 0,928 |
| texto | `C3_absent_hrv` | 0,489 / 0,920 | 0,472 / 0,975 | 0,571 / 0,887 |
| texto | `C4_absent_core` | sin claims | 0,560 | 0,550 |

- **En limpio los tres leen igual de bien** en numérico (0,679 / 0,690 / 0,585):
  las diferencias de decisión no son de percepción.
- **`unsupported` ≈ 0 en los tres** (1 nombre no reconocido en 2.880 claims):
  ninguno cita un canal que no esté en la entrada. H5 se replica con un tercer
  modelo.
- **La coherencia separa dos modos de fallo**: `medgemma` y `llama` son
  coherentes casi siempre (0,93–0,98) —sus claims apoyan su decisión—; `qwen`
  numérico cae a **0,508** en `C3`. Uno se equivoca de forma consistente, el
  otro de forma inconsistente. En ambos casos la decisión puede ser errónea.
- **Excepción a «los claims son verdaderos»**: en `C4` numérico los claims de
  `llama3.1:8b` son **27 de 42 contradichos** (0,335 respaldados). Cuando la
  única señal disponible es un confusor (temperatura), el modelo no solo decide:
  además **afirma en la dirección equivocada**. H5 no es «los claims siempre son
  ciertos», es «los claims son localmente ciertos **sobre la evidencia que el
  modelo elige mirar**».

### 19.7. ¿Mejora alguno de los otros dos modelos la tarea?

- `medgemma:4b` sube a **0,412** en limpio (Δ +0,112 sobre `qwen`) y coincide
  con el balanced de la referencia supervisada (0,418), pero con un predictor de
  **dos clases**: recall `wake` 0,65, `n2` **1,00**, `n1` 0,00, `rem` 0,00. Con
  20 ventanas por clase, 13 + 20 aciertos de 80 son exactamente 0,4125: el
  empate es aritmética del colapso.
- `llama3.1:8b` se queda en 0,350 pero **reparte más las decisiones** (`n1`
  recall 0,30 numérico y **0,50 textual**, frente a 0,10 de `qwen` y 0,00 de
  `medgemma`). Es el único que se acerca a las clases difíciles, y a la vez el
  que **menos se abstiene**. El techo de §18.1 (0,494 / 0,418) no se alcanza en
  ninguna configuración.
- **`rem` tiene recall 0 en los tres modelos y en las dos vistas** (0,00 de 12
  configuraciones): con 18 features de muñeca a 60 s la clase no se detecta, y
  el «modelo» efectivo es `wake`/`n1`/`n2`.
- **H2a no mejora con ningún modelo**: `medgemma` responde 95 constante (AUROC
  0,500); `llama3.1:8b` responde 80 en 79 de 80 ventanas (AUROC 0,518) con
  sobreconfianza **+0,45**. Tres familias, tres confianzas degeneradas.

### 19.8. Limitaciones de esta comparación

1. **4 sujetos, sin intervalos** (regla de §11). El informe guarda media y rango
   `[mín–máx]` entre sujetos; un bootstrap sobre cuatro sujetos fingiría
   precisión.
2. **No es un ranking de modelos**: 4,3 B, 7 B y 8 B, familias distintas.
3. **Motor**: los seis brazos se ejecutaron con **Ollama 0.30.7 local**. Sirven
   para compararse entre sí; no se mezclan con el corpus de WESAD (0.40.6). La
   cohorte de escala se re-ejecutó en el servidor (0.40.6) y es la que se cita;
   el contraste entre motores está en `escala_dreamt.md` §7.6 y muestra que el
   nivel de abstención **no** es transportable entre motores.
4. **Claims de `medgemma` sobre registros válidos** (86–91 %): las 111 inválidas
   aportan decisión, no claims.
5. **El vocabulario del verificador creció en esta sesión** (§19.2). El reparto
   verdad/coherencia de los tres modelos se calcula con la tabla ampliada y se
   declara como decisión post hoc, igual que los alias de canal de WESAD.
6. **`rem` sin recall** en los tres modelos: la tarea medida es, en la práctica,
   `wake`/`n1`/`n2`.

### 19.9. Reproducción

```sh
# piloto con un modelo (960 llamadas, reanudable)
python main.py --proc GENERATE --experiment dreamt --model llama3.1:8b

# evaluación de cada modelo (sin LLM)
python main.py --proc EVALUATE --experiment dreamt --model llama3.1:8b \
  --representation both --recovery

# comparación lado a lado (recalcula por sujeto con el mismo evaluador)
python scripts/compare_dreamt_models.py \
  --models qwen2.5:7b,medgemma:4b,llama3.1:8b \
  --representation both --recovery \
  --output results/output/DREAMT/PHASE2/pilot/model_comparison.json
```

---

## 20. Escala de la cohorte (12 sujetos)

El piloto de 4 sujetos deja una **dirección**, no una cifra con precisión. La
ampliación está preregistrada en [`escala_dreamt.md`](escala_dreamt.md), con su
cohorte **congelada antes de generar**: 12 sujetos elegidos por disponibilidad de
datos (≥ 12 ventanas puras por clase), excluyendo desarrollo y piloto, y
repartidos a lo largo del rango de identificadores. Diseño: 10 ventanas por
clase y sujeto, las mismas 6 condiciones y las dos vistas → 5.760 llamadas por
modelo, con intervalos bootstrap **sobre sujetos**.

Las cifras del piloto (§18–§19) **no se reescriben** con la escala: son dos
corpus con `n` distinto por sujeto y no se fusionan. La escala se ejecutó en el
**servidor** (Ollama 0.40.6) y es la que se cita: 17.280 respuestas (3 modelos ×
5.760). Los resultados completos, con intervalos, están en
[`escala_dreamt.md`](escala_dreamt.md) §7. En una línea:

1. **Invariante compartido**: solo «nada disponible» (`C4` y `NC_absent`), 100 %
   de abstención en los seis brazos.
2. **Tres modos de fallo distintos** bajo la misma degradación: `qwen2.5:7b` se
   retira en proporción a lo que falta (y pierde toda la utilidad en `C1`),
   `llama3.1:8b` **no se retira nunca** y queda al nivel del azar en todas las
   condiciones, y `medgemma:4b` **selecciona**: se retira sobre todo en `C3` y
   conserva la mejor corrección sobre lo que decide (balanced 0,417
   [0,333; 0,491] con cobertura 0,156).
3. **Ningún brazo se acerca a la referencia** (0,408 / 0,362) en limpio: `acc*`
   0,315 / 0,298 / 0,304, y `rem` sigue con recall ≈ 0 en los tres.
4. **El atajo de la temperatura no replica**: con el motor del servidor
   `llama3.1:8b` se abstiene 1,000 / 0,952 en `C4`. Era un artefacto del 0.30.7
   local, y por eso las cifras de esta sección no se citan.
