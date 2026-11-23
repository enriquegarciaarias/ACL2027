# Datos de entrada — qué debe existir para ejecutar el pipeline

**Actualizado:** 2026-11-11

Todo lo necesario para ejecutar el pipeline vive **dentro del proyecto**. No hay
referencias a otros repositorios (PSRI, SLIP) ni a rutas temporales: los
adaptadores de los experimentos externos leen de `results/input/EXTERNAL/`.

`results/` está **gitignorado** (por tamaño), así que al servidor se lleva el
árbol de trabajo completo, no solo un `git clone`:

```bash
rsync -av --exclude .venv --exclude .git --exclude __pycache__ \
  /ruta/local/ACL2027/ usuario@servidor:/ruta/ACL2027/
```

---

## 1. Inventario

| Dataset | Qué se necesita | Ruta dentro del proyecto | Tamaño | Origen |
|---|---|---|---:|---|
| **WESAD** | `S*/S*.pkl` (una carpeta por sujeto) | `results/input/WESAD/` | ~12,9 GB | dataset original |
| **PhysioNet/CinC 2017** | `training2017/` con `.mat`, `.hea` y `REFERENCE.csv` | `results/input/EXTERNAL/physionet2017/training2017/` | ~230 MB | `training2017.zip` (95 MB) de PhysioNet |
| **K-EmoCon** | `kemocon_feature_table.csv` y `kemocon_period_assignments.csv` | `results/input/EXTERNAL/KEMOCON/` | 5,1 MB | tablas derivadas (PSRI) |
| **DREAMT** | `data_64Hz/S0xx_whole_df.csv` (rama *wearable*; las etiquetas de sueño viajan ahí) | `results/input/dreamt/` | 14 GB | `DREAMT.zip` de PhysioNet (**licencia restringida 1.5.0**) |

## 2. Por qué los externos no pasan por `INGEST`

- **Fase 1 (`INGEST`)** implementa el contrato de entrada de WESAD: una carpeta
  por sujeto con su `.pkl` y señales multicanal de pecho y muñeca.
- **PhysioNet 2017** son 8.528 registros independientes de ECG de una
  derivación (9–61 s), sin sujetos: no encaja en ese contrato. El adaptador
  `sources/external/physionet2017.py` los lee directamente y hace su propio
  muestreo y split por CRC32.
- **K-EmoCon** no necesita ingesta en este proyecto: se usan las **tablas ya
  derivadas** (features por ventana de 5 s y asignación de periodos). Ingerir de
  nuevo las señales crudas de muñeca no aportaría nada y duplicaría el trabajo.
- **DREAMT** trae una rejilla de épocas de 30 s y etiquetas de sueño por fichero,
  no un `.pkl` por sujeto: el adaptador `sources/external/dreamt.py` ancla la
  rejilla, construye ventanas puras de 60 s y extrae las features. La rama
  `data_64Hz` (14 GB) basta porque las etiquetas PSG viajan en ella; la de
  100 Hz (101 GB) solo sería necesaria para HRV real (desviación D7 del
  preregistro).

Consecuencia: `results/input/EXTERNAL/` contiene **datos de entrada** (no
artefactos), y los artefactos siguen yendo a `results/output/{WESAD,
PHYSIONET2017, KEMOCON}/`. Si prefieres que los datos externos estén en
`results/output`, basta cambiar la ruta (ver §4); por defecto se tratan como
entrada porque son materia prima del experimento.

## 3. Cómo obtener cada cosa

**WESAD** (si no está):

```bash
# estructura esperada
results/input/WESAD/S2/S2.pkl ... S17/S17.pkl
```

**PhysioNet/CinC 2017**:

```bash
cd results/input/EXTERNAL/physionet2017
curl -sLO https://physionet.org/files/challenge-2017/1.0.0/training2017.zip
unzip -q training2017.zip && rm training2017.zip
ls training2017/REFERENCE.csv
```

**K-EmoCon** (tablas derivadas por PSRI):

```bash
cp <PSRI>/results/output/KEMOCON/kemocon_feature_table.csv \
   <PSRI>/results/output/KEMOCON/kemocon_period_assignments.csv \
   results/input/EXTERNAL/KEMOCON/
```

**DREAMT** (PhysioNet, licencia restringida: hay que aceptarla con la cuenta):

```bash
# descarga autenticada desde https://physionet.org/content/dreamt/1.0.0/
cd results/input
unzip -q ~/DREAMT.zip            # crea dreamt/data_64Hz/*.csv y data_100Hz/
ls dreamt/data_64Hz/S002_whole_df.csv
```

Solo se necesita `data_64Hz/` (14 GB). Si no se va a usar la rama PSG, la
carpeta `data_100Hz/` (101 GB) puede quedarse fuera: el adaptador no la lee.

## 4. Reubicar sin tocar código

En `config.json` (o en `results/input/{subject}/config.json`, que se fusiona):

```json
"params": {
  "external": {
    "physionet": { "data_dir": "/otra/ruta/training2017" },
    "kemocon":   { "data_dir": "/otra/ruta/KEMOCON" },
    "dreamt":    { "data_dir": "/otra/ruta/dreamt" }
  }
}
```

Vacío (`""`) = ruta por defecto del repositorio. Las banderas `--data-dir`
(también `--psri-dir` en K-EmoCon, por compatibilidad) siguen teniendo
prioridad si usas el script directamente.

## 5. Comprobación antes de lanzar

```bash
python main.py --proc GENERATE --experiment physionet --dry-run
python main.py --proc GENERATE --experiment kemocon  --dry-run
python main.py --proc GENERATE --experiment dreamt   --dry-run
```

El bloque `paths` del plan indica `external_data` y `external_data_exists`, y
`warnings` dice exactamente qué falta:

```json
"paths": {
  "external_data": ".../results/input/EXTERNAL/physionet2017/training2017/training2017",
  "external_data_exists": true,
  "warnings": []
}
```

Los tests no necesitan ninguno de estos datos (usan un dataset sintético).
