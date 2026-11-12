# Verificación del corpus reconstruido

Procedimiento para comprobar, **sin llamar al LLM**, que el corpus derivado de
`INGEST` es el mismo que alimentó las respuestas ya generadas.

---

## 1. Por qué hace falta

`INGEST` no es idempotente respecto al historial de git. Hasta la sesión
2026-11-12, el commit (`git rev-parse HEAD`) formaba parte de **todos** los
fingerprints:

| Artefacto | Módulo | Componente retirado |
|---|---|---|
| Canónico `.npz` | `ingestion/loader.py` | `"git": git_info().get("commit")` |
| Informe QC | `ingestion/qc.py` | ídem |
| Ventanas | `ingestion/window.py` | ídem |
| Features | `ingestion/features/__init__.py` | ídem |
| Representaciones | `ingestion/represent.py` | ídem |
| Manifiesto del corpus | `ingestion/package.py` | ídem |

Consecuencia: **cualquier commit invalidaba el corpus completo**, incluido un
cambio de documentación. El 2026-11-12 un `python main.py --proc INGEST
--subject WESAD` en el servidor recanonicalizó 14 de 15 sujetos (19 GB de
pickles), recalculó features de las 6 condiciones y regeneró las
representaciones, por un commit que solo tocaba documentación.

Además, los artefactos se escriben en **ruta fija y sin versión**
(`features/S10.features.jsonl`), así que la reconstrucción **sobrescribe** la
versión anterior. Después del run no queda en disco ninguna copia del corpus
antiguo que se pueda comparar.

La única huella que sobrevive son los prompts almacenados en las respuestas:
contienen las 23 features con sus z-scores tal como las vio el modelo.

---

## 2. Qué se corrigió

### 2.1 Fingerprint por contenido de código

`code_fingerprint(scope)` en `sources/common/provenance.py` calcula el SHA-256
de los ficheros que producen el artefacto:

```python
# sources/ingestion/features/__init__.py
_CODE_SCOPE = ("sources/ingestion/features",)   # todo el subpaquete
```

- Un commit de documentación **no** invalida nada.
- Un cambio en un extractor o en el SQI **sí** invalida.
- Los `*_VERSION` (`SCHEMA_VERSION`, `FEATURE_VERSION`,
  `REPRESENTATION_VERSION`) siguen siendo el interruptor explícito.
- El `git` no desaparece: queda como **metadata** en `S*.meta.json` y en
  `corpus_manifest.json`. La información de procedencia se conserva; lo que
  cambia es que ya no forma parte del hash.

La cadena de fingerprints ya propagaba el contenido aguas abajo
(`qc` incluye `canonical_sha256`; `features` incluye `windows_fingerprint`;
`represent` incluye `features_fingerprint`), así que cada paso solo necesita su
propio código.

### 2.2 Ancla en cada respuesta

El barrido de degradación guarda en cada registro:

```json
{"features_sha256": "9f2c…", "...": "..."}
```

SHA-256 del fichero de features exacto (`features/{S}.features.jsonl` o
`degraded_features/{cond}/{S}.features.jsonl`) con el que se construyó el
prompt. A partir de ahora la verificación es comparar dos hashes en O(1), sin
recomponer prompts.

---

## 3. Cómo se verifica

```bash
.venv/bin/python scripts/verify_corpus_rebuild.py; echo "EXIT=$?"
```

Sin `config.json`, sin GPU y sin Ollama: solo lee `results/`.

Para cada respuesta guardada:

| Nivel | Comprobación | Qué cubre |
|---|---|---|
| 1 | `features_sha256` (si existe) | comparación O(1) del fichero de features |
| 2 | `duration_s` | ventanas recalculadas con otro tramo |
| 3 | `prompt` byte a byte | features, referencia intra-sujeto, redondeo, few-shot y plantilla |
| 4 | features parseadas | si (3) falla, distingue deriva de datos de cambio de plantilla |

Salidas:

- `EXIT=0` · todo coincide.
- `EXIT=1` · hay divergencias (el informe indica sujeto, condición, ventana y
  las tres primeras líneas distintas).
- `EXIT=2` · no hay artefactos (el barrido no está en esa máquina).

Informe: `results/output/WESAD/PHASE2/verify_rebuild_report.json`.

---

## 4. Resultado de la verificación (2026-11-12)

Ejecutada en el servidor tras la reconstrucción provocada por el commit
`6f54614`:

```
TOTAL: 5670 registros · prompt idéntico 5670 · distinto 0 · veredicto: IDENTICAL
EXIT=0
```

5.670 = 2.835 respuestas numéricas + 2.835 textuales (12 sujetos de evaluación
× 567 ventanas × 5 condiciones efectivas). La reconstrucción fue
**content-identical**: solo cambiaron los fingerprints, no los datos.

Consecuencia: los resultados de H1–H6 en `docs/resultados_degradacion.md`
siguen válidos sin re-ejecutar el barrido.

---

## 5. Limitaciones

1. **Una reconstrucción más.** Cambiar la composición del fingerprint invalida
   los artefactos existentes una última vez. El siguiente `INGEST` los
   recalculará y, a partir de ahí, será estable frente a commits.
2. **Alcance verificado.** La comprobación cubre el barrido de degradación (el
   experimento primario). No cubre el corpus secundario de
   `llm_responses/` (P2/P4/E12), cuyos prompts se construyen desde las
   representaciones y las transformaciones `NC_*`.
3. **Sensibilidad del prompt a z-scores.** Un desplazamiento uniforme de todas
   las ventanas limpias de un sujeto es invisible en el prompt, porque la
   referencia intra-sujeto se desplaza con ellas. Es una propiedad del diseño
   (E12), no un fallo del verificador; el `features_sha256` sí lo detectaría.

---

## 6. Reproducir

```bash
cd <raíz del proyecto>
source .venv/bin/activate
python -m pytest tests/test_verify_rebuild.py tests/test_fingerprints.py -q
python scripts/verify_corpus_rebuild.py --max-records 5     # humo
python scripts/verify_corpus_rebuild.py                     # completo (~10 s)
```
