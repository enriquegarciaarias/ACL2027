"""
Verificación del confusor natural del movimiento en WESAD (S2).

Comprueba si el movimiento y el EMG actúan como atajos que separan
estrés de reposo independientemente de la fisiología, lo cual invalidaría
H1/H4 si no se controla.

Pasos:
1. SQI por etiqueta (distribución, media, prueba de diferencia)
2. AUC univariante de cada feature vs etiqueta
3. Modelo de referencia con y sin canales de movimiento
   (regresión logística simple + Random Forest)

Ejecución:
    python scripts/verify_motion_confounder.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:  # pragma: no cover - sklearn opcional
    HAS_SKLEARN = False

RESULTS_DIR = Path("results/output/WESAD/INGEST")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_label_id(lbl: Any) -> int | None:
    """Extrae el id numérico de una etiqueta que puede ser dict o int."""
    if isinstance(lbl, dict):
        return int(lbl.get("id"))
    if lbl is not None:
        return int(lbl)
    return None


def main() -> None:
    # Cargar datos de S2
    windows = load_jsonl(RESULTS_DIR / "windows" / "S2.windows.jsonl")
    features = load_jsonl(RESULTS_DIR / "features" / "S2.features.jsonl")
    labels = load_jsonl(RESULTS_DIR / "groundtruth" / "labels.jsonl")

    # Indexar labels por window_id
    label_by_window: dict[str, int] = {}
    for row in labels:
        wid = row.get("window_id")
        lbl = _get_label_id(row.get("label"))
        if wid is not None and lbl is not None:
            label_by_window[str(wid)] = lbl

    # Indexar features por window_id
    feature_by_window: dict[str, dict[str, Any]] = {}
    for row in features:
        wid = row.get("window_id")
        if wid:
            feature_by_window[str(wid)] = row

    print(f"Ventanas: {len(windows)}, Features: {len(features)}, Labels: {len(label_by_window)}")

    # Construir matriz X (features) y y (etiquetas)
    feature_names: list[str] = []
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    sqi_values: list[float] = []

    for row in windows:
        wid = str(row.get("window_id"))
        if wid not in feature_by_window:
            continue
        if wid not in label_by_window:
            continue

        feat_row = feature_by_window[wid]
        features_block = feat_row.get("features", {})

        row_values: list[float] = []
        for modality in ("ecg", "eda", "resp", "temp", "acc", "emg", "bvp",
                         "wrist_eda", "wrist_temp", "wrist_acc"):
            block = features_block.get(modality, {})
            for name in sorted(block.keys()):
                meta = block[name]
                v = meta.get("value")
                if v is not None and isinstance(v, (int, float)):
                    row_values.append(float(v))
                else:
                    row_values.append(float("nan"))
        if not feature_names:
            for modality in ("ecg", "eda", "resp", "temp", "acc", "emg", "bvp",
                             "wrist_eda", "wrist_temp", "wrist_acc"):
                block = features_block.get(modality, {})
                for name in sorted(block.keys()):
                    feature_names.append(f"{modality}.{name}")

        # SQI overall
        quality = feat_row.get("quality", {})
        sqi = quality.get("sqi_overall")
        sqi_values.append(float(sqi) if sqi is not None else float("nan"))

        X_rows.append(row_values)
        y_rows.append(int(label_by_window[wid]))

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    sqi = np.array(sqi_values, dtype=float)

    print(f"\nMatriz X: {X.shape}, Etiquetas: {y.shape}")
    print(f"Etiquetas únicas: {sorted(set(y.tolist()))}")
    vals, counts = np.unique(y, return_counts=True)
    print(f"Conteo por etiqueta: {dict(zip(vals.tolist(), counts.tolist()))}")

    # ── 1. SQI por etiqueta ──
    print("\n" + "=" * 70)
    print("1. SQI por etiqueta")
    print("=" * 70)
    for lbl in sorted(set(y.tolist())):
        mask = y == lbl
        sqi_lbl = sqi[mask]
        valid = sqi_lbl[~np.isnan(sqi_lbl)]
        if len(valid) > 0:
            print(f"  Etiqueta {lbl}: n={len(valid)}, media={valid.mean():.3f}, "
                  f"std={valid.std():.3f}, min={valid.min():.3f}, max={valid.max():.3f}")
        else:
            print(f"  Etiqueta {lbl}: sin datos válidos de SQI")

    # ── 2. AUC univariante ──
    print("\n" + "=" * 70)
    print("2. AUC univariante por feature (vs etiqueta binaria: estrés=2 vs baseline=1)")
    print("=" * 70)

    mask_bin = np.isin(y, [1, 2])
    X_bin = X[mask_bin]
    y_bin = (y[mask_bin] == 2).astype(int)  # 1 = estrés, 0 = baseline

    if HAS_SKLEARN:
        auc_results: list[tuple[str, float]] = []
        for j, name in enumerate(feature_names):
            col = X_bin[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 10:
                continue
            try:
                auc = roc_auc_score(y_bin[valid], col[valid])
                auc_results.append((name, float(auc)))
            except Exception:
                continue

        auc_results.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        print(f"\nTop 15 features por |AUC - 0.5| (mayor = más discriminativa):")
        for name, auc in auc_results[:15]:
            direction = "estrés" if auc > 0.5 else "baseline"
            print(f"  {name:35s} AUC={auc:.3f} (sesgo hacia {direction})")

        # Identificar features de movimiento
        motion_features = [n for n in feature_names if "acc" in n.lower() or "emg" in n.lower()]
        print(f"\nFeatures de movimiento (ACC/EMG):")
        for name, auc in auc_results:
            if name in motion_features:
                direction = "estrés" if auc > 0.5 else "baseline"
                print(f"  {name:35s} AUC={auc:.3f} (sesgo hacia {direction})")

        # ── 3. Modelo de referencia con y sin movimiento ──
        print("\n" + "=" * 70)
        print("3. Modelo de referencia: con y sin canales de movimiento")
        print("=" * 70)

        # Imputar NaN con mediana
        X_imputed = X_bin.copy()
        for j in range(X_imputed.shape[1]):
            col = X_imputed[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 0:
                med = np.median(col[valid])
                col[~valid] = med

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Índices de features de movimiento
        motion_idx = [j for j, name in enumerate(feature_names)
                      if "acc" in name.lower() or "emg" in name.lower()]
        non_motion_idx = [j for j in range(len(feature_names)) if j not in motion_idx]

        print(f"\nFeatures de movimiento: {len(motion_idx)}")
        print(f"Features sin movimiento: {len(non_motion_idx)}")

        for model_name, model in [("Regresión Logística", LogisticRegression(max_iter=1000)),
                                  ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))]:
            print(f"\n--- {model_name} ---")

            # Con todas las features
            preds_all = cross_val_predict(model, X_scaled, y_bin, cv=5, method="predict_proba")[:, 1]
            auc_all = roc_auc_score(y_bin, preds_all)
            acc_all = accuracy_score(y_bin, (preds_all > 0.5).astype(int))
            f1_all = f1_score(y_bin, (preds_all > 0.5).astype(int))
            print(f"  Todas las features:  AUC={auc_all:.3f}, Acc={acc_all:.3f}, F1={f1_all:.3f}")

            # Sin features de movimiento
            if len(non_motion_idx) > 0:
                X_no_motion = X_scaled[:, non_motion_idx]
                preds_nm = cross_val_predict(model, X_no_motion, y_bin, cv=5, method="predict_proba")[:, 1]
                auc_nm = roc_auc_score(y_bin, preds_nm)
                acc_nm = accuracy_score(y_bin, (preds_nm > 0.5).astype(int))
                f1_nm = f1_score(y_bin, (preds_nm > 0.5).astype(int))
                print(f"  Sin movimiento:      AUC={auc_nm:.3f}, Acc={acc_nm:.3f}, F1={f1_nm:.3f}")
                print(f"  Caída AUC:           {auc_all - auc_nm:+.3f}")
                print(f"  Caída F1:            {f1_all - f1_nm:+.3f}")
            else:
                print("  Sin movimiento: no aplica (no hay features no-movimiento)")
    else:
        print("scikit-learn no instalado. Saltando AUC y modelo de referencia.")
        print("Instalar con: pip install scikit-learn")


if __name__ == "__main__":
    main()
