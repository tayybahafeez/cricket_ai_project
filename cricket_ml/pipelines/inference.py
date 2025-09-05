import json, uuid
import joblib
import pandas as pd
from pathlib import Path
from cricket_ml.utils.config import BEST_MODEL_PATH, META_PATH, DATA_DIR
from cricket_ml.pipelines.preprocess import preprocess_for_inference
from cricket_ml.utils.logger import get_logger

log = get_logger()

def _load_model_and_meta():
    if not BEST_MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Best model or meta file not found. Train model first.")
    model = joblib.load(BEST_MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

def predict_from_df(df: pd.DataFrame):
    model, meta = _load_model_and_meta()

    # Preprocess only for inference
    df_proc = preprocess_for_inference(df)
    
    # Align columns with training
    X = pd.get_dummies(df_proc, drop_first=True)
    for col in meta["feature_columns"]:
        if col not in X.columns:
            X[col] = 0
    X = X[meta["feature_columns"]]

    # Make predictions + confidence
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        confidence = prob
    else:
        pred = model.predict(X)
        confidence = [0.5] * len(pred)

    out = df_proc.copy()
    out["prediction"] = pred
    out["confidence"] = confidence

    # Save predictions CSV
    pid = uuid.uuid4().hex[:8]
    save_dir = DATA_DIR / "predictions"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"predictions_{pid}.csv"
    out.to_csv(save_path, index=False)

    meta_info = {
        "prediction_id": pid,
        "predictions_file": str(save_path),
        "model_name": type(model).__name__,   # ✅ this fixes "unknown"
        "num_rows": len(df_proc),
        "dropped_rows_due_to_filter": len(df) - len(df_proc),
    }

    log.info(f"Saved predictions -> {save_path}")
    return out, meta_info
