import json, uuid
import joblib
import pandas as pd
from cricket_ai.utils.config import BEST_MODEL_PATH, META_PATH, DATA_DIR
from cricket_ai.pipelines.preprocess import preprocess_for_inference
from cricket_ai.utils.logger import get_logger

log = get_logger()

def _load_model_and_meta():
    model = joblib.load(BEST_MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

def predict_from_df(df: pd.DataFrame):
    model, meta = _load_model_and_meta()
    df_proc, dropped = preprocess_for_inference(df)

    # align columns with training
    X = pd.get_dummies(df_proc, drop_first=True)
    for col in meta["feature_columns"]:
        if col not in X.columns:
            X[col] = 0
    X = X[meta["feature_columns"]]

    # predict + confidence (use predict_proba if available)
    if hasattr(model, "predict_proba"):
        probablity = model.predict_proba(X)[:, 1]
        pred = (probablity >= 0.5).astype(int)
        confidence = probablity
    else:
        pred = model.predict(X)
        # fallback: fake confidence = 0.5 or distance-based (keep simple)
        confidence = [0.5] * len(pred)

    out = df_proc.copy()
    out["prediction"] = pred
    out["confidence"] = confidence

    # save predictions file with id
    pid = uuid.uuid4().hex[:8]
    save_path = DATA_DIR / f"predictions_{pid}.csv"
    out.to_csv(save_path, index=False)
    meta_info = {"prediction_id": pid, "dropped_rows_due_to_filter": dropped, "predictions_file": str(save_path)}
    log.info(f"Saved predictions -> {save_path}")
    return out, meta_info
