from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import json
from cricket_ml.pipelines.inference import predict_from_df, preprocess_for_inference
from cricket_ml.llm.llm_model import explain
from cricket_ml.utils.config import DATA_DIR, PREDICTIONS_DIR, META_PATH, REQUIRED_COLUMNS
from cricket_ml.utils.logger import get_logger
from pathlib import Path


log = get_logger()
app = FastAPI(title="Cricket Match Prediction API")

# Load model metadata once
try:
    with open(META_PATH, "r") as f:
        MODEL_META = json.load(f)
except Exception:
    MODEL_META = {"model_name": "unknown"}
    log.warning("META_PATH not found, model_name set to unknown")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="📂 Upload your CSV file here")):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file only.")
    
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed CSV or unreadable content.")
    
    # ===== Validation =====
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")
    if (df['balls_left'] < 0).any():
        raise HTTPException(status_code=400, detail="balls_left cannot be negative")
    # ======================

    # Preprocess for inference
    df_proc = preprocess_for_inference(df)
    log.debug(f"Original rows: {len(df)}, after preprocessing: {len(df_proc)}")

    if df_proc.empty:
        return JSONResponse({
            "status": "success",
            "predictions_file": None,
            "details": {
                "total rows": len(df),
                "filtered_row": 0,
                "prediction made": 0,
                "model used": MODEL_META.get("model_name", "unknown")
            },
            "message": "No valid rows left after preprocessing. Check 'balls_left' and 'target' values."
        })

    try:
        preds, meta = predict_from_df(df_proc)
        log.debug(f"Meta returned from predict_from_df: {meta}")

        return JSONResponse({
            "status": "success",
            "predictions_file": meta.get("predictions_file"),
            "details": {
                "total rows": len(df),
                "filtered_row": len(df_proc),
                "prediction made": int(preds["prediction"].sum()),
                "model used": meta.get("model_name", MODEL_META.get("model_name", "unknown"))
            }
        })
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction error")


@app.post("/explain/{prediction_id}")
def explain_prediction(prediction_id: str):
    path = PREDICTIONS_DIR / f"predictions_{prediction_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prediction ID not found.")
    
    df = pd.read_csv(path)
    if df.empty:
        raise HTTPException(status_code=400, detail="No rows to explain.")

    row = df.iloc[0].to_dict()
    confidence = float(row.get("confidence", 0.5))
    ctx = {k: row[k] for k in ["total_runs", "wickets", "target", "balls_left",
                                "current_run_rate", "required_run_rate", "prediction"]}
    text = explain(ctx, confidence)
    
    # Add status key
    return {
            "prediction_id": prediction_id,
            "confidence": confidence,
            "explanation": text}
