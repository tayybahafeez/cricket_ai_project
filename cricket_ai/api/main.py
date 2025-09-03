from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from cricket_ai.pipelines.predict import predict_from_df
from cricket_ai.llm.explain import explain
from cricket_ai.utils.logger import get_logger
from cricket_ai.utils.config import DATA_DIR

app = FastAPI(title="Cricket Match Prediction API")
log = get_logger()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed CSV or unreadable content.")

    try:
        preds, meta = predict_from_df(df)
        # Don't return entire CSV; return counts + path token
        return JSONResponse({
            "rows_received": len(df),
            "rows_predicted": len(preds),
            "dropped_by_filter": meta["dropped_rows_due_to_filter"],
            "prediction_id": meta["prediction_id"],
            "predictions_file": meta["predictions_file"]
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.get("/explain/{prediction_id}")
def explain_prediction(prediction_id: str):
    # Load the saved predictions file
    path = DATA_DIR / f"predictions_{prediction_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prediction ID not found.")

    df = pd.read_csv(path)
    if df.empty:
        raise HTTPException(status_code=400, detail="No rows to explain for this ID.")

    # Take first row for the demo explanation
    row = df.iloc[0].to_dict()
    confidence = float(row.get("confidence", 0.5))
    # Build small context
    ctx = {
        "total_runs": row.get("total_runs"),
        "wickets": row.get("wickets"),
        "target": row.get("target"),
        "balls_left": row.get("balls_left"),
        "current_run_rate": row.get("current_run_rate"),
        "required_run_rate": row.get("required_run_rate"),
        "prediction": int(row.get("prediction", 0)),
    }
    text = explain(ctx, confidence)
    return {"prediction_id": prediction_id, "context": ctx, "confidence": confidence, "explanation": text}
