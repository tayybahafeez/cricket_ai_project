import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from cricket_ml.pipelines.preprocess import preprocess_for_training
from cricket_ml.utils.config import BEST_MODEL_PATH, META_PATH, DATA_DIR
import logging

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def train(data_path: str) -> Dict:
    df = pd.read_csv(data_path)
    log.info(f"Loaded dataset with {len(df)} rows")

    if "won" not in df.columns:
        raise ValueError("Training data must include a binary target column 'won'.")

    df = preprocess_for_training(df)

    X = df.drop(columns=["won"])
    y = df["won"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, n_jobs=None, random_state=42),
            {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 10], "min_samples_split": [2, 5]},
        ),
    }

    best = {"name": None, "estimator": None, "f1": -1.0, "best_params": None}
    summary = {}

    for name, (est, grid) in models.items():
        log.info(f"Training {name}...")
        gs = GridSearchCV(est, grid, scoring="f1", cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)

        m = gs.best_estimator_
        y_pred = m.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "best_params": gs.best_params_,
        }
        summary[name] = metrics
        log.info(f"{name} metrics: {metrics}")

        if metrics["f1"] > best["f1"]:
            best.update({"name": name, "estimator": m, "f1": metrics["f1"], "best_params": gs.best_params_})

    # Save best model
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["estimator"], BEST_MODEL_PATH)

    # Save metadata
    meta = {
        "model_name": best["name"],
        "feature_columns": list(X.columns),
        "best_params": best["best_params"],
        "data_path": str(Path(data_path).resolve()),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Saved best model: {best['name']} @ {BEST_MODEL_PATH}")
    return {"best": best["name"], "summary": summary, "meta_path": str(META_PATH)}


if __name__ == "__main__":
    dataset_path = DATA_DIR / "cricket_dataset_cleaned.csv"
    if not dataset_path.exists():
        log.error(f"Dataset not found at {dataset_path}")
        exit(1)

    result = train(str(dataset_path))

# Log the training summary instead of printing
log.info("=== Training Summary ===")
log.info(f"Best Model: {result['best']}")
for model_name, metrics in result["summary"].items():
    log.info(f"\nModel: {model_name}")
    for metric, value in metrics.items():
        log.info(f"  {metric}: {value}")

