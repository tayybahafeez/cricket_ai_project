"""
Preprocessing module for Cricket Match Prediction
-------------------------------------------------
- Separate functions for training vs inference
- Training: cleans data, engineers features
- Inference: cleans data, engineers features, applies filters for live predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

REQUIRED_COLUMNS = ["total_runs", "wickets", "target", "balls_left", "won"]


def validate_schema(df: pd.DataFrame) -> list:
    """Check if all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def clean_invalid_values(df: pd.DataFrame, save_invalid: bool = False, output_dir: Path = None) -> pd.DataFrame:
    """
    Remove negative or impossible values, log and optionally save dropped rows.
    """
    df = df.copy()
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_mask = (
        (df["total_runs"] >= 0) &
        (df["wickets"].between(0, 10)) &
        (df["target"] >= 0) &
        (df["balls_left"].between(0, 120))
    )

    df_cleaned = df.loc[valid_mask].copy()
    dropped_rows = df.loc[~valid_mask]
    dropped_count = len(dropped_rows)

    if dropped_count > 0:
        logging.warning(f"Dropped {dropped_count} invalid rows")
        logging.info(f"Details of dropped rows:\n{dropped_rows}")
        if save_invalid and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            invalid_path = output_dir / "invalid_rows.csv"
            dropped_rows.to_csv(invalid_path, index=False)
            logging.info(f"Dropped rows saved to {invalid_path}")

    return df_cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features: current run rate and required run rate."""
    df = df.copy()
    total_balls_in_innings = 120  # 20 overs
    df["balls_played"] = total_balls_in_innings - df["balls_left"]

    # Current run rate
    df["current_run_rate"] = np.where(
        df["balls_played"] > 0,
        (df["total_runs"] / df["balls_played"]) * 6,
        0.0
    )

    # Required run rate
    df["required_run_rate"] = np.where(
        df["balls_left"] > 0,
        (df["target"] / df["balls_left"]) * 6,
        df["target"] * 6
    )

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.drop(columns=["balls_played"], inplace=True)
    return df


def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows for inference (e.g., live predictions).
    Only keep rows relevant for prediction.
    """
    return df[(df["balls_left"] < 60) & (df["target"] > 120)].copy()


def preprocess_for_training(df: pd.DataFrame, save_invalid: bool = True, output_dir: Path = None) -> pd.DataFrame:
    """
    Preprocessing for model training.
    - Cleans invalid rows
    - Engineers features
    - Always retrain if feature engineering changes
    """
    missing = validate_schema(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df_cleaned = clean_invalid_values(df, save_invalid=save_invalid, output_dir=output_dir)
    df_cleaned = engineer_features(df_cleaned)
    return df_cleaned


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess incoming CSV for prediction (inference).

    Steps:
    1. Convert columns to numeric.
    2. Remove invalid rows (negative runs, wickets > 10, balls_left > 120).
    3. Engineer features (current_run_rate, required_run_rate).
    4. Apply inference filter: balls_left < 60 and target > 120.
    """
    df = df.copy()

    # Ensure numeric values
    for col in ["total_runs", "wickets", "target", "balls_left"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN after conversion
    df = df.dropna(subset=["total_runs", "wickets", "target", "balls_left"])

    # Remove impossible values
    df = df[
        (df["total_runs"] >= 0) &
        (df["wickets"].between(0, 10)) &
        (df["target"] >= 0) &
        (df["balls_left"].between(0, 120))
    ]

    # Feature engineering
    total_balls_in_innings = 120
    df["current_run_rate"] = np.where(
        df["balls_left"] < total_balls_in_innings,
        df["total_runs"] / (total_balls_in_innings - df["balls_left"]) * 6,
        0
    )
    df["required_run_rate"] = np.where(
        df["balls_left"] > 0,
        df["target"] / df["balls_left"] * 6,
        df["target"] * 6
    )

    # Apply inference filter
    df = df[(df["balls_left"] < 60) & (df["target"] > 120)].copy()

    # Replace inf or -inf if any
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df

