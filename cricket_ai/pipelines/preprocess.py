import pandas as pd
import numpy as np
from typing import Tuple, List
from cricket_ai.utils.config import REQUIRED_COLUMNS
from cricket_ai.utils.logger import get_logger

log = get_logger()

def validate_schema(df: pd.DataFrame) -> List[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Safe copies
    df = df.copy()
    total_balls_in_innings = 120  # 20 overs

    # balls_played could be inferred if needed; we use given columns
    df["balls_played"] = total_balls_in_innings - df["balls_left"]
    df["current_run_rate"] = np.where(
        df["balls_played"] > 0, (df["total_runs"] / df["balls_played"]) * 6, 0.0
    )
    df["required_run_rate"] = np.where(
        df["balls_left"] > 0, (df["target"] / df["balls_left"]) * 6, df["target"] * 6
    )

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.drop(columns=["balls_played"], errors="ignore", inplace=True)
    return df

def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    # keep rows where balls_left < 60 AND target > 120
    return df[(df["balls_left"] < 60) & (df["target"] > 120)].copy()

def preprocess_for_training(df: pd.DataFrame) -> pd.DataFrame:
    # basic cleaning (drop NA rows only on required cols)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df = engineer_features(df)
    return df

def preprocess_for_inference(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    missing = validate_schema(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = engineer_features(df)
    before = len(df)
    df = apply_filter(df)
    return df, before - len(df)
