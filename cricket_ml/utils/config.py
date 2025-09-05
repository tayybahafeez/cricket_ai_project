import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    ENV: str = os.getenv("ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "datasets"
PREDICTIONS_DIR = DATA_DIR / "predictions"   # <--- ADD THIS

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)  # <--- ALSO CREATE IT

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
META_PATH = MODELS_DIR / "model_meta.json"

REQUIRED_COLUMNS = ["total_runs", "wickets", "target", "balls_left"]
