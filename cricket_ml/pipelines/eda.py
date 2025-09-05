"""
EDA Script for Cricket Match Outcome Prediction
------------------------------------------------
Performs Exploratory Data Analysis on the cricket dataset:
- Validates dataset (missing, negative, impossible values)
- Generates 4 meaningful visualizations
- Saves plots in datasets/visuals/
"""

import logging
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_data(df: pd.DataFrame) -> None:
    """Validate dataset: check missing, negative, and impossible values."""
    logging.info("Validating dataset...")

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        logging.warning(f"Missing values detected:\n{missing[missing > 0]}")
    else:
        logging.info("No missing values found.")

    # Negative values
    numeric_cols = df.select_dtypes(include=["number"]).columns
    negatives = (df[numeric_cols] < 0).sum()
    if negatives.any():
        logging.warning(f"Negative values found in columns:\n{negatives[negatives > 0]}")
    else:
        logging.info("No negative values found.")

    # Impossible values
    if (df['wickets'] > 10).any():
        logging.warning("Some rows have wickets > 10")
    if (df['balls_left'] > 120).any():
        logging.warning("Some rows have balls_left > 120")

    # Basic stats
    logging.info("Dataset statistics:\n%s", df.describe())


def create_comprehensive_visuals(df: pd.DataFrame, output_dir: Path):
    """Generate 4 meaningful visualizations."""
    output_dir.mkdir(exist_ok=True)

    # 1. Outcome Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='won', data=df)
    plt.title("Match Outcome Distribution")
    plt.xlabel("Won (1=Yes, 0=No)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "outcome_distribution.png")
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[['total_runs', 'wickets', 'target', 'balls_left', 'won']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    # 3. Win Rate vs Pressure (balls_left & target bins)
    df['balls_bin'] = pd.cut(df['balls_left'], bins=3)
    df['target_bin'] = pd.cut(df['target'], bins=3)
    win_rate_pressure = df.groupby(['balls_bin','target_bin'])['won'].mean().unstack()
    plt.figure(figsize=(8,6))
    sns.heatmap(win_rate_pressure, annot=True, fmt=".2f", cmap='YlGnBu')
    plt.title("Win Rate by Balls Left & Target")
    plt.tight_layout()
    plt.savefig(output_dir / "winrate_pressure.png")
    plt.close()
    df.drop(columns=['balls_bin','target_bin'], inplace=True)

    # 4. Total Runs vs Balls Left (Scatter)
    plt.figure(figsize=(8,6))
    colors = df['won'].map({0:'red', 1:'green'})
    plt.scatter(df['balls_left'], df['total_runs'], c=colors, alpha=0.6)
    plt.xlabel("Balls Left")
    plt.ylabel("Total Runs")
    plt.title("Total Runs vs Balls Left (Red=Lost, Green=Won)")
    plt.tight_layout()
    plt.savefig(output_dir / "runs_vs_balls_scatter.png")
    plt.close()

    logging.info("Visualizations saved in %s", output_dir)


def main():
    try:
        # Paths
        base_path = Path(__file__).resolve().parents[2]  # repo root
        dataset_path = base_path / "datasets" / "cricket_dataset.csv"
        visuals_path = base_path / "datasets" / "visuals"

        if not dataset_path.exists():
            logging.error("Dataset not found at %s", dataset_path)
            sys.exit(1)

        # Load dataset
        logging.info("Loading dataset from %s", dataset_path)
        df = pd.read_csv(dataset_path)

        # Validate dataset
        validate_data(df)

        # Generate visualizations
        create_comprehensive_visuals(df, visuals_path)

        logging.info("EDA completed successfully!")

    except Exception as e:
        logging.error("Error during EDA: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
