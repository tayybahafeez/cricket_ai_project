"""
Exploratory Data Analysis module for cricket match prediction.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from cricket_ai.utils.logger import get_logger
from cricket_ai.utils.config import DATA_DIR

log = get_logger()

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Load cricket dataset and perform basic validation.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Validated DataFrame
    """
    log.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Columns: {list(df.columns)}")
    
    # Basic validation
    log.info("Performing data validation...")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        log.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
    else:
        log.info("No missing values found")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        log.warning(f"Found {duplicates} duplicate rows")
    else:
        log.info("No duplicate rows found")
    
    # Check data types
    log.info(f"Data types:\n{df.dtypes}")
    
    # Check for anomalies
    log.info("Checking for data anomalies...")
    
    # Check for negative values where they shouldn't be
    negative_checks = {
        'total_runs': (df['total_runs'] < 0).sum(),
        'wickets': (df['wickets'] < 0).sum(),
        'target': (df['target'] < 0).sum(),
        'balls_left': (df['balls_left'] < 0).sum()
    }
    
    for col, count in negative_checks.items():
        if count > 0:
            log.warning(f"Found {count} negative values in {col}")
        else:
            log.info(f"No negative values in {col}")
    
    # Check for impossible values
    impossible_wickets = (df['wickets'] > 10).sum()
    if impossible_wickets > 0:
        log.warning(f"Found {impossible_wickets} rows with wickets > 10")
    
    impossible_balls = (df['balls_left'] > 120).sum()
    if impossible_balls > 0:
        log.warning(f"Found {impossible_balls} rows with balls_left > 120")
    
    return df

def create_visualizations(df: pd.DataFrame, output_dir: Path = None) -> None:
    """
    Create meaningful visualizations for EDA.
    
    Args:
        df: DataFrame to visualize
        output_dir: Directory to save plots (default: DATA_DIR/visuals)
    """
    if output_dir is None:
        output_dir = DATA_DIR / "visuals"
    
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribution of target variable
    plt.figure(figsize=(10, 6))
    df['won'].value_counts().plot(kind='bar')
    plt.title('Distribution of Match Outcomes')
    plt.xlabel('Won (1=Yes, 0=No)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'outcome_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['total_runs', 'wickets', 'target', 'balls_left', 'won']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Correlation Matrix of Cricket Match Features')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Balls left vs Win probability
    plt.figure(figsize=(12, 6))
    
    # Create bins for balls_left
    df['balls_left_bin'] = pd.cut(df['balls_left'], bins=10, precision=0)
    win_rate_by_balls = df.groupby('balls_left_bin')['won'].mean()
    
    plt.subplot(1, 2, 1)
    win_rate_by_balls.plot(kind='bar')
    plt.title('Win Rate by Balls Remaining')
    plt.xlabel('Balls Left (binned)')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45)
    
    # 4. Target vs Win probability
    plt.subplot(1, 2, 2)
    df['target_bin'] = pd.cut(df['target'], bins=8, precision=0)
    win_rate_by_target = df.groupby('target_bin')['won'].mean()
    win_rate_by_target.plot(kind='bar')
    plt.title('Win Rate by Target Score')
    plt.xlabel('Target Score (binned)')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Scatter plot: Total runs vs Balls left colored by outcome
    plt.figure(figsize=(10, 8))
    colors = ['red' if x == 0 else 'green' for x in df['won']]
    plt.scatter(df['balls_left'], df['total_runs'], c=colors, alpha=0.6, s=20)
    plt.xlabel('Balls Left')
    plt.ylabel('Total Runs')
    plt.title('Total Runs vs Balls Left (Red=Lost, Green=Won)')
    plt.colorbar(label='Outcome')
    plt.tight_layout()
    plt.savefig(output_dir / 'runs_vs_balls_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Visualizations saved to {output_dir}")

def generate_eda_insights(df: pd.DataFrame) -> dict:
    """
    Generate key insights from EDA.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary of insights
    """
    insights = {}
    
    # Basic statistics
    insights['dataset_size'] = len(df)
    insights['win_rate'] = df['won'].mean()
    insights['columns'] = list(df.columns)
    
    # Feature statistics
    insights['feature_stats'] = {
        'total_runs': {
            'mean': df['total_runs'].mean(),
            'std': df['total_runs'].std(),
            'min': df['total_runs'].min(),
            'max': df['total_runs'].max()
        },
        'wickets': {
            'mean': df['wickets'].mean(),
            'std': df['wickets'].std(),
            'min': df['wickets'].min(),
            'max': df['wickets'].max()
        },
        'target': {
            'mean': df['target'].mean(),
            'std': df['target'].std(),
            'min': df['target'].min(),
            'max': df['target'].max()
        },
        'balls_left': {
            'mean': df['balls_left'].mean(),
            'std': df['balls_left'].std(),
            'min': df['balls_left'].min(),
            'max': df['balls_left'].max()
        }
    }
    
    # Key insights
    insights['key_findings'] = []
    
    # Win rate analysis
    overall_win_rate = df['won'].mean()
    insights['key_findings'].append(f"Overall win rate: {overall_win_rate:.2%}")
    
    # High pressure situations (low balls left, high target)
    high_pressure = df[(df['balls_left'] < 30) & (df['target'] > 150)]
    if len(high_pressure) > 0:
        high_pressure_win_rate = high_pressure['won'].mean()
        insights['key_findings'].append(f"High pressure situations (balls_left < 30, target > 150): {high_pressure_win_rate:.2%} win rate")
    
    # Early innings advantage
    early_innings = df[df['balls_left'] > 90]
    if len(early_innings) > 0:
        early_win_rate = early_innings['won'].mean()
        insights['key_findings'].append(f"Early innings (balls_left > 90): {early_win_rate:.2%} win rate")
    
    # Wicket impact
    low_wickets = df[df['wickets'] <= 2]
    high_wickets = df[df['wickets'] >= 6]
    
    if len(low_wickets) > 0 and len(high_wickets) > 0:
        low_wicket_win_rate = low_wickets['won'].mean()
        high_wicket_win_rate = high_wickets['won'].mean()
        insights['key_findings'].append(f"Low wickets (≤2): {low_wicket_win_rate:.2%} win rate")
        insights['key_findings'].append(f"High wickets (≥6): {high_wicket_win_rate:.2%} win rate")
    
    # Correlation insights
    correlations = df[['total_runs', 'wickets', 'target', 'balls_left', 'won']].corr()['won'].drop('won')
    strongest_correlation = correlations.abs().idxmax()
    insights['key_findings'].append(f"Strongest correlation with outcome: {strongest_correlation} ({correlations[strongest_correlation]:.3f})")
    
    return insights

def run_eda(data_path: str) -> dict:
    """
    Run complete EDA pipeline.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Dictionary of insights and statistics
    """
    log.info("Starting Exploratory Data Analysis...")
    
    # Load and validate data
    df = load_and_validate_data(data_path)
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate insights
    insights = generate_eda_insights(df)
    
    log.info("EDA completed successfully")
    return insights

if __name__ == "__main__":
    # Try different possible paths for the dataset
    possible_paths = [
        DATA_DIR / "cricket_dataset.csv",
        Path("cricket_ai/datasets/cricket_dataset.csv"),
        Path("datasets/cricket_dataset.csv")
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = str(path)
            break
    
    if data_path is None:
        print("❌ Dataset not found. Please ensure cricket_dataset.csv is in one of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        sys.exit(1)
    
    insights = run_eda(data_path)
    
    print("\n=== EDA INSIGHTS ===")
    for finding in insights['key_findings']:
        print(f"• {finding}")
    
    print(f"\nDataset size: {insights['dataset_size']} rows")
    print(f"Overall win rate: {insights['win_rate']:.2%}")
