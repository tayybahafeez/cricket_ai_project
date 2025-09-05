# run_test_predictions.py
import pandas as pd
from cricket_ml.pipelines.inference import predict_from_df
from cricket_ml.utils.config import DATA_DIR

def main():
    test_file = DATA_DIR / "cricket_dataset_test.csv"
    
    try:
        test_df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Test dataset not found at {test_file}")
        return

    preds, meta = predict_from_df(test_df)
    print("✅ Predictions completed.")
    print(f"Predictions saved at: {meta['predictions_file']}")
    print(f"Total rows: {len(test_df)} | Filtered rows: {len(preds)} | Predictions made: {int(preds['prediction'].sum())}")

if __name__ == "__main__":
    main()
