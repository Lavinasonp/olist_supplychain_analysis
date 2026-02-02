import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_and_merge_data, save_processed
from src.preprocessing import load_processed_data, create_time_series_features, save_features
from src.training import train_model
from src.evaluation import generate_report
from src.inventory import generate_inventory_plan

def main():
    print("===================================================")
    print("   STARTING OLIST SUPPLY CHAIN PIPELINE (E2E)    ")
    print("===================================================")

    # 1. ETL (Extract, Transform, Load)
    print("\n[STEP 1] LOADING & MERGING RAW DATA...")
    try:
        df_master = load_and_merge_data()
        save_processed(df_master)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your CSV files are in 'data/raw/'")
        return

    # 2. Feature Engineering
    print("\n[STEP 2] FEATURE ENGINEERING (V3)...")
    df_raw = load_processed_data("master_table.parquet")
    df_feats = create_time_series_features(df_raw)
    save_features(df_feats, "features_v3.parquet")

    # 3. Model Training
    print("\n[STEP 3] MODEL TRAINING...")
    train_model()

    # 4. Evaluation & Reporting
    print("\n[STEP 4] GENERATING REPORTS...")
    generate_report()

    # 5. Inventory Planning
    print("\n[STEP 5] GENERATING INVENTORY PLAN...")
    generate_inventory_plan(service_level=0.95)

    print("\n===================================================")
    print("   PIPELINE COMPLETE SUCCESSFULY")
    print("   Check 'reports/' for graphs and inventory plans.")
    print("===================================================")

if __name__ == "__main__":
    main()