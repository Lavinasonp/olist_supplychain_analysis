import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score
from src.config import PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR

def load_data_and_model():
    # 1. Load V3 Data and Model
    data_path = os.path.join(PROCESSED_DATA_DIR, "features_v3.parquet")
    df = pd.read_parquet(data_path)
    
    model_path = os.path.join(MODEL_DIR, "lgbm_demand_v3.pkl")
    model = joblib.load(model_path)
    
    return df, model

def generate_report():
    print("Generating Performance Report...")
    df, model = load_data_and_model()
    
    # --- RECREATE TEST SPLIT ---
    # We filter for the Test period (June 2018+)
    # Note: We must ensure column types match training
    df['product_category'] = df['product_category'].astype('category')
    
    # Define features exactly as done in training
    TARGET = 'quantity_sold'
    ignore_cols = ['order_purchase_timestamp', 'year', TARGET]
    features = [c for c in df.columns if c not in ignore_cols]
    
    # Split
    test_mask = (df['year'] == 2018) & (df['month'] >= 6)
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, TARGET]
    
    # Metadata for plotting
    test_dates = df.loc[test_mask, 'order_purchase_timestamp']
    test_cats = df.loc[test_mask, 'product_category']
    
    print(f"Predicting on {len(X_test)} test samples...")
    preds = model.predict(X_test)
    
    # Create Results DataFrame
    results = pd.DataFrame({
        'date': test_dates,
        'category': test_cats,
        'actual': y_test,
        'predicted': preds
    })
    
    # --- PLOT 1: GLOBAL DEMAND (Aggregate) ---
    # Sum of all sales across all categories per week
    global_demand = results.groupby('date')[['actual', 'predicted']].sum().reset_index()
    
    plt.figure(figsize=(14, 6))
    plt.plot(global_demand['date'], global_demand['actual'], label='Actual Sales', color='black', alpha=0.7)
    plt.plot(global_demand['date'], global_demand['predicted'], label='Model Forecast', color='#2ca02c', linestyle='--', linewidth=2)
    plt.title("Global Demand Forecast: Actual vs Predicted (Test Period)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Total Units Sold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "01_global_forecast_vs_actual.png"))
    print("Saved Plot 1: Global Forecast")

    # --- PLOT 2: TOP 3 CATEGORIES ZOOM-IN ---
    top_cats = results.groupby('category')['actual'].sum().nlargest(3).index.tolist()
    
    plt.figure(figsize=(14, 10))
    for i, cat in enumerate(top_cats):
        cat_data = results[results['category'] == cat]
        # Group by date to handle cases where there might be duplicate rows (though in V3 there shouldn't be)
        cat_data = cat_data.groupby('date')[['actual', 'predicted']].sum().reset_index()
        
        plt.subplot(3, 1, i+1)
        plt.plot(cat_data['date'], cat_data['actual'], label='Actual', marker='.', color='gray')
        plt.plot(cat_data['date'], cat_data['predicted'], label='Forecast', color='blue', linestyle='--')
        plt.title(f"Forecast: {cat}")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "02_category_forecasts.png"))
    print(f"Saved Plot 2: Top Category Forecasts ({top_cats})")

    # --- PLOT 3: ERROR DISTRIBUTION ---
    results['error'] = results['actual'] - results['predicted']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(results['error'], bins=50, kde=True, color='crimson')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title("Forecast Error Distribution (Residuals)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.xlim(-10, 10) # Zoom in to see the core performance
    plt.savefig(os.path.join(REPORTS_DIR, "03_error_distribution.png"))
    print("Saved Plot 3: Error Distribution")

if __name__ == "__main__":
    generate_report()