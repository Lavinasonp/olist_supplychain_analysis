import pandas as pd
import numpy as np
import joblib
import os
from src.config import PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR

def load_resources():
    # Load V3 Features and V3 Model
    print("Loading resources...")
    df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "features_v3.parquet"))
    model = joblib.load(os.path.join(MODEL_DIR, "lgbm_demand_v3.pkl"))
    return df, model

def generate_inventory_plan(service_level=0.95):
    print("Generating Inventory Plan...")
    df, model = load_resources()
    
    # 1. Get the Latest Week's Data (The "Now")
    # In a real pipeline, this would be a separate "scoring" dataset for Next Week.
    # Here, we simulate it by taking the very last known week of data.
    latest_date = df['order_purchase_timestamp'].max()
    print(f"Planning inventory based on data from week: {latest_date}")
    
    current_stock_data = df[df['order_purchase_timestamp'] == latest_date].copy()
    
    # 2. Prepare Features for Prediction
    # MUST match the training feature set exactly
    ignore_cols = ['order_purchase_timestamp', 'year', 'quantity_sold']
    features = [c for c in df.columns if c not in ignore_cols]
    
    # Ensure category type matches training
    current_stock_data['product_category'] = current_stock_data['product_category'].astype('category')
    
    X_pred = current_stock_data[features]
    
    # 3. Predict Next Week's Demand
    predicted_demand = model.predict(X_pred)
    
    # 4. Calculate Inventory Policy
    
    # Setup Results Table
    results = pd.DataFrame({
        'product_category': current_stock_data['product_category'],
        'predicted_weekly_demand': np.ceil(predicted_demand), # Round up
        'current_price': current_stock_data['price_lag_1'],    # Use the lagged price (the one we know)
        'demand_std_dev': current_stock_data['sales_roll_std_4'] # ACTUAL Volatility
    })
    
    # --- SAFETY STOCK FORMULA ---
    # Formula: Z * sqrt(LeadTime) * StdDev
    
    # A. Z-Score (Service Level)
    # 95% Service Level = 1.645 Z-score (Standard Normal Distribution)
    z_score = 1.645 if service_level == 0.95 else 1.28
    
    # B. Lead Time
    # Assumption: It takes 2 weeks to get products from suppliers
    lead_time_weeks = 2
    
    # C. Calculate Safety Stock
    # If std_dev is NaN (new product), assume it matches the mean demand (high uncertainty)
    results['demand_std_dev'] = results['demand_std_dev'].fillna(results['predicted_weekly_demand'])
    
    results['safety_stock'] = np.ceil(
        z_score * np.sqrt(lead_time_weeks) * results['demand_std_dev']
    )
    
    # 5. Reorder Point (ROP)
    # When stock drops to this level, place an order.
    # ROP = (Daily_Demand * Lead_Time) + Safety_Stock
    # Since our demand is weekly, ROP = (Weekly_Demand * Lead_Time_Weeks) + Safety_Stock
    results['reorder_point'] = (results['predicted_weekly_demand'] * lead_time_weeks) + results['safety_stock']
    
    # 6. Formatting & Recommendations
    results['order_recommendation'] = results['reorder_point'].apply(lambda x: f"Maintain ~{int(x)} units")
    
    # Sort by Urgency (Highest Demand Volume)
    results = results.sort_values('predicted_weekly_demand', ascending=False)
    
    # Save Report
    save_path = os.path.join(REPORTS_DIR, "inventory_replenishment_plan.csv")
    results.to_csv(save_path, index=False)
    
    print("\n--- INVENTORY REPLENISHMENT PLAN (Top 5 Categories) ---")
    print(results[['product_category', 'predicted_weekly_demand', 'demand_std_dev', 'safety_stock', 'reorder_point']].head(5))
    print(f"\nFull plan saved to {save_path}")

if __name__ == "__main__":
    generate_inventory_plan()