import pandas as pd
import numpy as np
import os
from src.config import PROCESSED_DATA_DIR

def load_processed_data(filename="master_table.parquet"):
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    print(f"Loading {path}...")
    return pd.read_parquet(path)

def create_time_series_features(df):
    print("Starting Feature Engineering (Leakage-Free Version)...")
    
    # 1. Setup Dates
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # 2. Identify "Life Cycle" of each Category (Fixing the Zombie Product issue)
    # We only want to track a category starting from its first ever sale.
    category_start_dates = df.groupby('product_category')['order_purchase_timestamp'].min().reset_index()
    category_start_dates.rename(columns={'order_purchase_timestamp': 'first_sale_date'}, inplace=True)

    # 3. Weekly Aggregation
    # Note: We aggregate "everything" here, but we will SHIFT the predictive columns later.
    weekly_data = df.groupby([
        'product_category', 
        pd.Grouper(key='order_purchase_timestamp', freq='W-MON')
    ]).agg(
        quantity_sold=('order_id', 'count'),       # TARGET
        revenue=('price', 'sum'),
        curr_avg_price=('price', 'mean'),          # POTENTIAL LEAK (Must be lagged)
        curr_avg_freight=('freight_value', 'mean'),# POTENTIAL LEAK (Must be lagged)
        curr_active_sellers=('seller_id', 'nunique')    # POTENTIAL LEAK (Must be lagged)
    ).reset_index()

    # 4. Handle Sparsity (The Grid)
    # We need a continuous timeline, but only within valid dates
    all_categories = weekly_data['product_category'].unique()
    all_weeks = weekly_data['order_purchase_timestamp'].unique()
    
    # Create full grid
    full_index = pd.MultiIndex.from_product(
        [all_categories, all_weeks], 
        names=['product_category', 'order_purchase_timestamp']
    )
    weekly_data = weekly_data.set_index(['product_category', 'order_purchase_timestamp'])
    weekly_data = weekly_data.reindex(full_index, fill_value=0).reset_index()
    
    # Merge Start Dates back in
    weekly_data = pd.merge(weekly_data, category_start_dates, on='product_category', how='left')
    
    # 5. FILTER: Remove "Zombie Rows" (Dates before the category existed)
    # We keep rows where date >= first_sale_date
    original_len = len(weekly_data)
    weekly_data = weekly_data[weekly_data['order_purchase_timestamp'] >= weekly_data['first_sale_date']].copy()
    print(f"Dropped {original_len - len(weekly_data)} rows of 'Zombie' data (pre-launch periods).")

    weekly_data = weekly_data.sort_values(['product_category', 'order_purchase_timestamp'])

    # ==============================================================================
    # 🛠️ CRITICAL FIX: LAGGING EXPLANATORY VARIABLES
    # We cannot use 'curr_avg_price' to predict 'quantity_sold' of the same week.
    # We must use LAST WEEK'S price/freight/sellers.
    # ==============================================================================
    
    # Create Base Lags (1 Week Shift)
    weekly_data['price_lag_1'] = weekly_data.groupby('product_category')['curr_avg_price'].shift(1)
    weekly_data['freight_lag_1'] = weekly_data.groupby('product_category')['curr_avg_freight'].shift(1)
    weekly_data['sellers_lag_1'] = weekly_data.groupby('product_category')['curr_active_sellers'].shift(1)
    
    # Forward fill missing prices/freight (if no sales last week, assume price didn't change from week before)
    # However, for pure zero-inflated logic, we can fill with 0 or mean. 
    # Better approach: If price_lag_1 is NaN (start of series), fill with current to avoid dropping data, 
    # but strictly speaking, ffill is better for time series.
    weekly_data['price_lag_1'] = weekly_data.groupby('product_category')['price_lag_1'].ffill().fillna(0)
    weekly_data['freight_lag_1'] = weekly_data.groupby('product_category')['freight_lag_1'].ffill().fillna(0)
    weekly_data['sellers_lag_1'] = weekly_data['sellers_lag_1'].fillna(0)

    # 6. Feature Engineering (Safe from Leakage)
    
    # A. Date Features
    weekly_data['week_of_year'] = weekly_data['order_purchase_timestamp'].dt.isocalendar().week.astype(int)
    weekly_data['month'] = weekly_data['order_purchase_timestamp'].dt.month
    weekly_data['year'] = weekly_data['order_purchase_timestamp'].dt.year
    
    # Cyclical Encoding
    weekly_data['month_sin'] = np.sin(2 * np.pi * weekly_data['month']/12)
    weekly_data['month_cos'] = np.cos(2 * np.pi * weekly_data['month']/12)

    # B. Economic Features (Using LAGGED data)
    # "Pain Index": Freight cost relative to product price (as known from last week)
    weekly_data['freight_ratio'] = weekly_data['freight_lag_1'] / weekly_data['price_lag_1']
    weekly_data['freight_ratio'] = weekly_data['freight_ratio'].fillna(0).replace([np.inf, -np.inf], 0)

    # C. Price Momentum
    # Compare Last Week's Price to the 4-week average BEFORE that.
    weekly_data['price_roll_mean_4'] = weekly_data.groupby('product_category')['price_lag_1'].transform(
        lambda x: x.rolling(window=4).mean()
    )
    weekly_data['price_momentum'] = weekly_data['price_lag_1'] / weekly_data['price_roll_mean_4']
    weekly_data['price_momentum'] = weekly_data['price_momentum'].fillna(1.0)

    # D. Target Lags (Autoregression)
    # How much did we sell in previous weeks?
    lags = [1, 2, 3, 4, 8] # Removed 12 to save data rows if dataset is short
    for lag in lags:
        weekly_data[f'sales_lag_{lag}'] = weekly_data.groupby('product_category')['quantity_sold'].shift(lag)

    # E. Rolling Sales Stats (Volatility)
    # Shift(1) ensures we don't include current week in the mean
    weekly_data['sales_roll_mean_4'] = weekly_data.groupby('product_category')['quantity_sold'].transform(
        lambda x: x.shift(1).rolling(window=4).mean()
    )
    # Standard Deviation (Crucial for Safety Stock calculation later)
    weekly_data['sales_roll_std_4'] = weekly_data.groupby('product_category')['quantity_sold'].transform(
        lambda x: x.shift(1).rolling(window=4).std()
    )

    # F. Events: Weeks to Black Friday
    def get_weeks_to_bf(date):
        bf_date = pd.Timestamp(year=date.year, month=11, day=24)
        if date > bf_date:
            bf_date = pd.Timestamp(year=date.year + 1, month=11, day=24)
        delta = bf_date - date
        return int(delta.days / 7)

    weekly_data['weeks_to_bf'] = weekly_data['order_purchase_timestamp'].apply(get_weeks_to_bf)
    weekly_data['weeks_to_bf'] = weekly_data['weeks_to_bf'].clip(upper=20) # Cap at 20

    # 7. Cleanup
    # We must drop rows with NaNs generated by lags (e.g., the first 8 weeks of data)
    # Otherwise, the model crashes or learns garbage.
    weekly_data.dropna(inplace=True)
    
    # Drop the "Current" columns to prevent accidental usage in training
    # We only keep the TARGET (quantity_sold) and the FEATURES (lags, dates, etc)
    drop_cols = ['revenue', 'curr_avg_price', 'curr_avg_freight', 'curr_active_sellers', 'first_sale_date']
    weekly_data.drop(columns=drop_cols, inplace=True)

    print(f"Feature Engineering Complete. Final Shape: {weekly_data.shape}")
    return weekly_data

def save_features(df, filename="features_v3.parquet"):
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    print(f"Saving features to {path}...")
    df.to_parquet(path, index=False)
    print("Save complete.")

if __name__ == "__main__":
    df_raw = load_processed_data()
    df_feats = create_time_series_features(df_raw)
    save_features(df_feats)