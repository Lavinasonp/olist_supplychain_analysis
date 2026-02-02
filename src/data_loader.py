import pandas as pd
import os
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_dataset(filename):
    """Helper to load a CSV from the raw data directory."""
    path = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {filename} not found in {RAW_DATA_DIR}")
    print(f"Loading {filename}...")
    return pd.read_csv(path)

def load_and_merge_data():
    """
    Loads raw Olist datasets and merges them into a single master dataframe.
    """
    # 1. Load Raw Data
    orders = load_dataset("olist_orders_dataset.csv")
    items = load_dataset("olist_order_items_dataset.csv")
    products = load_dataset("olist_products_dataset.csv")
    payments = load_dataset("olist_order_payments_dataset.csv")
    customers = load_dataset("olist_customers_dataset.csv")
    sellers = load_dataset("olist_sellers_dataset.csv")
    category_translation = load_dataset("product_category_name_translation.csv")
    
    # Optional: Geolocation (Heavy file, often not needed for pure demand forecasting, skipping for speed unless requested)
    # geo = load_dataset("olist_geolocation_dataset.csv") 

    print("Data loaded. Starting merge process...")

    # 2. Merge Strategy
    # Start with Items (Granular level: Product per Order)
    # We use INNER JOIN on orders because we only care about items that actually have an order record
    df = pd.merge(items, orders, on="order_id", how="inner")
    
    # Add Product Details
    df = pd.merge(df, products, on="product_id", how="left")
    
    # Add Category Translations (Critical for readable features)
    df = pd.merge(df, category_translation, on="product_category_name", how="left")
    
    # Add Customer Location (for regional demand analysis)
    df = pd.merge(df, customers, on="customer_id", how="left")
    
    # Add Seller Info
    df = pd.merge(df, sellers, on="seller_id", how="left")
    
    # 3. Handle Dates
    # Convert timestamp columns to datetime objects immediately
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date', 'shipping_limit_date']
    
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 4. Clean Category Name
    # Use English name if available, else Portuguese, else 'unknown'
    df['product_category'] = df['product_category_name_english'].fillna(df['product_category_name']).fillna('unknown')
    
    # Drop unnecessary columns to save memory
    drop_cols = ['product_category_name', 'product_category_name_english', 
                 'product_name_lenght', 'product_description_lenght', 'product_photos_qty']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    print(f"Merge Complete. Final Shape: {df.shape}")
    return df

def save_processed(df, filename="master_table.parquet"):
    """Saves the dataframe to the processed directory as Parquet."""
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    print(f"Saving processed data to {path}...")
    df.to_parquet(path, index=False)
    print("Save complete.")

if __name__ == "__main__":
    # execute the pipeline
    df_master = load_and_merge_data()
    save_processed(df_master)