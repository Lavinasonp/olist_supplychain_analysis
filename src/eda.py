import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, RAW_DATA_DIR

# --- 1. SETUP ---
FIGURES_DIR = os.path.join(REPORTS_DIR, "eda_visuals")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", palette="viridis")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def load_master_data():
    path = os.path.join(PROCESSED_DATA_DIR, "master_table.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run src/data_loader.py first.")
    df = pd.read_parquet(path)
    # Ensure datetimes
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# --- 2. EXISTING PLOTS (Streamlined) ---
def plot_basic_trends(df):
    print("Generating Basic Trends...")
    # Monthly Revenue
    monthly_sales = df.set_index('order_purchase_timestamp').resample('M')['price'].sum()
    plt.figure()
    monthly_sales.plot(kind='line', linewidth=3, marker='o', color='#2ecc71')
    plt.title('Total Monthly Revenue', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_monthly_revenue.png"))
    plt.close()

# --- 3. NEW: ADVANCED DELAY ANALYSIS ---
def plot_delay_analysis(df):
    print("Generating Delay Analysis...")
    
    # Calculate "Lateness" (Actual - Estimated)
    # Positive = Late, Negative = Early
    df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    
    # Flag: Is it actually late?
    df['is_late'] = df['delay_days'] > 0
    
    # A. Average Delay by State (Boxplot to show variance)
    # Filter out extreme outliers for visualization (> 50 days late is rare)
    plot_data = df[(df['delay_days'] > -20) & (df['delay_days'] < 20)]
    
    # Sort states by median delay
    state_order = plot_data.groupby('customer_state')['delay_days'].median().sort_values(ascending=False).index
    
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=plot_data, x='customer_state', y='delay_days', order=state_order, palette='RdBu_r')
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='On Time')
    plt.title('Delivery Reliability by State (Actual vs Estimated)', fontweight='bold')
    plt.ylabel('Days Late (Positive) or Early (Negative)')
    plt.xlabel('State')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "07_state_delay_distribution.png"))
    plt.close()

    # B. Percentage of Late Orders by State
    late_metrics = df.groupby('customer_state')['is_late'].mean().sort_values(ascending=False).reset_index()
    late_metrics['is_late'] = late_metrics['is_late'] * 100 # Convert to %
    
    plt.figure(figsize=(16, 8))
    sns.barplot(data=late_metrics, x='customer_state', y='is_late', palette='magma')
    plt.title('Percentage of Orders Delivered Late per State', fontweight='bold')
    plt.ylabel('% Late Orders')
    plt.axhline(late_metrics['is_late'].mean(), color='red', linestyle='--', label='National Average')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "08_percent_late_by_state.png"))
    plt.close()

# --- 4. NEW: CITY LEVEL ANALYSIS ---
def plot_city_analysis(df):
    print("Generating City Analysis...")
    
    # Filter: Top 20 Cities by Order Volume
    top_cities = df['customer_city'].value_counts().head(20).index
    city_df = df[df['customer_city'].isin(top_cities)].copy()
    
    # Calculate Delivery Time
    city_df['delivery_time'] = (city_df['order_delivered_customer_date'] - city_df['order_purchase_timestamp']).dt.days
    
    # Sort by Avg Delivery Time
    order = city_df.groupby('customer_city')['delivery_time'].mean().sort_values(ascending=False).index
    
    plt.figure(figsize=(16, 10))
    sns.barplot(data=city_df, y='customer_city', x='delivery_time', order=order, palette='cool_r')
    plt.title('Average Delivery Time (Days) - Top 20 Cities', fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "09_city_delivery_speed.png"))
    plt.close()

# --- 5. NEW: HEATMAP OF BUYING HABITS ---
def plot_buying_heatmap(df):
    print("Generating Buying Pattern Heatmap...")
    
    # Extract Hour and Day
    df['hour'] = df['order_purchase_timestamp'].dt.hour
    df['day_of_week'] = df['order_purchase_timestamp'].dt.day_name()
    
    # Order of days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Pivot Table for Heatmap
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack()
    heatmap_data = heatmap_data.reindex(days_order)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False, fmt='d')
    plt.title('Heatmap of Orders: Day vs Hour', fontweight='bold')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_buying_heatmap.png"))
    plt.close()

# --- 6. NEW: PAYMENT ANALYSIS ---
def plot_payment_stats():
    # Need to load payment dataset specifically
    try:
        path = os.path.join(RAW_DATA_DIR, "olist_order_payments_dataset.csv")
        if os.path.exists(path):
            payments = pd.read_csv(path)
            
            # Pie Chart of Payment Types
            payment_counts = payments['payment_type'].value_counts()
            
            plt.figure(figsize=(10, 10))
            plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
            plt.title('Distribution of Payment Methods', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, "11_payment_methods.png"))
            plt.close()
            print("Payment plot generated.")
    except Exception as e:
        print(f"Skipping payment analysis: {e}")

def plot_logistics_breakdown(df):
    print("Generating Logistics Breakdown...")
    
    # 1. Feature Engineering (Recreating your Kaggle logic)
    # Time to Ship: From Approval -> Handover to Carrier (Seller Responsibility)
    df['time_to_ship'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days
    
    # Carrier Time: From Handover -> Delivery to Customer (Carrier Responsibility)
    df['carrier_time'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.days
    
    # Total Delivery Time: Purchase -> Customer
    df['total_delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    
    # Filter valid positive times (negative times are data errors)
    valid_logistics = df[
        (df['time_to_ship'] >= 0) & 
        (df['carrier_time'] >= 0) & 
        (df['total_delivery_time'] < 100) # Remove extreme outliers
    ].copy()
    
    # A. Comparative Histogram (Seller vs Carrier)
    plt.figure(figsize=(14, 7))
    sns.histplot(valid_logistics['time_to_ship'], color='blue', alpha=0.5, label='Time to Ship (Seller)', bins=30, kde=True)
    sns.histplot(valid_logistics['carrier_time'], color='orange', alpha=0.5, label='Transit Time (Carrier)', bins=30, kde=True)
    plt.title('Seller Speed vs. Carrier Speed', fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "12_seller_vs_carrier_time.png"))
    plt.close()

    # B. The "Who is slower?" Ratio
    avg_ship = valid_logistics['time_to_ship'].mean()
    avg_transit = valid_logistics['carrier_time'].mean()
    
    # Pie chart of Total Time Composition
    labels = ['Seller Processing', 'Carrier Transit']
    sizes = [avg_ship, avg_transit]
    colors = ['#3498db', '#e67e22']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
    plt.title(f'Average Delivery Breakdown\n(Total Avg: {avg_ship+avg_transit:.1f} days)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "13_delivery_time_breakdown.png"))
    plt.close()

# ... (Keep imports and previous functions)

def plot_logistics_breakdown(df):
    print("Generating Logistics Breakdown...")
    
    # 1. Feature Engineering
    # Seller Speed: Approved -> Handover
    df['time_to_ship'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days
    # Carrier Speed: Handover -> Customer
    df['carrier_time'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.days
    
    # Filter valid data
    valid_logistics = df[
        (df['time_to_ship'] >= 0) & 
        (df['carrier_time'] >= 0) & 
        (df['time_to_ship'] < 30) & # Filter extremes for clean plots
        (df['carrier_time'] < 60)
    ].copy()
    
    # --- PLOT A: Global Histogram (Seller vs Carrier) ---
    plt.figure(figsize=(14, 7))
    sns.histplot(valid_logistics['time_to_ship'], color='#3498db', alpha=0.6, label='Seller Processing Time', bins=20, kde=True)
    sns.histplot(valid_logistics['carrier_time'], color='#e67e22', alpha=0.6, label='Carrier Transit Time', bins=40, kde=True)
    plt.title('Global Speed Comparison: Seller vs. Carrier', fontweight='bold')
    plt.xlabel('Days')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "12_seller_vs_carrier_hist.png"))
    plt.close()

    # --- PLOT B: State-Wise Breakdown (Stacked Bar) ---
    print("Generating State-wise Logistics Stacked Bar...")
    
    # Group by State and get average times
    state_logs = valid_logistics.groupby('customer_state')[['time_to_ship', 'carrier_time']].mean().sort_values('carrier_time', ascending=False)
    
    # Plot
    # We use a stacked bar chart: Bottom = Seller Time, Top = Carrier Time
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create the bottom bar (Seller Time)
    ax.bar(state_logs.index, state_logs['time_to_ship'], label='Seller Processing', color='#3498db', alpha=0.8)
    
    # Create the top bar (Carrier Time), stacked on the bottom
    ax.bar(state_logs.index, state_logs['carrier_time'], bottom=state_logs['time_to_ship'], label='Carrier Transit', color='#e67e22', alpha=0.8)
    
    # Formatting
    plt.title('Average Delivery Composition by State (Who is the bottleneck?)', fontweight='bold')
    plt.ylabel('Average Days')
    plt.xlabel('Customer State')
    plt.legend()
    
    # Add value labels for total time
    for i, state in enumerate(state_logs.index):
        total = state_logs.loc[state, 'time_to_ship'] + state_logs.loc[state, 'carrier_time']
        ax.text(i, total + 0.5, f"{total:.1f}", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "13_state_logistics_breakdown.png"))
    plt.close()

if __name__ == "__main__":
    df_master = load_master_data()
    
    plot_basic_trends(df_master)
    plot_delay_analysis(df_master)
    plot_city_analysis(df_master)
    plot_buying_heatmap(df_master)
    plot_payment_stats()
    
    # Run the new detailed breakdown
    plot_logistics_breakdown(df_master)
    
    print(f"\nAll plots (including state logistics) saved to: {FIGURES_DIR}")