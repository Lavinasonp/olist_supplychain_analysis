import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import PROCESSED_DATA_DIR, MODEL_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
import os
import joblib

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def load_features(filename="features_v3.parquet"):
    """Loads the Clean, Leakage-Free features."""
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Did you run src/preprocessing.py?")
    return pd.read_parquet(path)

def train_model():
    print("Loading data for training...")
    df = load_features()
    
    TARGET = 'quantity_sold'
    
    # --- FEATURE SELECTION ---
    # We exclude identifiers and columns that don't generalize (like specific years).
    ignore_cols = [
        'order_purchase_timestamp', 
        'year',  # CRITICAL FIX: Removed Year to force model to learn trends, not specific dates
        TARGET
    ]
    
    # Select all other columns as features
    features = [c for c in df.columns if c not in ignore_cols]
    
    print(f"Training with {len(features)} features:")
    print(features)
    
    # LightGBM requires 'category' type for categorical features
    if 'product_category' in df.columns:
        df['product_category'] = df['product_category'].astype('category')
    
    # --- TIME BASED SPLIT ---
    # Train: Jan 2017 -> May 2018
    # Test: June 2018 -> End of Data
    print("Splitting data by time...")
    
    # We use the 'year' and 'month' columns for splitting, even though we don't train on 'year'
    train_mask = (df['year'] < 2018) | ((df['year'] == 2018) & (df['month'] < 6))
    test_mask = (df['year'] == 2018) & (df['month'] >= 6)
    
    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, TARGET]
    
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, TARGET]
    
    print(f"Train Rows: {len(X_train)} | Test Rows: {len(X_test)}")
    
    # --- MLFLOW RUN ---
    with mlflow.start_run(run_name="LGBM_Tweedie_LeakageFree_v3"):
        
        # --- MODEL PARAMETERS ---
        params = {
            'objective': 'tweedie',         # Optimized for zero-inflated sales (0, 0, 0, 5, 0...)
            'tweedie_variance_power': 1.1,  # Closer to Poisson (1.0) than Gamma (2.0)
            'metric': 'mae',                # We optimize for MAE, but evaluate on WMAPE
            'boosting_type': 'gbdt',
            'n_estimators': 3000,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 8,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        mlflow.log_params(params)
        
        # Train
        print("Fitting LightGBM (Tweedie Objective)...")
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=['product_category'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=150),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Evaluate
        print("Evaluating model...")
        preds = model.predict(X_test)
        
        # 1. Standard Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # 2. Supply Chain Metric: WMAPE (Weighted Mean Absolute Percentage Error)
        # Formula: Sum(|Actual - Forecast|) / Sum(Actual)
        # This is robust against zero-demand weeks (unlike standard MAPE).
        total_actual_sales = np.sum(y_test)
        total_absolute_error = np.sum(np.abs(y_test - preds))
        wmape = total_absolute_error / total_actual_sales
        
        print(f"\n--- FINAL METRICS ---")
        print(f"MAE:   {mae:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"WMAPE: {wmape:.2%} (Industry Standard)")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("wmape", wmape)
        
        # Save Feature Importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        importance_path = os.path.join(MODEL_DIR, "feature_importance_v3.csv")
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        print("\nTop 5 Features:")
        print(importance.head(5))
        
        # Save Model
        model_path = os.path.join(MODEL_DIR, "lgbm_demand_v3.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Model saved to {model_path}")
        print("Training complete.")

if __name__ == "__main__":
    train_model()