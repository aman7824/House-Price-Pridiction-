"""
Training script for Flate Price Prediction model.

This script:
1. Loads and preprocesses the Bengaluru housing dataset
2. Trains RandomForest and XGBoost models
3. Compares their performance using cross-validation
4. Saves the best model as a pipeline
Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import preprocess_data, get_feature_columns
from src.model import (
    train_random_forest, 
    train_xgboost, 
    evaluate_model,
    cross_validate_model,
    get_feature_importance
)


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("Indian Flate Price Prediction - Model Training")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    data_path = "data/bengaluru_housing.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure the dataset file exists.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    print(f"Features: {list(df.columns)}")
    
    # Preprocess data
    print("\n[2/6] Preprocessing data...")
    df_processed, encoders = preprocess_data(df, is_training=True)
    print(f"After preprocessing: {len(df_processed)} records")
    print(f"Removed {len(df) - len(df_processed)} outlier records")
    
    # Prepare features and target
    print("\n[3/6] Preparing features and target...")
    feature_cols = get_feature_columns()
    X = df_processed[feature_cols]
    y = df_processed['price']
    
    print(f"Feature columns: {feature_cols}")
    print(f"Target: price (mean: ₹{y.mean():,.2f}, std: ₹{y.std():,.2f})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    print("\n[4/6] Training models...")
    
    # Train Random Forest
    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics_test = evaluate_model(rf_model, X_test, y_test, "Random Forest (Test Set)")
    rf_cv_metrics = cross_validate_model(rf_model, X_train, y_train)
    rf_metrics_test.update(rf_cv_metrics)
    
    # Train XGBoost
    print("\n--- XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics_test = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Test Set)")
    xgb_cv_metrics = cross_validate_model(xgb_model, X_train, y_train)
    xgb_metrics_test.update(xgb_cv_metrics)
    
    # Compare models
    print("\n[5/6] Comparing models...")
    print("\nModel Comparison (Test Set RMSE):")
    print(f"  Random Forest: ₹ {rf_metrics_test['rmse']:,.2f}")
    print(f"  XGBoost:       ₹ {xgb_metrics_test['rmse']:,.2f}")
    
    # Select best model
    if rf_metrics_test['rmse'] < xgb_metrics_test['rmse']:
        best_model = rf_model
        best_model_name = "RandomForest"
        best_metrics = rf_metrics_test
        print("\n✓ Best Model: Random Forest")
    else:
        best_model = xgb_model
        best_model_name = "XGBoost"
        best_metrics = xgb_metrics_test
        print("\n✓ Best Model: XGBoost")
    
    # Feature importance
    print("\n[6/6] Analyzing feature importance...")
    feature_importance = get_feature_importance(best_model, feature_cols)
    print("\nTop 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save pipeline
    print("\nSaving model pipeline...")
    os.makedirs("models", exist_ok=True)
    
    pipeline = {
        'model': best_model,
        'encoders': encoders,
        'feature_columns': feature_cols,
        'model_name': best_model_name,
        'metrics': best_metrics,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    joblib.dump(pipeline, "models/pipeline.joblib")
    print("✓ Model saved to models/pipeline.joblib")
    
    # Summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model: {best_model_name}")
    print(f"RMSE: ₹ {best_metrics['rmse']:,.2f}")
    print(f"MAE:  ₹ {best_metrics['mae']:,.2f}")
    print(f"R²:   {best_metrics['r2']:.4f}")
    print("\nThe model is ready for predictions!")
    print("=" * 70)


if __name__ == "__main__":
    main()
