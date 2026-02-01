"""
Model training, evaluation, and prediction functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


def train_random_forest(X: pd.DataFrame, y: pd.Series, 
                       n_estimators: int = 100, 
                       random_state: int = 42) -> RandomForestRegressor:
    """
    Train Random Forest Regressor.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        Trained RandomForestRegressor
    """
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    return rf_model


def train_xgboost(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> xgb.XGBRegressor:
    """
    Train XGBoost Regressor.
    
    Args:
        X: Feature matrix
        y: Target variable
        random_state: Random seed
        
    Returns:
        Trained XGBRegressor
    """
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1
    )
    xgb_model.fit(X, y)
    return xgb_model


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                   model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True target values
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: ₹ {rmse:,.2f}")
    print(f"  MAE:  ₹ {mae:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                        cv: int = 5) -> Dict[str, float]:
    """
    Perform cross-validation on model.
    
    Args:
        model: Model to evaluate
        X: Feature matrix
        y: Target variable
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation scores
    """
    # Negative MSE for scoring
    neg_mse_scores = cross_val_score(model, X, y, cv=cv, 
                                     scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-neg_mse_scores)
    
    mae_scores = cross_val_score(model, X, y, cv=cv, 
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    mae_scores = -mae_scores
    
    print(f"Cross-Validation Scores ({cv}-fold):")
    print(f"  RMSE: ₹ {rmse_scores.mean():,.2f} (+/- ₹ {rmse_scores.std():,.2f})")
    print(f"  MAE:  ₹ {mae_scores.mean():,.2f} (+/- ₹ {mae_scores.std():,.2f})")
    
    return {
        'cv_rmse_mean': rmse_scores.mean(),
        'cv_rmse_std': rmse_scores.std(),
        'cv_mae_mean': mae_scores.mean(),
        'cv_mae_std': mae_scores.std()
    }


def calculate_prediction_interval(model: Any, X: pd.DataFrame, 
                                 confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction interval using model's estimators (for ensemble models).
    
    Args:
        model: Trained ensemble model (RandomForest or XGBoost)
        X: Feature matrix
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    if isinstance(model, RandomForestRegressor):
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in model.estimators_])
        
        # Calculate percentiles
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        lower_bounds = np.percentile(predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(predictions, upper_percentile, axis=0)
        
    elif isinstance(model, xgb.XGBRegressor):
        # For XGBoost, use a simple approach based on training residuals
        # This is an approximation
        predictions = model.predict(X)
        
        # Use a factor based on confidence level (approximation)
        # For 95% CI, use about 1.96 standard deviations
        factor = 1.96 if confidence == 0.95 else 2.58
        
        # Estimate standard deviation from the model (rough approximation)
        std_estimate = predictions * 0.15  # Assume 15% uncertainty
        
        lower_bounds = predictions - factor * std_estimate
        upper_bounds = predictions + factor * std_estimate
    else:
        # Fallback: use simple percentage-based interval
        predictions = model.predict(X)
        margin = predictions * 0.1  # 10% margin
        lower_bounds = predictions - margin
        upper_bounds = predictions + margin
    
    return lower_bounds, upper_bounds


def predict_price(input_dict: Dict[str, Any], 
                 model_path: str = "models/pipeline.joblib") -> Dict[str, Any]:
    """
    Predict Flate price from input dictionary.
    
    Args:
        input_dict: Dictionary with property features
        model_path: Path to saved model pipeline
        
    Returns:
        Dictionary with prediction, interval, and metadata
    """
    from src.utils import prepare_input_for_prediction, format_indian_currency
    
    # Load the saved pipeline
    pipeline = joblib.load(model_path)
    
    model = pipeline['model']
    encoders = pipeline['encoders']
    metrics = pipeline['metrics']
    model_name = pipeline['model_name']
    
    # Prepare input
    X = prepare_input_for_prediction(input_dict, encoders)
    
    # Make prediction
    predicted_price = model.predict(X)[0]
    
    # Calculate prediction interval
    lower_bounds, upper_bounds = calculate_prediction_interval(model, X)
    
    result = {
        'predicted_price_inr': float(predicted_price),
        'formatted': format_indian_currency(predicted_price),
        'lower': float(max(0, lower_bounds[0])),  # Ensure non-negative
        'upper': float(upper_bounds[0]),
        'lower_formatted': format_indian_currency(max(0, lower_bounds[0])),
        'upper_formatted': format_indian_currency(upper_bounds[0]),
        'model': model_name,
        'rmse': float(metrics.get('rmse', 0)),
        'mae': float(metrics.get('mae', 0)),
        'r2': float(metrics.get('r2', 0))
    }
    
    return result


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance sorted by importance
    """
    if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
        importance = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    else:
        return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})
