"""
Utility functions for data preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def format_indian_currency(amount: float) -> str:
    """
    Format amount in Indian currency notation (lakhs and crores).
    
    Args:
        amount: Amount in rupees
        
    Returns:
        Formatted string (e.g., "₹ 45,00,000" or "₹ 1.2 Cr")
    """
    if amount >= 10000000:  # 1 crore or more
        crores = amount / 10000000
        return f"₹ {crores:.2f} Cr"
    elif amount >= 100000:  # 1 lakh or more
        lakhs = amount / 100000
        return f"₹ {lakhs:.2f} L"
    else:
        # Regular formatting with commas
        return f"₹ {amount:,.0f}"


def create_age_buckets(age: int) -> str:
    """
    Create age buckets for property age.
    
    Args:
        age: Age of property in years
        
    Returns:
        Age bucket category
    """
    if age <= 2:
        return "new"
    elif age <= 5:
        return "recent"
    elif age <= 10:
        return "established"
    else:
        return "old"


def calculate_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price per square foot.
    
    Args:
        df: DataFrame with 'price' and 'area_sqft' columns
        
    Returns:
        DataFrame with added 'price_per_sqft' column
    """
    df = df.copy()
    df['price_per_sqft'] = df['price'] / df['area_sqft']
    return df


def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]


def remove_price_outliers_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove price outliers based on locality-wise price per sqft.
    
    Args:
        df: DataFrame with 'locality' and 'price_per_sqft' columns
        
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    df_out = pd.DataFrame()
    
    for locality in df['locality'].unique():
        locality_df = df[df['locality'] == locality]
        
        if len(locality_df) < 10:
            df_out = pd.concat([df_out, locality_df], ignore_index=True)
            continue
            
        mean = locality_df['price_per_sqft'].mean()
        std = locality_df['price_per_sqft'].std()
        
        # Keep data within 3 standard deviations
        reduced_df = locality_df[(locality_df['price_per_sqft'] >= (mean - 3 * std)) & 
                                  (locality_df['price_per_sqft'] <= (mean + 3 * std))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    
    return df_out


def preprocess_data(df: pd.DataFrame, is_training: bool = True, 
                    encoders: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete preprocessing pipeline for housing data.
    
    Args:
        df: Raw input DataFrame
        is_training: Whether this is training data
        encoders: Dictionary of fitted encoders (for prediction)
        
    Returns:
        Tuple of (processed DataFrame, encoders dictionary)
    """
    df = df.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Calculate price per sqft if price exists
    if 'price' in df.columns:
        df = calculate_price_per_sqft(df)
    
    # Create age buckets
    df['age_bucket'] = df['age'].apply(create_age_buckets)
    
    # Create total rooms feature
    df['total_rooms'] = df['bhk'] + df['bathrooms']
    
    # Encode categorical variables
    if is_training:
        encoders = {}
        
        # Label encode locality (target encoding would be better but requires target)
        encoders['locality'] = LabelEncoder()
        df['locality_encoded'] = encoders['locality'].fit_transform(df['locality'])
        
        # Label encode age bucket
        encoders['age_bucket'] = LabelEncoder()
        df['age_bucket_encoded'] = encoders['age_bucket'].fit_transform(df['age_bucket'])
        
        # Remove outliers (only during training)
        if 'price' in df.columns:
            df = remove_price_outliers_by_location(df)
            df = remove_outliers(df, 'price_per_sqft')
            df = remove_outliers(df, 'area_sqft')
    else:
        # Use existing encoders for prediction
        if encoders is None:
            raise ValueError("Encoders must be provided for prediction")
        
        # Handle unseen localities
        df['locality_encoded'] = df['locality'].apply(
            lambda x: encoders['locality'].transform([x])[0] 
            if x in encoders['locality'].classes_ else -1
        )
        
        # Handle unseen age buckets (shouldn't happen with our bucketing)
        df['age_bucket_encoded'] = df['age_bucket'].apply(
            lambda x: encoders['age_bucket'].transform([x])[0]
            if x in encoders['age_bucket'].classes_ else 0
        )
    
    return df, encoders


def get_feature_columns() -> List[str]:
    """
    Get the list of feature columns used for model training.
    
    Returns:
        List of feature column names
    """
    return [
        'area_sqft',
        'bhk',
        'bathrooms',
        'balconies',
        'floor',
        'age',
        'parking',
        'lift',
        'locality_encoded',
        'age_bucket_encoded',
        'total_rooms'
    ]


def prepare_input_for_prediction(input_dict: Dict[str, Any], encoders: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare user input for model prediction.
    
    Args:
        input_dict: Dictionary with user input
        encoders: Dictionary of fitted encoders
        
    Returns:
        Processed DataFrame ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_dict])
    
    # Preprocess
    df, _ = preprocess_data(df, is_training=False, encoders=encoders)
    
    # Select feature columns
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    
    return X
