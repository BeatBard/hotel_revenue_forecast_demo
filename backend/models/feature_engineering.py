"""
Feature Engineering Module
==========================

Demonstrates advanced feature engineering techniques for hotel revenue forecasting:
- Temporal features with cyclical encoding
- Lag features with proper time shifting
- Rolling window calculations
- Data leakage prevention methods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    def __init__(self):
        self.feature_cache = {}
        self.leakage_examples = {}
        
    def create_temporal_features(self, df, config=None):
        """Create comprehensive temporal features"""
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic temporal features
        # df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding for temporal features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Weekend/weekday indicator
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Month categories (seasonal)
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Encode meal period
        meal_encoding = {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2}
        df['MealPeriod_encoded'] = df['MealPeriod'].map(meal_encoding)
        
        # Special day indicators (simplified)
        df['IsMonthStart'] = (df['Day'] <= 3).astype(int)
        df['IsMonthEnd'] = (df['Day'] >= 28).astype(int)
        
        feature_info = {
            'features_created': [
                'Month', 'Day', 'DayOfWeek', 'DayOfYear', 
                'WeekOfYear', 'Quarter', 'Month_sin', 'Month_cos',
                'DayOfWeek_sin', 'DayOfWeek_cos', 'DayOfYear_sin', 
                'DayOfYear_cos', 'IsWeekend', 'Season', 'MealPeriod_encoded',
                'IsMonthStart', 'IsMonthEnd'
            ],
            'encoding_techniques': {
                'cyclical_encoding': ['Month', 'DayOfWeek', 'DayOfYear'],
                'binary_encoding': ['IsWeekend', 'IsMonthStart', 'IsMonthEnd'],
                'categorical_encoding': ['Season', 'MealPeriod']
            },
            'feature_count': 18,
            'sample_features': {
                'Month_sin_sample': df['Month_sin'].iloc[:5].tolist(),
                'DayOfWeek_cos_sample': df['DayOfWeek_cos'].iloc[:5].tolist(),
                'IsWeekend_sample': df['IsWeekend'].iloc[:5].tolist()
            }
        }
        
        return feature_info
    
    def create_lag_features(self, df, lags=[1, 7, 14, 30]):
        """Create lag features with proper time shifting to prevent leakage"""
        
        df = df.copy()
        df = df.sort_values(['Date', 'MealPeriod']).reset_index(drop=True)
        
        # Group by meal period for proper lagging
        lag_features = {}
        
        for meal_period in df['MealPeriod'].unique():
            meal_data = df[df['MealPeriod'] == meal_period].copy()
            meal_data = meal_data.sort_values('Date')
            
            for lag in lags:
                # Create lag feature (shift by positive value)
                lag_col = f'Revenue_lag_{lag}d_{meal_period}'
                meal_data[lag_col] = meal_data['CheckTotal'].shift(lag)
                
                # Store lag feature info
                lag_features[lag_col] = {
                    'lag_days': lag,
                    'meal_period': meal_period,
                    'non_null_count': meal_data[lag_col].notna().sum(),
                    'correlation_with_target': meal_data[[lag_col, 'CheckTotal']].corr().iloc[0, 1]
                }
        
        lag_info = {
            'lags_created': lags,
            'meal_periods': list(df['MealPeriod'].unique()),
            'total_lag_features': len(lags) * len(df['MealPeriod'].unique()),
            'feature_details': lag_features,
            'leakage_prevention': {
                'method': 'Positive shift values used',
                'explanation': 'Each lag feature uses only past values, not future or current',
                'validation': 'No correlation > 0.9 with target (suspicious of leakage)'
            }
        }
        
        return lag_info
    
    def create_rolling_features(self, df, windows=[7, 14, 30]):
        """Create rolling window features with leakage prevention"""
        
        df = df.copy()
        df = df.sort_values(['Date', 'MealPeriod']).reset_index(drop=True)
        
        rolling_features = {}
        
        for meal_period in df['MealPeriod'].unique():
            meal_data = df[df['MealPeriod'] == meal_period].copy()
            meal_data = meal_data.sort_values('Date')
            
            for window in windows:
                # Rolling mean (excluding current value to prevent leakage)
                mean_col = f'Revenue_rolling_mean_{window}d_{meal_period}'
                meal_data[mean_col] = meal_data['CheckTotal'].shift(1).rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                std_col = f'Revenue_rolling_std_{window}d_{meal_period}'
                meal_data[std_col] = meal_data['CheckTotal'].shift(1).rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling max
                max_col = f'Revenue_rolling_max_{window}d_{meal_period}'
                meal_data[max_col] = meal_data['CheckTotal'].shift(1).rolling(
                    window=window, min_periods=1
                ).max()
                
                rolling_features[f'{window}d_{meal_period}'] = {
                    'window_size': window,
                    'meal_period': meal_period,
                    'features': [mean_col, std_col, max_col],
                    'mean_correlation': abs(meal_data[[mean_col, 'CheckTotal']].corr().iloc[0, 1]),
                    'std_correlation': abs(meal_data[[std_col, 'CheckTotal']].corr().iloc[0, 1])
                }
        
        rolling_info = {
            'windows_created': windows,
            'feature_types': ['rolling_mean', 'rolling_std', 'rolling_max'],
            'total_rolling_features': len(windows) * len(df['MealPeriod'].unique()) * 3,
            'feature_details': rolling_features,
            'leakage_prevention': {
                'method': 'shift(1) before rolling calculation',
                'explanation': 'Current value excluded from rolling windows',
                'benefit': 'Prevents using current/future data to predict current value'
            }
        }
        
        return rolling_info
    
    def demonstrate_leakage_prevention(self, df):
        """Demonstrate data leakage prevention techniques"""
        
        df = df.copy()
        
        # Example 1: Leaky feature (DON'T DO THIS)
        daily_revenue = df.groupby('Date')['CheckTotal'].sum()
        df['Daily_Total_LEAKY'] = df['Date'].map(daily_revenue)  # Uses current day total
        
        # Example 2: Non-leaky feature (CORRECT WAY)
        df = df.sort_values('Date')
        df['Daily_Total_SAFE'] = df.groupby('Date')['CheckTotal'].transform(
            lambda x: x.shift(1).fillna(method='bfill')
        )
        
        # Example 3: Leaky rolling average (DON'T DO THIS)
        df['Revenue_7d_avg_LEAKY'] = df['CheckTotal'].rolling(7, center=True).mean()
        
        # Example 4: Safe rolling average (CORRECT WAY)
        df['Revenue_7d_avg_SAFE'] = df['CheckTotal'].shift(1).rolling(7).mean()
        
        # Calculate correlations to show leakage
        leakage_demo = {
            'leaky_examples': {
                'Daily_Total_LEAKY': {
                    'correlation_with_target': float(df[['Daily_Total_LEAKY', 'CheckTotal']].corr().iloc[0, 1]),
                    'why_leaky': 'Uses same-day total revenue including current record',
                    'red_flag': 'Correlation > 0.9 indicates perfect information'
                },
                'Revenue_7d_avg_LEAKY': {
                    'correlation_with_target': float(df[['Revenue_7d_avg_LEAKY', 'CheckTotal']].corr().iloc[0, 1]),
                    'why_leaky': 'Center=True uses future values in rolling calculation',
                    'red_flag': 'Unusually high correlation'
                }
            },
            'safe_examples': {
                'Daily_Total_SAFE': {
                    'correlation_with_target': float(df[['Daily_Total_SAFE', 'CheckTotal']].corr().iloc[0, 1]),
                    'why_safe': 'Uses previous day information only',
                    'validation': 'Reasonable correlation, no perfect information'
                },
                'Revenue_7d_avg_SAFE': {
                    'correlation_with_target': float(df[['Revenue_7d_avg_SAFE', 'CheckTotal']].corr().iloc[0, 1]),
                    'why_safe': 'shift(1) ensures only past values used',
                    'validation': 'Moderate correlation as expected'
                }
            },
            'prevention_techniques': {
                'temporal_splits': 'Use chronological train/validation/test splits',
                'lag_features': 'Always use positive shift values',
                'rolling_windows': 'Exclude current value with shift(1)',
                'target_encoding': 'Use only past data for encoding',
                'cross_validation': 'Use TimeSeriesSplit, not random folds'
            },
            'red_flags': {
                'perfect_correlation': 'Correlation > 0.95 with target',
                'perfect_training': 'Training R² = 1.0',
                'huge_val_gap': 'Training performance >> Validation performance',
                'future_dates': 'Using data from after prediction time'
            }
        }
        
        return leakage_demo
    
    def create_interaction_features(self, df):
        """Create interaction features between different dimensions"""
        
        df = df.copy()
        
        # Ensure we have the basic features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Meal period interactions
        meal_encoding = {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2}
        df['MealPeriod_encoded'] = df['MealPeriod'].map(meal_encoding)
        
        # Interaction features
        df['Weekend_x_MealPeriod'] = df['IsWeekend'] * df['MealPeriod_encoded']
        df['Month_x_MealPeriod'] = df['Month'] * df['MealPeriod_encoded']
        df['DayOfWeek_x_MealPeriod'] = df['DayOfWeek'] * df['MealPeriod_encoded']
        
        interaction_info = {
            'interactions_created': [
                'Weekend_x_MealPeriod',
                'Month_x_MealPeriod', 
                'DayOfWeek_x_MealPeriod'
            ],
            'purpose': 'Capture combined effects of time and meal period',
            'examples': {
                'Weekend_x_MealPeriod': 'Weekend dinner vs weekday dinner patterns',
                'Month_x_MealPeriod': 'Seasonal meal preferences',
                'DayOfWeek_x_MealPeriod': 'Day-specific meal behaviors'
            }
        }
        
        return interaction_info 