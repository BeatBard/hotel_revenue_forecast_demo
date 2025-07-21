"""
Ensemble Model Implementation
============================

Comprehensive ensemble model for hotel revenue forecasting following the original approach.
Includes proper data loading, temporal splits, feature engineering, and ensemble methods.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HotelRevenueEnsemble:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.ensemble_weights = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        self.training_data = None
        self.splits = None
        self.original_data = None
        
    def dubai_islamic_event_mapper(self, date):
        """Comprehensive mapper for Islamic periods and Dubai festivals affecting hotel revenue"""
        
        # Islamic Calendar Events
        ramadan_dates = {
            2023: (pd.to_datetime('2023-03-23'), pd.to_datetime('2023-04-20')),
            2024: (pd.to_datetime('2024-03-10'), pd.to_datetime('2024-04-08')),
            2025: (pd.to_datetime('2025-02-28'), pd.to_datetime('2025-03-29'))
        }
        
        eid_dates = {
            'Eid al-Fitr': {
                2023: pd.to_datetime('2023-04-21'),
                2024: pd.to_datetime('2024-04-09'),
                2025: pd.to_datetime('2025-03-30')
            },
            'Eid al-Adha': {
                2023: pd.to_datetime('2023-06-28'),
                2024: pd.to_datetime('2024-06-16'),
                2025: pd.to_datetime('2025-06-06')
            }
        }
        
        # Dubai Major Festivals & Events
        dubai_events = {
            'Dubai Shopping Festival': {
                2023: (pd.to_datetime('2023-12-15'), pd.to_datetime('2024-01-29')),
                2024: (pd.to_datetime('2024-12-06'), pd.to_datetime('2025-02-02')),
                2025: (pd.to_datetime('2025-12-06'), pd.to_datetime('2026-02-08'))
            },
            'Dubai World Cup': {
                2023: pd.to_datetime('2023-03-25'),
                2024: pd.to_datetime('2024-03-30'),
                2025: pd.to_datetime('2025-03-29')
            },
            'GITEX Technology Week': {
                2023: (pd.to_datetime('2023-10-16'), pd.to_datetime('2023-10-20')),
                2024: (pd.to_datetime('2024-10-14'), pd.to_datetime('2024-10-18')),
                2025: (pd.to_datetime('2025-10-13'), pd.to_datetime('2025-10-17'))
            },
            'New Year Celebrations': {
                2023: pd.to_datetime('2023-01-01'),
                2024: pd.to_datetime('2024-01-01'),
                2025: pd.to_datetime('2025-01-01')
            }
        }
        
        # National UAE Events
        uae_national_events = {
            'UAE National Day': {
                2023: pd.to_datetime('2023-12-02'),
                2024: pd.to_datetime('2024-12-02'),
                2025: pd.to_datetime('2025-12-02')
            }
        }
        
        date = pd.to_datetime(date)
        year = date.year
        
        # Check Ramadan periods
        if year in ramadan_dates:
            start, end = ramadan_dates[year]
            
            # Pre-Ramadan preparation
            if date >= (start - timedelta(days=7)) and date < start:
                return 'Pre-Ramadan'
            elif date >= start and date <= end:
                ramadan_duration = (end - start).days
                days_into_ramadan = (date - start).days
                
                if days_into_ramadan <= 9:
                    return 'Ramadan-First10Days'
                elif days_into_ramadan <= ramadan_duration - 10:
                    return 'Ramadan-Middle'
                else:
                    return 'Ramadan-Last10Days'
            elif date > end and date <= (end + timedelta(days=7)):
                return 'Post-Ramadan'
        
        # Check Eid holidays
        for eid_name, year_dates in eid_dates.items():
            if year in year_dates:
                eid_date = year_dates[year]
                if date >= (eid_date - timedelta(days=2)) and date <= (eid_date + timedelta(days=2)):
                    return f'{eid_name.replace(" ", "-")}'
        
        # Check Dubai events
        for event_name, event_dates in dubai_events.items():
            if year in event_dates:
                if isinstance(event_dates[year], tuple):
                    start, end = event_dates[year]
                    if date >= start and date <= end:
                        return event_name.replace(' ', '-')
                else:
                    event_date = event_dates[year]
                    if date >= (event_date - timedelta(days=1)) and date <= (event_date + timedelta(days=1)):
                        return event_name.replace(' ', '-')
        
        # Check UAE National events
        for event_name, event_dates in uae_national_events.items():
            if year in event_dates:
                event_date = event_dates[year]
                if date >= (event_date - timedelta(days=1)) and date <= (event_date + timedelta(days=1)):
                    return event_name.replace(' ', '-')
        
        return 'Normal'
    
    def add_dubai_event_features(self, df):
        """Add comprehensive Dubai event features to the dataframe"""
        
        # Only add IslamicPeriod if it doesn't exist (to avoid overwriting existing data)
        if 'IslamicPeriod' not in df.columns:
            df['IslamicPeriod'] = df['Date'].apply(self.dubai_islamic_event_mapper)
        
        # Islamic Period Binary Flags (only if they don't exist)
        if 'IsRamadan' not in df.columns:
            df['IsRamadan'] = df['IslamicPeriod'].str.contains('Ramadan', na=False).astype(int)
        if 'IsEid' not in df.columns:
            df['IsEid'] = df['IslamicPeriod'].str.contains('Eid', na=False).astype(int)
        if 'IsPreRamadan' not in df.columns:
            df['IsPreRamadan'] = (df['IslamicPeriod'] == 'Pre-Ramadan').astype(int)
        if 'IsPostRamadan' not in df.columns:
            df['IsPostRamadan'] = (df['IslamicPeriod'] == 'Post-Ramadan').astype(int)
        if 'IsLast10Ramadan' not in df.columns:
            df['IsLast10Ramadan'] = (df['IslamicPeriod'] == 'Ramadan-Last10Days').astype(int)
        
        # Festival Binary Flags - Create all event features that might exist in training data
        if 'IsDSF' not in df.columns:
            df['IsDSF'] = df['IslamicPeriod'].str.contains('Dubai-Shopping-Festival', na=False).astype(int)
        if 'IsSummerEvent' not in df.columns:
            df['IsSummerEvent'] = df['IslamicPeriod'].str.contains('Dubai-Summer-Surprises', na=False).astype(int)
        if 'IsWorldCup' not in df.columns:
            df['IsWorldCup'] = df['IslamicPeriod'].str.contains('Dubai-World-Cup', na=False).astype(int)
        if 'IsNationalDay' not in df.columns:
            df['IsNationalDay'] = df['IslamicPeriod'].str.contains('UAE-National-Day', na=False).astype(int)
        if 'IsNewYear' not in df.columns:
            df['IsNewYear'] = df['IslamicPeriod'].str.contains('New-Year-Celebrations', na=False).astype(int)
        if 'IsMarathon' not in df.columns:
            df['IsMarathon'] = df['IslamicPeriod'].str.contains('Marathon', na=False).astype(int)
        if 'IsGITEX' not in df.columns:
            df['IsGITEX'] = df['IslamicPeriod'].str.contains('GITEX', na=False).astype(int)
        if 'IsFilmFestival' not in df.columns:
            df['IsFilmFestival'] = df['IslamicPeriod'].str.contains('Film-Festival', na=False).astype(int)
        if 'IsAirshow' not in df.columns:
            df['IsAirshow'] = df['IslamicPeriod'].str.contains('Airshow', na=False).astype(int)
        if 'IsArtDubai' not in df.columns:
            df['IsArtDubai'] = df['IslamicPeriod'].str.contains('Art-Dubai', na=False).astype(int)
        if 'IsFoodFestival' not in df.columns:
            df['IsFoodFestival'] = df['IslamicPeriod'].str.contains('Food-Festival', na=False).astype(int)
        
        # Tourism Intensity Levels (only if it doesn't exist)
        if 'TourismIntensity' not in df.columns:
            high_tourism = ['Dubai-Shopping-Festival', 'New-Year-Celebrations', 'Dubai-World-Cup']
            medium_tourism = ['GITEX-Technology-Week']
            low_tourism = ['Ramadan-First10Days', 'Ramadan-Middle', 'Ramadan-Last10Days']
            
            df['TourismIntensity'] = 'Normal'
            df.loc[df['IslamicPeriod'].isin(high_tourism), 'TourismIntensity'] = 'High'
            df.loc[df['IslamicPeriod'].isin(medium_tourism), 'TourismIntensity'] = 'Medium'
            df.loc[df['IslamicPeriod'].isin(low_tourism), 'TourismIntensity'] = 'Low'
        
        # Revenue Impact Categories (only if it doesn't exist)
        if 'RevenueImpact' not in df.columns:
            revenue_boost = ['Dubai-Shopping-Festival', 'New-Year-Celebrations', 'Dubai-World-Cup']
            revenue_decrease = ['Ramadan-First10Days', 'Ramadan-Middle', 'Eid-al-Fitr', 'Eid-al-Adha']
            
            df['RevenueImpact'] = 'Neutral'
            df.loc[df['IslamicPeriod'].isin(revenue_boost), 'RevenueImpact'] = 'Boost'
            df.loc[df['IslamicPeriod'].isin(revenue_decrease), 'RevenueImpact'] = 'Decrease'
        
        return df
        
    def load_and_explore_data(self, df):
        """Load and perform comprehensive exploration of the revenue data"""
        logger.info("🎯 LOADING AND EXPLORING REVENUE CENTER 1 DATA")
        logger.info("=" * 80)
        
        # Store original data
        self.original_data = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info(f"Dataset Shape: {df.shape}")
        logger.info(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Total Days: {(df['Date'].max() - df['Date'].min()).days}")
        logger.info(f"Revenue Range: ${df['CheckTotal'].min():.2f} - ${df['CheckTotal'].max():.2f}")
        logger.info(f"Average Revenue: ${df['CheckTotal'].mean():.2f}")
        
        # Data quality checks
        logger.info("🔍 DATA QUALITY CHECKS:")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        logger.info(f"Duplicates: {df.duplicated(['Date', 'MealPeriod']).sum()}")
        logger.info(f"Zero revenue records: {(df['CheckTotal'] == 0).sum()}")
        
        # Revenue by meal period
        logger.info("📊 REVENUE BY MEAL PERIOD:")
        revenue_stats = df.groupby('MealPeriod')['CheckTotal'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        logger.info(f"\n{revenue_stats}")
        
        return df
    
    def create_temporal_splits(self, df):
        """Create temporal train/validation/test splits to prevent data leakage"""
        logger.info("📅 CREATING TEMPORAL SPLITS")
        logger.info("-" * 50)
        
        # Sort chronologically - CRITICAL for preventing leakage
        df_sorted = df.sort_values(['Date', 'MealPeriod']).reset_index(drop=True)
        
        # Define split boundaries
        total_records = len(df_sorted)
        train_end_idx = int(total_records * 0.6)
        val_end_idx = int(total_records * 0.8)
        
        train_data = df_sorted.iloc[:train_end_idx].copy()
        val_data = df_sorted.iloc[train_end_idx:val_end_idx].copy()
        test_data = df_sorted.iloc[val_end_idx:].copy()
        
        logger.info(
            f"Training: {len(train_data)} samples "
            f"({train_data['Date'].min()} to {train_data['Date'].max()})"
        )
        logger.info(
            f"Validation: {len(val_data)} samples "
            f"({val_data['Date'].min()} to {val_data['Date'].max()})"
        )
        logger.info(
            f"Test: {len(test_data)} samples "
            f"({test_data['Date'].min()} to {test_data['Date'].max()})"
        )
        
        self.splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        return self.splits

    def engineer_features(self, splits):
        """Create comprehensive features while preventing data leakage"""
        logger.info("🔧 FEATURE ENGINEERING (LEAKAGE-FREE)")
        logger.info("-" * 50)
        
        def safe_feature_engineering(data, is_training=False):
            """Create features without data leakage"""
            df = data.copy()
            df = df.sort_values(['Date', 'MealPeriod']).reset_index(drop=True)
            
            # Ensure proper data types and handle missing values
            numeric_columns = ['DayOfWeek', 'Month', 'Year', 'CheckTotal', 'is_zero',
                             'IsRamadan', 'IsEid', 'IsPreRamadan', 'IsPostRamadan', 'IsLast10Ramadan',
                             'IsDSF', 'IsSummerEvent', 'IsWorldCup', 'IsNationalDay', 'IsNewYear', 
                             'IsMarathon', 'IsGITEX', 'IsFilmFestival', 'IsAirshow', 'IsArtDubai', 
                             'IsFoodFestival', 'IsPreEvent', 'IsPostEvent']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN values with appropriate defaults
                    if col in ['DayOfWeek', 'Month', 'Year']:
                        df[col] = df[col].fillna(0)  # Temporal columns
                    elif col == 'CheckTotal':
                        df[col] = df[col].fillna(0)  # Revenue column
                    else:
                        df[col] = df[col].fillna(0)  # Binary event columns
            
            # Use existing temporal features from dataset when available
            # Only create additional temporal features if they don't exist
            if 'Year' not in df.columns:
                df['Year'] = df['Date'].dt.year.astype('int32')
            if 'Month' not in df.columns:
                df['Month'] = df['Date'].dt.month.astype('int32')
            if 'DayOfWeek' not in df.columns:
                df['DayOfWeek'] = df['Date'].dt.dayofweek.astype('int32')
            
            # Additional temporal features using the dataset's column naming convention
            # Handle NaN/inf values before converting to int32
            df['day'] = df['Date'].dt.day.fillna(1).astype('int32')
            df['day_of_year'] = df['Date'].dt.dayofyear.fillna(1).astype('int32')
            df['week_of_year'] = df['Date'].dt.isocalendar().week.fillna(1).astype('int32')
            df['quarter'] = df['Date'].dt.quarter.fillna(1).astype('int32')
            df['is_weekend'] = df['DayOfWeek'].fillna(0).isin([5, 6]).astype('int32')
            df['is_month_start'] = df['Date'].dt.is_month_start.fillna(False).astype('int32')
            df['is_month_end'] = df['Date'].dt.is_month_end.fillna(False).astype('int32')
            
            # Meal period encoding - use consistent mapping
            meal_mapping = {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2, 'Unknown': 3}
            df['MealPeriod'] = df['MealPeriod'].fillna('Unknown')
            df['meal_period_encoded'] = df['MealPeriod'].map(meal_mapping).fillna(3).astype('int32')
            
            # Add meal hour encoding
            meal_hour_mapping = {'Breakfast': 8, 'Lunch': 13, 'Dinner': 19, 'Unknown': 8}
            df['meal_hour'] = df['MealPeriod'].map(meal_hour_mapping).fillna(8).astype('int32')
            
            # Cyclical encoding for temporal features - ensure float64 for XGBoost
            # Use the actual column names from the dataset, but only if they're numeric
            for col, max_val in [('Month', 12), ('DayOfWeek', 7), ('quarter', 4), ('meal_period_encoded', 3), ('meal_hour', 24)]:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val).astype('float64')
                    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val).astype('float64')
            
            # Safe lag features (only past values) - ensure float64 for XGBoost
            lag_periods = [1, 2, 3, 7, 14, 21, 30]
            for lag in lag_periods:
                df[f'CheckTotal_lag_{lag}'] = df.groupby('meal_period_encoded')['CheckTotal'].shift(lag).fillna(0).astype('float64')
            
            # Safe rolling features (historical only) - ensure float64 for XGBoost  
            for window in [3, 7, 14, 21, 30]:
                df[f'CheckTotal_roll_{window}d_mean'] = (
                    df.groupby('meal_period_encoded')['CheckTotal']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .shift(1)  # Critical: shift to avoid current value
                    .reset_index(0, drop=True)
                ).astype('float64')
            
            # Critical: Meal period interaction features for differentiation
            # These are essential for the model to learn meal-specific patterns
            if pd.api.types.is_numeric_dtype(df['DayOfWeek']):
                interaction = (df['meal_period_encoded'] * df['DayOfWeek']).fillna(0)
                df['meal_dow_interaction'] = interaction.astype('int32')
            if pd.api.types.is_numeric_dtype(df['Month']):
                interaction = (df['meal_period_encoded'] * df['Month']).fillna(0)
                df['meal_month_interaction'] = interaction.astype('int32')
            if pd.api.types.is_numeric_dtype(df['Year']):
                interaction = (df['meal_period_encoded'] * df['Year']).fillna(0)
                df['meal_year_interaction'] = interaction.astype('int32')
            # Add weekend interaction for meal periods
            if 'is_weekend' in df.columns:
                interaction = (df['meal_period_encoded'] * df['is_weekend']).fillna(0)
                df['meal_weekend_interaction'] = interaction.astype('int32')
            
            # Forward fill missing values to maintain temporal consistency
            lag_cols = [col for col in df.columns if 'lag_' in col or 'roll_' in col]
            for col in lag_cols:
                df[col] = df.groupby('meal_period_encoded')[col].fillna(method='ffill')
                df[col] = df[col].fillna(df[col].median()).astype('float64')  # Final fallback with proper type
            
            return df
        
        # Apply feature engineering to each split
        logger.info("Creating features for training data...")
        train_features = safe_feature_engineering(splits['train'], is_training=True)
        
        logger.info("Creating features for validation data...")
        val_features = safe_feature_engineering(splits['validation'])
        
        logger.info("Creating features for test data...")
        test_features = safe_feature_engineering(splits['test'])
        
        # Identify feature columns (exclude target, metadata, and categorical text columns)
        exclude_cols = ['CheckTotal', 'Date', 'MealPeriod', 'RevenueCenterName', 'is_zero', 
                       'IslamicPeriod', 'TourismIntensity', 'RevenueImpact']
        feature_cols = [col for col in train_features.columns if col not in exclude_cols]
        
        # Filter for numeric columns only
        numeric_feature_cols = []
        for col in feature_cols:
            if col in train_features.columns and pd.api.types.is_numeric_dtype(train_features[col]):
                numeric_feature_cols.append(col)
        
        # Ensure consistent feature sets across all datasets
        train_cols = set(numeric_feature_cols)
        val_cols = set(val_features.columns)
        test_cols = set(test_features.columns)
        common_features = list(train_cols & val_cols & test_cols)
        
        # Final filter to ensure all are numeric in all datasets
        final_features = []
        for col in common_features:
            train_numeric = pd.api.types.is_numeric_dtype(train_features[col])
            val_numeric = pd.api.types.is_numeric_dtype(val_features[col])
            test_numeric = pd.api.types.is_numeric_dtype(test_features[col])
            if train_numeric and val_numeric and test_numeric:
                final_features.append(col)
        
        common_features = final_features
        
        # FEATURE SELECTION: Drop features with very low correlation to revenue
        logger.info("🔍 Analyzing feature correlations to revenue for feature selection...")
        correlation_threshold = 0.01  # Minimum correlation threshold
        
        # Calculate correlations with revenue using training data
        X_temp = train_features[common_features].fillna(0)
        y_temp = train_features['CheckTotal']
        
        feature_correlations = {}
        low_correlation_features = []
        
        for feature in common_features:
            try:
                correlation = abs(X_temp[feature].corr(y_temp))
                if not pd.isna(correlation):
                    feature_correlations[feature] = correlation
                    if correlation < correlation_threshold:
                        low_correlation_features.append(feature)
                else:
                    # Remove features with NaN correlation
                    low_correlation_features.append(feature)
            except:
                # Remove problematic features
                low_correlation_features.append(feature)
        
        # Remove low correlation features
        selected_features = [
            f for f in common_features if f not in low_correlation_features
        ]
        
        logger.info(f"   📊 Feature correlation analysis completed:")
        logger.info(f"   📈 Features before selection: {len(common_features)}")
        num_dropped = len(low_correlation_features)
        logger.info(f"   📉 Features with correlation < {correlation_threshold}: {num_dropped}")
        logger.info(f"   ✅ Features after selection: {len(selected_features)}")
        
        if low_correlation_features:
            dropped_preview = low_correlation_features[:5]
            more_text = '...' if len(low_correlation_features) > 5 else ''
            logger.info(f"   🗑️ Dropped low-correlation features: {dropped_preview}{more_text}")
        
        # Update feature list to selected features
        common_features = selected_features
        
        # Prepare final datasets with selected features only
        X_train = train_features[common_features].fillna(0)
        y_train = train_features['CheckTotal']
        
        X_val = val_features[common_features].fillna(0)
        y_val = val_features['CheckTotal']
        
        X_test = test_features[common_features].fillna(0)
        y_test = test_features['CheckTotal']
        
        logger.info(f"✅ Feature engineering completed:")
        logger.info(f"  Features created: {len(common_features)}")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")
        logger.info(f"  Test samples: {len(X_test)}")
        
        self.feature_names = common_features
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': common_features
        }
    
    def train_individual_models(self, df=None, model_types=None):
        """Train individual base models using proper temporal splits"""
        if df is None:
            df = self.training_data
        if df is None:
            return {'error': 'No training data available. Please load data first.'}
            
        if model_types is None:
            model_types = ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
        
        logger.info(f"🔧 Starting individual model training for {len(model_types)} models...")
        
        # Use existing splits if available, otherwise create them
        if self.splits is None:
            splits = self.create_temporal_splits(df)
        else:
            splits = self.splits
            
        # Engineer features
        dataset = self.engineer_features(splits)
        
        logger.info(f"📊 Training with {len(dataset['X_train'])} samples, {len(dataset['feature_names'])} features")
        
        # Extract training data
        X_train, X_val = dataset['X_train'], dataset['X_val']
        y_train, y_val = dataset['y_train'], dataset['y_val']
        
        # Scale features for linear models
        logger.info("🔄 Scaling features for linear models...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['standard'] = scaler
        
        results = {}
        logger.info(f"🎯 Starting training of {len(model_types)} individual models...")
        
        total_models = len(model_types)
        current_model = 0
        
        # Ridge Regression
        if 'ridge' in model_types:
            current_model += 1
            logger.info(f"🔵 [{current_model}/{total_models}] Training Ridge Regression...")
            ridge = Ridge(alpha=1.0, random_state=self.random_state)
            ridge.fit(X_train_scaled, y_train)
            ridge_pred = ridge.predict(X_val_scaled)
            
            r2_score_val = r2_score(y_val, ridge_pred)
            mae_val = mean_absolute_error(y_val, ridge_pred)
            
            self.models['ridge'] = ridge
            results['ridge'] = {
                'r2': float(r2_score_val),
                'mae': float(mae_val),
                'rmse': float(np.sqrt(mean_squared_error(y_val, ridge_pred)))
            }
            logger.info(f"   ✅ Ridge completed! R² = {r2_score_val:.3f}, MAE = ${mae_val:.0f}")
        
        # Random Forest
        if 'random_forest' in model_types:
            current_model += 1
            logger.info(f"🌳 [{current_model}/{total_models}] Training Random Forest...")
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_val)
            
            r2_score_val = r2_score(y_val, rf_pred)
            mae_val = mean_absolute_error(y_val, rf_pred)
            
            self.models['random_forest'] = rf
            results['random_forest'] = {
                'r2': float(r2_score_val),
                'mae': float(mae_val),
                'rmse': float(np.sqrt(mean_squared_error(y_val, rf_pred)))
            }
            logger.info(f"   ✅ Random Forest completed! R² = {r2_score_val:.3f}, MAE = ${mae_val:.0f}")
        
        # XGBoost
        if 'xgboost' in model_types:
            current_model += 1
            logger.info(f"🚀 [{current_model}/{total_models}] Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0  # Suppress XGBoost output
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_val)
            
            r2_score_val = r2_score(y_val, xgb_pred)
            mae_val = mean_absolute_error(y_val, xgb_pred)
            
            self.models['xgboost'] = xgb_model
            results['xgboost'] = {
                'r2': float(r2_score_val),
                'mae': float(mae_val),
                'rmse': float(np.sqrt(mean_squared_error(y_val, xgb_pred)))
            }
            logger.info(f"   ✅ XGBoost completed! R² = {r2_score_val:.3f}, MAE = ${mae_val:.0f}")
        
        # LightGBM
        if 'lightgbm' in model_types:
            current_model += 1
            logger.info(f"⚡ [{current_model}/{total_models}] Training LightGBM...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_val)
            
            r2_score_val = r2_score(y_val, lgb_pred)
            mae_val = mean_absolute_error(y_val, lgb_pred)
            
            self.models['lightgbm'] = lgb_model
            results['lightgbm'] = {
                'r2': float(r2_score_val),
                'mae': float(mae_val),
                'rmse': float(np.sqrt(mean_squared_error(y_val, lgb_pred)))
            }
            logger.info(f"   ✅ LightGBM completed! R² = {r2_score_val:.3f}, MAE = ${mae_val:.0f}")
        
        # Gradient Boosting
        if 'gradient_boosting' in model_types:
            current_model += 1
            logger.info(f"📈 [{current_model}/{total_models}] Training Gradient Boosting...")
            gb = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            gb.fit(X_train, y_train)
            gb_pred = gb.predict(X_val)
            
            r2_score_val = r2_score(y_val, gb_pred)
            mae_val = mean_absolute_error(y_val, gb_pred)
            
            self.models['gradient_boosting'] = gb
            results['gradient_boosting'] = {
                'r2': r2_score_val,
                'mae': mae_val,
                'rmse': np.sqrt(mean_squared_error(y_val, gb_pred))
            }
            logger.info(f"   ✅ Gradient Boosting completed! R² = {r2_score_val:.3f}, MAE = ${mae_val:.0f}")
        
        self.results['individual_models'] = results
        
        # Log summary
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        logger.info(f"🏆 Best individual model: {best_model[0]} with R² = {best_model[1]['r2']:.3f}")
        logger.info(f"📊 Individual model training completed successfully!")
        
        return {
            'success': True,
            'models_trained': list(results.keys()),
            'performance': results,
            'features_used': len(self.feature_names),
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
    
    def train_ensemble(self, df, ensemble_types=None):
        """Train ensemble models with different strategies"""
        if ensemble_types is None:
            ensemble_types = ['simple', 'weighted', 'top3', 'median']
        
        logger.info("🎯 Starting ensemble model training...")
        logger.info(f"   📋 Ensemble strategies: {', '.join(ensemble_types)}")
        
        # First ensure individual models are trained
        if not self.models:
            logger.info("   ⚠️ Individual models not found, training them first...")
            self.train_individual_models(df)
        
        logger.info("   🔄 Generating predictions from individual models...")
        
        # Use the same data splitting approach as individual models
        if df is None and self.training_data is not None:
            df = self.training_data
        
        # Engineer features using the same approach as individual training
        splits = self.create_temporal_splits(df)
        dataset = self.engineer_features(splits)
        
        # Extract validation data (same as used in individual model training)
        X_val, y_val = dataset['X_val'], dataset['y_val']
        
        # Get individual model predictions on validation set
        predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'ridge':
                # Use scaled features for ridge
                X_val_scaled = self.scalers['standard'].transform(X_val)
                pred = model.predict(X_val_scaled)
            else:
                pred = model.predict(X_val)
            predictions[model_name] = pred
        
        logger.info("   ⚖️ Calculating ensemble weights...")
        
        ensemble_results = {}
        
        logger.info("   🧮 Creating ensemble combinations...")
        
        # Simple Average Ensemble
        if 'simple' in ensemble_types:
            simple_pred = np.mean(list(predictions.values()), axis=0)
            ensemble_results['simple_average'] = {
                'r2': r2_score(y_val, simple_pred),
                'mae': mean_absolute_error(y_val, simple_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, simple_pred))
            }
        
        # Weighted Average (based on individual R² scores)
        if 'weighted' in ensemble_types:
            weights = []
            for model_name in predictions.keys():
                r2 = self.results['individual_models'][model_name]['r2']
                weights.append(max(0, r2))  # Ensure non-negative weights
            
            # Normalize weights
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            weighted_pred = np.average(list(predictions.values()), axis=0, weights=weights)
            ensemble_results['weighted_average'] = {
                'r2': r2_score(y_val, weighted_pred),
                'mae': mean_absolute_error(y_val, weighted_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, weighted_pred)),
                'weights': {name: float(w) for name, w in zip(predictions.keys(), weights)}
            }
        
        # Top-3 Models Ensemble
        if 'top3' in ensemble_types:
            # Get top 3 models by R² score
            model_scores = {name: self.results['individual_models'][name]['r2'] 
                          for name in predictions.keys()}
            top3_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            top3_predictions = [predictions[name] for name, _ in top3_models]
            top3_pred = np.mean(top3_predictions, axis=0)
            
            ensemble_results['top3_average'] = {
                'r2': r2_score(y_val, top3_pred),
                'mae': mean_absolute_error(y_val, top3_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, top3_pred)),
                'models_used': [name for name, _ in top3_models]
            }
        
        # Median Ensemble
        if 'median' in ensemble_types:
            median_pred = np.median(list(predictions.values()), axis=0)
            ensemble_results['median'] = {
                'r2': r2_score(y_val, median_pred),
                'mae': mean_absolute_error(y_val, median_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, median_pred))
            }
        
        self.results['ensemble_models'] = ensemble_results
        
        # Log success
        best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['r2'])
        logger.info(f"🏆 Best ensemble strategy: {best_ensemble[0]} with R² = {best_ensemble[1]['r2']:.3f}")
        logger.info("✅ Ensemble model training completed successfully!")
        
        return {
            'success': True,
            'ensemble_strategies': list(ensemble_results.keys()),
            'performance': ensemble_results,
            'best_ensemble': best_ensemble,
            'improvement_over_best_individual': self._calculate_ensemble_improvement()
        }
    
    def _calculate_ensemble_improvement(self):
        """Calculate improvement of ensemble over best individual model"""
        if 'individual_models' not in self.results or 'ensemble_models' not in self.results:
            return None
        
        best_individual_r2 = max(
            model['r2'] for model in self.results['individual_models'].values()
        )
        best_ensemble_r2 = max(
            model['r2'] for model in self.results['ensemble_models'].values()
        )
        
        return {
            'best_individual_r2': best_individual_r2,
            'best_ensemble_r2': best_ensemble_r2,
            'improvement': best_ensemble_r2 - best_individual_r2,
            'improvement_percentage': ((best_ensemble_r2 - best_individual_r2) / best_individual_r2) * 100
        }
    
    def compare_model_performance(self):
        """Get comprehensive model performance comparison"""
        if not self.results:
            return {'error': 'No models trained yet'}
        
        comparison = {
            'individual_models': self.results.get('individual_models', {}),
            'ensemble_models': self.results.get('ensemble_models', {}),
            'summary': {
                'total_models': len(self.models),
                'features_used': len(self.feature_names),
                'feature_names': self.feature_names
            }
        }
        
        # Find best performing models
        if self.results.get('individual_models'):
            best_individual = max(
                self.results['individual_models'].items(),
                key=lambda x: x[1]['r2']
            )
            comparison['best_individual'] = {
                'model': best_individual[0],
                'performance': best_individual[1]
            }
        
        if self.results.get('ensemble_models'):
            best_ensemble = max(
                self.results['ensemble_models'].items(),
                key=lambda x: x[1]['r2']
            )
            comparison['best_ensemble'] = {
                'strategy': best_ensemble[0],
                'performance': best_ensemble[1]
            }
        
        return comparison
    
    def generate_predictions(self, df=None, model_name='best'):
        """Generate predictions using specified model"""
        if df is None:
            return {'error': 'No data provided for predictions'}
            
        if not self.models:
            return {'error': 'No trained models available for prediction'}
        
        if model_name == 'best':
            # Use best performing ensemble or individual model
            all_results = {**self.results.get('individual_models', {}), 
                          **self.results.get('ensemble_models', {})}
            if all_results:
                model_name = max(all_results.items(), key=lambda x: x[1]['r2'])[0]
            else:
                return {'error': 'No models available for prediction'}
        
        # Generate predictions based on model type
        if model_name in self.models:
            model = self.models[model_name]
            if model_name == 'ridge':
                X_scaled = self.scalers['standard'].transform(X)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X)
        else:
            return {'error': f'Model {model_name} not found'}
        
        return {
            'model_used': model_name,
            'predictions': predictions.tolist(),
            'actual_values': y.tolist(),
            'performance': {
                'r2': r2_score(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions))
            }
        }
    
    def get_feature_importance(self, model_name='xgboost'):
        """Get feature importance from tree-based models"""
        # Check if models are trained
        if not self.models:
            return {'error': 'No models have been trained yet. Please train models first.'}
        
        if model_name not in self.models:
            available_models = list(self.models.keys())
            return {
                'error': f'Model {model_name} not found. Available models: {available_models}'
            }
        
        # Check if feature names are available
        if not self.feature_names:
            return {'error': 'Feature names not available. Please ensure data has been processed.'}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return {'error': f'Model {model_name} does not support feature importance'}
        
        # Ensure importance array matches feature names length
        if len(importance) != len(self.feature_names):
            return {
                'error': f'Feature importance length ({len(importance)}) does not match feature names length ({len(self.feature_names)})'
            }
        
        feature_importance = {
            'features': self.feature_names,
            'importance': [float(x) for x in importance],
            'sorted_features': [
                (feature, float(imp)) 
                for feature, imp in sorted(
                    zip(self.feature_names, importance),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
        }
        
        return feature_importance
    
    def analyze_original_data_features(self):
        """Analyze feature importance on the original dataset using statistical methods"""
        if self.original_data is None:
            return {'error': 'No original data available. Please load data first.'}
        
        logger.info("🔍 ANALYZING ORIGINAL DATA FEATURES")
        logger.info("-" * 50)
        
        try:
            df = self.original_data.copy()
            
            # Encode categorical features
            meal_encoder = LabelEncoder()
            df['meal_period_encoded'] = meal_encoder.fit_transform(df['MealPeriod'])
            
            # Tourism Intensity encoding
            if 'TourismIntensity' in df.columns:
                tourism_encoder = LabelEncoder()
                df['tourism_intensity_encoded'] = tourism_encoder.fit_transform(df['TourismIntensity'])
            
            # Revenue Impact encoding
            if 'RevenueImpact' in df.columns:
                impact_encoder = LabelEncoder()
                df['revenue_impact_encoded'] = impact_encoder.fit_transform(df['RevenueImpact'])
            
            # Islamic Period encoding
            if 'IslamicPeriod' in df.columns:
                islamic_encoder = LabelEncoder()
                df['islamic_period_encoded'] = islamic_encoder.fit_transform(df['IslamicPeriod'])
            
            # Add cyclical encoding for temporal features
            for col, max_val in [('Month', 12), ('DayOfWeek', 7)]:
                if col in df.columns:
                    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
                    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
            
            # Define feature categories for comprehensive analysis
            feature_categories = {
                'temporal_basic': ['Month', 'DayOfWeek', 'Year'],
                'temporal_cyclical': [col for col in df.columns if col.endswith('_sin') or col.endswith('_cos')],
                'encoded_categorical': ['meal_period_encoded', 'tourism_intensity_encoded', 
                                      'revenue_impact_encoded', 'islamic_period_encoded'],
                'islamic_events': ['IsRamadan', 'IsEid', 'IsPreRamadan', 'IsPostRamadan', 'IsLast10Ramadan'],
                'dubai_events': ['IsDSF', 'IsSummerEvent', 'IsWorldCup', 'IsNationalDay', 'IsNewYear', 
                               'IsMarathon', 'IsGITEX', 'IsFilmFestival', 'IsAirshow', 'IsArtDubai', 'IsFoodFestival'],
                'event_timing': ['IsPreEvent', 'IsPostEvent'],
                'other_flags': ['is_zero']
            }
            
            # Get all numeric features that exist in the dataframe
            all_numeric_features = []
            for category, features in feature_categories.items():
                for feature in features:
                    if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                        all_numeric_features.append(feature)
            
            # Calculate correlations with target
            feature_correlations = {}
            target = df['CheckTotal']
            
            logger.info(f"📊 Analyzing {len(all_numeric_features)} features across {len(feature_categories)} categories")
            
            for col in all_numeric_features:
                if col in df.columns and col != 'CheckTotal':
                    correlation = abs(df[col].corr(target))
                    if not np.isnan(correlation):
                        feature_correlations[col] = float(correlation)
            
            # Sort by correlation strength
            sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Organize features by category for analysis
            features_by_category = {}
            for category, features in feature_categories.items():
                category_features = []
                for feature in features:
                    if feature in feature_correlations:
                        category_features.append((feature, feature_correlations[feature]))
                category_features.sort(key=lambda x: x[1], reverse=True)
                features_by_category[category] = category_features
            
            # Detailed categorical analysis
            categorical_analysis = {}
            
            # Meal period analysis
            if 'MealPeriod' in df.columns:
                meal_stats = df.groupby('MealPeriod')['CheckTotal'].agg(['mean', 'std', 'count']).round(2)
                categorical_analysis['meal_period'] = {
                    'feature_name': 'MealPeriod',
                    'categories': {k: {key: float(val) if key != 'count' else int(val) 
                                     for key, val in v.items()} for k, v in meal_stats.to_dict('index').items()},
                    'importance_score': float(feature_correlations.get('meal_period_encoded', 0))
                }
            
            # Day of week analysis
            if 'DayOfWeek' in df.columns:
                dow_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                              4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                df['day_name'] = df['DayOfWeek'].map(dow_mapping)
                dow_stats = df.groupby('day_name')['CheckTotal'].agg(['mean', 'std', 'count']).round(2)
                categorical_analysis['day_of_week'] = {
                    'feature_name': 'DayOfWeek',
                    'categories': {k: {key: float(val) if key != 'count' else int(val) 
                                     for key, val in v.items()} for k, v in dow_stats.to_dict('index').items()},
                    'importance_score': float(feature_correlations.get('DayOfWeek', 0))
                }
            
            # Month analysis
            if 'Month' in df.columns:
                month_stats = df.groupby('Month')['CheckTotal'].agg(['mean', 'std', 'count']).round(2)
                categorical_analysis['month'] = {
                    'feature_name': 'Month',
                    'categories': {k: {key: float(val) if key != 'count' else int(val) 
                                     for key, val in v.items()} for k, v in month_stats.to_dict('index').items()},
                    'importance_score': float(feature_correlations.get('Month', 0))
                }
            
            # Tourism Intensity analysis
            if 'TourismIntensity' in df.columns:
                tourism_stats = df.groupby('TourismIntensity')['CheckTotal'].agg(['mean', 'std', 'count']).round(2)
                categorical_analysis['tourism_intensity'] = {
                    'feature_name': 'TourismIntensity',
                    'categories': {k: {key: float(val) if key != 'count' else int(val) 
                                     for key, val in v.items()} for k, v in tourism_stats.to_dict('index').items()},
                    'importance_score': float(feature_correlations.get('tourism_intensity_encoded', 0))
                }
            
            # Revenue Impact analysis
            if 'RevenueImpact' in df.columns:
                impact_stats = df.groupby('RevenueImpact')['CheckTotal'].agg(['mean', 'std', 'count']).round(2)
                categorical_analysis['revenue_impact'] = {
                    'feature_name': 'RevenueImpact',
                    'categories': {k: {key: float(val) if key != 'count' else int(val) 
                                     for key, val in v.items()} for k, v in impact_stats.to_dict('index').items()},
                    'importance_score': float(feature_correlations.get('revenue_impact_encoded', 0))
                }
            
            # Islamic Period analysis
            if 'IslamicPeriod' in df.columns:
                islamic_stats = df.groupby('IslamicPeriod')['CheckTotal'].agg(['mean', 'std', 'count']).round(2)
                categorical_analysis['islamic_period'] = {
                    'feature_name': 'IslamicPeriod',
                    'categories': {k: {key: float(val) if key != 'count' else int(val) 
                                     for key, val in v.items()} for k, v in islamic_stats.to_dict('index').items()},
                    'importance_score': float(feature_correlations.get('islamic_period_encoded', 0))
                }
            
            # Event impact analysis
            event_impact_analysis = {}
            islamic_events = ['IsRamadan', 'IsEid', 'IsPreRamadan', 'IsPostRamadan', 'IsLast10Ramadan']
            dubai_events = ['IsDSF', 'IsSummerEvent', 'IsWorldCup', 'IsNationalDay', 'IsNewYear', 
                           'IsMarathon', 'IsGITEX', 'IsFilmFestival', 'IsAirshow', 'IsArtDubai', 'IsFoodFestival']
            
            for event_list, category_name in [(islamic_events, 'Islamic Events'), (dubai_events, 'Dubai Events')]:
                category_analysis = {}
                for event in event_list:
                    if event in df.columns:
                        event_revenue = df[df[event] == 1]['CheckTotal'].mean() if (df[event] == 1).any() else 0
                        normal_revenue = df[df[event] == 0]['CheckTotal'].mean() if (df[event] == 0).any() else 0
                        impact = ((event_revenue - normal_revenue) / normal_revenue * 100) if normal_revenue > 0 else 0
                        category_analysis[event] = {
                            'event_avg_revenue': float(event_revenue),
                            'normal_avg_revenue': float(normal_revenue),
                            'impact_percentage': float(impact),
                            'correlation': float(feature_correlations.get(event, 0)),
                            'event_days': int((df[event] == 1).sum())
                        }
                event_impact_analysis[category_name] = category_analysis
            
            # Generate comprehensive insights
            insights = []
            
            # Revenue distribution insights
            if 'MealPeriod' in df.columns:
                total_revenue = float(df['CheckTotal'].sum())
                meal_revenue = df.groupby('MealPeriod')['CheckTotal'].sum()
                for meal, revenue in meal_revenue.items():
                    percentage = (float(revenue) / total_revenue) * 100
                    insights.append({
                        'category': 'Revenue Distribution',
                        'insight': f"{meal} accounts for {percentage:.1f}% of total revenue",
                        'value': float(percentage),
                        'importance': 'high' if percentage > 35 else 'medium'
                    })
            
            # Temporal patterns
            if 'DayOfWeek' in df.columns:
                weekend_days = [5, 6]  # Saturday, Sunday
                weekend_avg = df[df['DayOfWeek'].isin(weekend_days)]['CheckTotal'].mean()
                weekday_avg = df[~df['DayOfWeek'].isin(weekend_days)]['CheckTotal'].mean()
                weekend_diff = ((weekend_avg - weekday_avg) / weekday_avg) * 100 if weekday_avg > 0 else 0
                
                insights.append({
                    'category': 'Temporal Patterns',
                    'insight': f"Weekend revenue is {abs(weekend_diff):.1f}% {'higher' if weekend_diff > 0 else 'lower'} than weekdays",
                    'value': float(weekend_diff),
                    'importance': 'high' if abs(weekend_diff) > 10 else 'medium'
                })
            
            # Event impact insights
            top_events = []
            for category_name, category_analysis in event_impact_analysis.items():
                for event, analysis in category_analysis.items():
                    if analysis['impact_percentage'] != 0:
                        top_events.append((event, analysis['impact_percentage'], category_name))
            
            top_events.sort(key=lambda x: abs(x[1]), reverse=True)
            for event, impact, category in top_events[:5]:  # Top 5 events
                insights.append({
                    'category': f'{category} Impact',
                    'insight': f"{event} {'increases' if impact > 0 else 'decreases'} revenue by {abs(impact):.1f}%",
                    'value': float(impact),
                    'importance': 'high' if abs(impact) > 15 else 'medium'
                })
            
            # Tourism impact
            if 'TourismIntensity' in df.columns:
                tourism_impact = df.groupby('TourismIntensity')['CheckTotal'].mean()
                if len(tourism_impact) > 1:
                    max_tourism = tourism_impact.max()
                    min_tourism = tourism_impact.min()
                    tourism_diff = ((max_tourism - min_tourism) / min_tourism) * 100 if min_tourism > 0 else 0
                    insights.append({
                        'category': 'Tourism Impact',
                        'insight': f"High tourism periods generate {tourism_diff:.1f}% more revenue than low periods",
                        'value': float(tourism_diff),
                        'importance': 'high' if tourism_diff > 20 else 'medium'
                    })
            
            logger.info(f"✅ Original data feature analysis completed")
            logger.info(f"   📊 Analyzed {len(feature_correlations)} features")
            logger.info(f"   🔝 Top feature: {sorted_features[0][0]} (correlation: {sorted_features[0][1]:.3f})" if sorted_features else "No features found")
            logger.info(f"   🎪 Islamic events analyzed: {len([f for f in islamic_events if f in df.columns])}")
            logger.info(f"   🏙️ Dubai events analyzed: {len([f for f in dubai_events if f in df.columns])}")
            
            return {
                'feature_correlations': feature_correlations,
                'sorted_features': sorted_features,
                'features_by_category': features_by_category,
                'categorical_analysis': categorical_analysis,
                'event_impact_analysis': event_impact_analysis,
                'insights': insights,
                'data_summary': {
                    'total_records': len(df),
                    'total_features_analyzed': len(feature_correlations),
                    'feature_categories': {cat: len(feats) for cat, feats in features_by_category.items()},
                    'date_range': {
                        'start': df['Date'].min().strftime('%Y-%m-%d'),
                        'end': df['Date'].max().strftime('%Y-%m-%d')
                    },
                    'revenue_stats': {
                        'total': float(df['CheckTotal'].sum()),
                        'mean': float(df['CheckTotal'].mean()),
                        'std': float(df['CheckTotal'].std()),
                        'min': float(df['CheckTotal'].min()),
                        'max': float(df['CheckTotal'].max())
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing original data features: {str(e)}")
            return {'error': f'Error analyzing features: {str(e)}'}
        
        return feature_importance
    
    def get_evaluation_metrics(self):
        """Get comprehensive evaluation metrics"""
        return {
            'individual_models': self.results.get('individual_models', {}),
            'ensemble_models': self.results.get('ensemble_models', {}),
            'model_comparison': self.compare_model_performance(),
            'feature_count': len(self.feature_names),
            'training_summary': {
                'total_models_trained': len(self.models),
                'ensemble_strategies': len(self.results.get('ensemble_models', {})),
                'best_performance': self._get_best_performance()
            }
        }
    
    def _get_best_performance(self):
        """Get the best performance across all models"""
        all_results = {**self.results.get('individual_models', {}), 
                      **self.results.get('ensemble_models', {})}
        
        if not all_results:
            return None
        
        best_model = max(all_results.items(), key=lambda x: x[1]['r2'])
        return {
            'model_name': best_model[0],
            'performance': best_model[1]
        }
    
    def create_evaluation_visualization(self, viz_type='predictions_vs_actual'):
        """Create evaluation visualizations (placeholder for demo)"""
        # This would integrate with plotting libraries in a full implementation
        return {
            'visualization_type': viz_type,
            'message': 'Visualization would be generated here',
            'available_types': [
                'predictions_vs_actual',
                'residuals_plot', 
                'feature_importance',
                'model_comparison'
            ]
        }
    
    def create_prediction_dataframe(self, date, meal_period):
        """Create a single-row dataframe for prediction with proper meal period features"""
        
        # Create basic dataframe with all the columns from the original dataset structure
        prediction_data = pd.DataFrame({
            'Date': [pd.to_datetime(date)],
            'MealPeriod': [meal_period],
            'RevenueCenterName': ['RevenueCenter_1'],
            'CheckTotal': [0],  # Will be predicted
            'is_zero': [0]
        })
        
        # Add basic temporal features that exist in original dataset
        prediction_data['DayOfWeek'] = prediction_data['Date'].dt.dayofweek
        prediction_data['Month'] = prediction_data['Date'].dt.month
        prediction_data['Year'] = prediction_data['Date'].dt.year
        
        # Critical: Use EXACT same meal period encoding as training data
        # This must match the encoding used in engineer_features()
        meal_mapping = {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2, 'Unknown': 3}
        prediction_data['meal_period_encoded'] = meal_mapping.get(meal_period, 0)
        
        # Add meal-specific hour encoding (important differentiator)  
        meal_hour_mapping = {'Breakfast': 8, 'Lunch': 13, 'Dinner': 19, 'Unknown': 8}
        prediction_data['meal_hour'] = meal_hour_mapping.get(meal_period, 8)
        
        # Add cyclical encoding that matches training exactly
        prediction_data['meal_period_encoded_sin'] = np.sin(2 * np.pi * prediction_data['meal_period_encoded'] / 3)
        prediction_data['meal_period_encoded_cos'] = np.cos(2 * np.pi * prediction_data['meal_period_encoded'] / 3)
        prediction_data['meal_hour_sin'] = np.sin(2 * np.pi * prediction_data['meal_hour'] / 24)
        prediction_data['meal_hour_cos'] = np.cos(2 * np.pi * prediction_data['meal_hour'] / 24)
        
        # Add additional temporal features needed for feature engineering
        prediction_data['day'] = prediction_data['Date'].dt.day
        prediction_data['day_of_year'] = prediction_data['Date'].dt.dayofyear
        prediction_data['week_of_year'] = prediction_data['Date'].dt.isocalendar().week
        prediction_data['quarter'] = prediction_data['Date'].dt.quarter
        prediction_data['is_weekend'] = prediction_data['DayOfWeek'].isin([5, 6]).astype(int)
        prediction_data['is_month_start'] = prediction_data['Date'].dt.is_month_start.astype(int)
        prediction_data['is_month_end'] = prediction_data['Date'].dt.is_month_end.astype(int)
        
        # Initialize all event columns with default values (0) before applying event mapper
        event_columns = [
            'IsRamadan', 'IsEid', 'IsPreRamadan', 'IsPostRamadan', 'IsLast10Ramadan',
            'IsDSF', 'IsSummerEvent', 'IsWorldCup', 'IsNationalDay', 'IsNewYear', 
            'IsMarathon', 'IsGITEX', 'IsFilmFestival', 'IsAirshow', 'IsArtDubai', 
            'IsFoodFestival', 'IsPreEvent', 'IsPostEvent'
        ]
        
        for col in event_columns:
            prediction_data[col] = 0
        
        # Initialize categorical columns
        prediction_data['IslamicPeriod'] = 'Normal'
        prediction_data['TourismIntensity'] = 'Normal'
        prediction_data['RevenueImpact'] = 'Neutral'
        
        # Now apply Dubai event features using our mapper (this will update the values)
        prediction_data = self.add_dubai_event_features(prediction_data)
        
        # No synthetic base patterns - use only real features from data
        
        # Initialize lag features with zero (no historical data available for prediction)
        lag_periods = [1, 2, 3, 7, 14, 21, 30]
        for lag in lag_periods:
            prediction_data[f'CheckTotal_lag_{lag}'] = 0.0
        
        # Initialize rolling features with zero (no historical data available for prediction)
        for window in [3, 7, 14, 21, 30]:
            prediction_data[f'CheckTotal_roll_{window}d_mean'] = 0.0
        
        # Initialize interaction features
        prediction_data['meal_dow_interaction'] = prediction_data['meal_period_encoded'] * prediction_data['DayOfWeek']
        prediction_data['meal_month_interaction'] = prediction_data['meal_period_encoded'] * prediction_data['Month']
        
        return prediction_data
    

    
    def predict_single(self, date, meal_period, model_type='best_ensemble'):
        """Predict revenue for a specific date and meal period using trained models"""
        
        if not self.models:
            return {'error': 'Models not trained yet. Please train models first.'}
        
        if not hasattr(self, 'feature_names') or not self.feature_names:
            return {'error': 'Model features not available. Please retrain models.'}
        
        try:
            logger.info(f"🔮 Making prediction for {date} {meal_period}")
            
            # Create prediction dataframe with required features
            pred_df = self.create_prediction_dataframe(date, meal_period)
            logger.info(f"   📊 Created prediction dataframe with {len(pred_df.columns)} columns")
            
            # Apply same feature engineering as training (but simpler for single prediction)
            pred_df = self._apply_prediction_features(pred_df)
            logger.info(f"   🔧 Applied features, now {len(pred_df.columns)} columns")
            
            # Select only the features the model was trained on
            missing_features = [f for f in self.feature_names if f not in pred_df.columns]
            if missing_features:
                logger.info(f"   ⚠️ Creating {len(missing_features)} missing features with defaults")
                # Create missing features with default values
                for feature in missing_features:
                    pred_df[feature] = 0.0
            
            # Select features in correct order and fill any remaining NaN
            X_pred = pred_df[self.feature_names].fillna(0)
            logger.info(f"   ✅ Final feature matrix: {X_pred.shape}")
            
            # Debug: Log meal period features to verify differentiation
            meal_features = ['meal_period_encoded', 'meal_hour', 'meal_period_encoded_sin', 
                           'meal_period_encoded_cos', 'meal_hour_sin', 'meal_hour_cos',
                           'meal_dow_interaction', 'meal_month_interaction']
            available_meal_features = [f for f in meal_features if f in X_pred.columns]
            if available_meal_features:
                logger.info(f"   🍽️ Meal features for {meal_period}:")
                for feature in available_meal_features:
                    value = X_pred[feature].iloc[0] if feature in X_pred.columns else 'MISSING'
                    logger.info(f"      {feature}: {value}")
            else:
                logger.warning(f"   ⚠️ No meal period features found in prediction data!")
            
            predictions = {}
            
            # Get predictions from all models
            logger.info(f"   🤖 Getting predictions from {len(self.models)} models...")
            for model_name, model in self.models.items():
                try:
                    if model_name == 'ridge':
                        # Use scaled features for ridge
                        X_pred_scaled = self.scalers['standard'].transform(X_pred)
                        pred = model.predict(X_pred_scaled)[0]
                    else:
                        pred = model.predict(X_pred)[0]
                    # Use actual model prediction without modification
                    predictions[model_name] = float(pred)
                    logger.info(f"      {model_name}: ${pred:.2f}")
                except Exception as model_error:
                    logger.error(f"      ❌ {model_name} failed: {str(model_error)}")
                    # Skip failed models rather than using dummy data
                    continue
            
            # Calculate ensemble predictions if available
            ensemble_predictions = {}
            if len(predictions) > 1:  # Only create ensemble if we have multiple predictions
                # Simple average - use actual model outputs
                simple_avg = np.mean(list(predictions.values()))
                ensemble_predictions['simple_average'] = float(simple_avg)
                
                # Weighted average if weights are available
                if hasattr(self, 'ensemble_weights') and self.ensemble_weights:
                    try:
                        weighted_avg = np.average(list(predictions.values()), 
                                                weights=list(self.ensemble_weights.values()))
                        ensemble_predictions['weighted_average'] = float(weighted_avg)
                    except:
                        # Fallback to simple average if weighting fails
                        ensemble_predictions['weighted_average'] = ensemble_predictions['simple_average']
            
            # Determine best prediction using model performance, not prediction value
            if model_type == 'best_ensemble' and ensemble_predictions:
                best_pred = ensemble_predictions.get('weighted_average') or ensemble_predictions.get('simple_average')
                best_model = 'ensemble'
            else:
                # Use best individual model based on validation performance
                if hasattr(self, 'results') and 'individual_models' in self.results:
                    # Get best model by R² score
                    best_model_name = max(self.results['individual_models'].items(), 
                                        key=lambda x: x[1].get('r2', 0))[0]
                    best_pred = predictions.get(best_model_name, predictions.get('xgboost', 0))
                    best_model = best_model_name
                else:
                    # Fallback to XGBoost or first available model
                    if 'xgboost' in predictions:
                        best_pred = predictions['xgboost']
                        best_model = 'xgboost'
                    else:
                        best_model, best_pred = next(iter(predictions.items()))
            
            # Use actual model prediction without any fallback or modification
            logger.info(f"   🎯 Final prediction: ${best_pred:.2f} from {best_model}")
            
            return {
                'success': True,
                'date': date,
                'meal_period': meal_period,
                'predicted_revenue': round(best_pred, 2),
                'best_model': best_model,
                'individual_predictions': predictions,
                'ensemble_predictions': ensemble_predictions,
                'islamic_period': pred_df['IslamicPeriod'].iloc[0] if 'IslamicPeriod' in pred_df.columns else 'Normal',
                'tourism_intensity': pred_df['TourismIntensity'].iloc[0] if 'TourismIntensity' in pred_df.columns else 'Normal'
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def predict_90_days(self, start_date=None):
        """Generate 90-day revenue forecast for all meal periods"""
        
        if not self.models:
            return {'error': 'Models not trained yet. Please train models first.'}
        
        try:
            # Default to tomorrow if no start date provided
            if start_date is None:
                start_date = datetime.now().date() + timedelta(days=1)
            else:
                start_date = pd.to_datetime(start_date).date()
            
            # Generate date range for 90 days
            date_range = [start_date + timedelta(days=i) for i in range(90)]
            meal_periods = ['Breakfast', 'Lunch', 'Dinner']
            
            forecast_results = []
            
            for date in date_range:
                for meal_period in meal_periods:
                    # Get prediction for this date/meal combination
                    prediction = self.predict_single(date.strftime('%Y-%m-%d'), meal_period)
                    
                    if prediction.get('success'):
                        forecast_results.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'meal_period': meal_period,
                            'predicted_revenue': prediction['predicted_revenue'],
                            'islamic_period': prediction.get('islamic_period', 'Normal'),
                            'tourism_intensity': prediction.get('tourism_intensity', 'Normal'),
                            'day_of_week': date.strftime('%A'),
                            'week_number': int(date.strftime('%U'))
                        })
            
            # Calculate summary statistics
            total_forecast = sum(item['predicted_revenue'] for item in forecast_results)
            daily_averages = {}
            
            # Group by meal period for analysis
            for meal in meal_periods:
                meal_predictions = [item['predicted_revenue'] for item in forecast_results 
                                 if item['meal_period'] == meal]
                
                # Handle empty meal predictions gracefully
                if len(meal_predictions) > 0:
                    daily_averages[meal] = {
                        'average': round(np.mean(meal_predictions), 2),
                        'min': round(min(meal_predictions), 2),
                        'max': round(max(meal_predictions), 2),
                        'std': round(np.std(meal_predictions), 2),
                        'count': len(meal_predictions)
                    }
                else:
                    # No successful predictions for this meal period
                    daily_averages[meal] = {
                        'average': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'std': 0.0,
                        'count': 0,
                        'error': 'No successful predictions for this meal period'
                    }
            
            return {
                'success': True,
                'forecast_period': f"{start_date} to {(start_date + timedelta(days=89))}",
                'total_predictions': len(forecast_results),
                'total_forecast_revenue': round(total_forecast, 2),
                'daily_average_revenue': round(total_forecast / 90, 2),
                'meal_period_statistics': daily_averages,
                'detailed_forecast': forecast_results
            }
            
        except Exception as e:
            return {'error': f'90-day forecast failed: {str(e)}'}
    
    def create_accuracy_plots(self):
        """Create model accuracy visualization plots"""
        
        if not self.models or 'individual_models' not in self.results:
            return {'error': 'Models not trained yet. Please train models first.'}
        
        try:
            # Use test data for accuracy evaluation
            if self.splits is None:
                return {'error': 'No test data available for accuracy evaluation'}
            
            dataset = self.engineer_features(self.splits)
            X_test, y_test = dataset['X_test'], dataset['y_test']
            
            # Get predictions from all models
            model_predictions = {}
            
            for model_name, model in self.models.items():
                if model_name == 'ridge':
                    X_test_scaled = self.scalers['standard'].transform(X_test)
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                model_predictions[model_name] = pred
            
            # Create plots with better error handling
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Smaller figure size
            fig.suptitle('Model Accuracy Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1-5: Predictions vs Actual for each model
            for i, (model_name, predictions) in enumerate(model_predictions.items()):
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # Create scatter plot
                ax.scatter(y_test, predictions, alpha=0.6, s=30, c='blue')
                
                # Add perfect prediction line
                min_val = min(y_test.min(), predictions.min())
                max_val = max(y_test.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                
                ax.set_xlabel('Actual Revenue ($)')
                ax.set_ylabel('Predicted Revenue ($)')
                ax.set_title(f'{model_name.replace("_", " ").title()} Model')
                ax.grid(True, alpha=0.3)
                
                # Add R² score
                r2 = r2_score(y_test, predictions)
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10, fontweight='bold')
            
            # Plot 6: Model comparison (if we have a 6th position)
            if len(model_predictions) <= 5:
                # Use the last position for model comparison
                ax = axes[1, 2]
                model_names = [name.replace('_', ' ').title() for name in model_predictions.keys()]
                r2_scores = [r2_score(y_test, model_predictions[name]) for name in model_predictions.keys()]
                
                # Create bar chart
                bars = ax.bar(range(len(model_names)), r2_scores, alpha=0.7, color=['blue', 'green', 'red', 'purple', 'orange'][:len(model_names)])
                
                ax.set_xlabel('Models')
                ax.set_ylabel('R² Score')
                ax.set_title('Model Performance Comparison')
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, score) in enumerate(zip(bars, r2_scores)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Initialize plot_data
            plot_data = None
            
            # Convert plot to base64 string with error handling
            try:
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
                plt.close(fig)  # Close the specific figure
                plt.clf()       # Clear the current figure
                logger.info(f"🖼️ Plot image generated successfully, size: {len(plot_data)} characters")
                
            except Exception as plot_error:
                plt.close('all')  # Close all figures in case of error
                logger.error(f"Plot generation error: {str(plot_error)}")
                return {'error': f'Plot generation failed: {str(plot_error)}'}
            
            # Verify plot_data was created
            if plot_data is None or len(plot_data) == 0:
                logger.error("❌ Plot data is empty or None")
                return {'error': 'Plot image data is empty'}
            
            # Calculate detailed metrics
            metrics_summary = {}
            for model_name, predictions in model_predictions.items():
                # Calculate MAPE safely (avoid division by zero)
                # Only calculate MAPE for non-zero actual values
                non_zero_mask = y_test != 0
                if non_zero_mask.sum() > 0:
                    mape_values = np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])
                    mape = float(np.mean(mape_values) * 100)
                else:
                    mape = 0.0  # If all actual values are zero, set MAPE to 0
                
                # Ensure MAPE is finite
                if not np.isfinite(mape):
                    mape = 0.0
                
                metrics_summary[model_name] = {
                    'r2_score': float(r2_score(y_test, predictions)),
                    'mae': float(mean_absolute_error(y_test, predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                    'mape': mape
                }
            
            logger.info(f"📊 Returning response with plot_image length: {len(plot_data)}")
            
            return {
                'success': True,
                'plot_image': plot_data,
                'metrics_summary': metrics_summary,
                'test_samples': len(y_test),
                'best_model': max(metrics_summary.items(), key=lambda x: x[1]['r2_score'])[0]
            }
            
        except Exception as e:
            return {'error': f'Accuracy plot generation failed: {str(e)}'}
    
    def create_time_series_plots(self):
        """Create time series plots showing actual vs predicted values over time"""
        
        if not self.models or 'individual_models' not in self.results:
            return {'error': 'Models not trained yet. Please train models first.'}
        
        try:
            logger.info("📈 Generating time series actual vs predicted plots...")
            
            # Use test data for time series visualization
            if self.splits is None:
                return {'error': 'No test data available for time series evaluation'}
            
            dataset = self.engineer_features(self.splits)
            X_test, y_test = dataset['X_test'], dataset['y_test']
            
            # Get dates for the test period
            test_data = self.splits['test']
            test_dates = test_data['Date'].reset_index(drop=True)
            test_meal_periods = test_data['MealPeriod'].reset_index(drop=True)
            
            # Get predictions from all models
            model_predictions = {}
            
            for model_name, model in self.models.items():
                if model_name == 'ridge':
                    X_test_scaled = self.scalers['standard'].transform(X_test)
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                model_predictions[model_name] = pred
            
            # Create comprehensive time series plots - overall + individual meal periods
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Time Series: Actual vs Predicted Revenue Analysis', fontsize=16, fontweight='bold')
            
            # Get best model
            best_model_name = max(self.results['individual_models'].items(), 
                                key=lambda x: x[1]['r2'])[0]
            
            # Sort by date for proper time series display
            time_data = pd.DataFrame({
                'Date': test_dates,
                'Actual': y_test.values,
                'Predicted': model_predictions[best_model_name],
                'MealPeriod': test_meal_periods
            }).sort_values('Date')
            
            # Plot 1: Overall Time Series (All Meal Periods Combined)
            ax1 = axes[0, 0]
            ax1.plot(time_data['Date'], time_data['Actual'], 'b-', 
                    label='Actual Revenue', linewidth=2, alpha=0.8)
            ax1.plot(time_data['Date'], time_data['Predicted'], 'r--', 
                    label='Predicted Revenue', linewidth=2, alpha=0.8)
            ax1.set_title(f'Overall: Actual vs Predicted ({best_model_name.replace("_", " ").title()})')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Revenue ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Breakfast Time Series
            ax2 = axes[0, 1]
            breakfast_data = time_data[time_data['MealPeriod'] == 'Breakfast']
            if len(breakfast_data) > 0:
                ax2.plot(breakfast_data['Date'], breakfast_data['Actual'], 'b-', 
                        label='Actual Breakfast', linewidth=2, alpha=0.8)
                ax2.plot(breakfast_data['Date'], breakfast_data['Predicted'], 'r--', 
                        label='Predicted Breakfast', linewidth=2, alpha=0.8)
            ax2.set_title('Breakfast: Predicted vs Actual')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Revenue ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Lunch Time Series
            ax3 = axes[0, 2]
            lunch_data = time_data[time_data['MealPeriod'] == 'Lunch']
            if len(lunch_data) > 0:
                ax3.plot(lunch_data['Date'], lunch_data['Actual'], 'g-', 
                        label='Actual Lunch', linewidth=2, alpha=0.8)
                ax3.plot(lunch_data['Date'], lunch_data['Predicted'], 'orange', 
                        linestyle='--', label='Predicted Lunch', linewidth=2, alpha=0.8)
            ax3.set_title('Lunch: Predicted vs Actual')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Revenue ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Dinner Time Series
            ax4 = axes[1, 0]
            dinner_data = time_data[time_data['MealPeriod'] == 'Dinner']
            if len(dinner_data) > 0:
                ax4.plot(dinner_data['Date'], dinner_data['Actual'], 'purple', 
                        label='Actual Dinner', linewidth=2, alpha=0.8)
                ax4.plot(dinner_data['Date'], dinner_data['Predicted'], 'red', 
                        linestyle='--', label='Predicted Dinner', linewidth=2, alpha=0.8)
            ax4.set_title('Dinner: Predicted vs Actual')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Revenue ($)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
            
            # Plot 5: Performance Summary by Meal Period
            ax5 = axes[1, 1]
            meal_periods = ['Breakfast', 'Lunch', 'Dinner']
            colors = ['blue', 'green', 'purple']
            
            # Calculate R² for each meal period
            meal_r2_scores = []
            for meal in meal_periods:
                meal_mask = time_data['MealPeriod'] == meal
                if meal_mask.sum() > 0:
                    actual_meal = time_data[meal_mask]['Actual']
                    predicted_meal = time_data[meal_mask]['Predicted']
                    r2 = r2_score(actual_meal, predicted_meal)
                    meal_r2_scores.append(max(0, r2))  # Ensure non-negative for display
                else:
                    meal_r2_scores.append(0)
            
            bars = ax5.bar(meal_periods, meal_r2_scores, color=colors, alpha=0.7)
            ax5.set_title('Model Performance by Meal Period (R² Score)')
            ax5.set_xlabel('Meal Period')
            ax5.set_ylabel('R² Score')
            ax5.set_ylim(0, max(1, max(meal_r2_scores) * 1.1))
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, score in zip(bars, meal_r2_scores):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            # Plot 6: Meal Period Comparison (Combined View)
            ax6 = axes[1, 2]
            meal_periods = ['Breakfast', 'Lunch', 'Dinner']
            colors = ['blue', 'green', 'purple']
            
            for i, meal in enumerate(meal_periods):
                meal_mask = time_data['MealPeriod'] == meal
                meal_data = time_data[meal_mask].head(20)  # Show first 20 days for clarity
                
                if len(meal_data) > 0:
                    ax6.plot(meal_data['Date'], meal_data['Actual'], 
                            color=colors[i], linestyle='-', label=f'{meal} Actual', alpha=0.7)
                    ax6.plot(meal_data['Date'], meal_data['Predicted'], 
                            color=colors[i], linestyle='--', label=f'{meal} Predicted', alpha=0.7)
            
            ax6.set_title('All Meal Periods Comparison (First 20 Days)')
            ax6.set_xlabel('Date')
            ax6.set_ylabel('Revenue ($)')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            plot_data = None
            try:
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
                plt.close(fig)
                plt.clf()
                logger.info(f"📈 Time series plot generated successfully, size: {len(plot_data)} characters")
                
            except Exception as plot_error:
                plt.close('all')
                logger.error(f"Time series plot generation error: {str(plot_error)}")
                return {'error': f'Time series plot generation failed: {str(plot_error)}'}
            
            # Verify plot_data was created
            if plot_data is None or len(plot_data) == 0:
                logger.error("❌ Time series plot data is empty")
                return {'error': 'Time series plot image data is empty'}
            
            # Calculate time series metrics
            best_predictions = model_predictions[best_model_name]
            time_series_metrics = {
                'best_model': best_model_name,
                'r2_score': float(r2_score(y_test, best_predictions)),
                'mae': float(mean_absolute_error(y_test, best_predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, best_predictions))),
                'mean_actual': float(np.mean(y_test)),
                'mean_predicted': float(np.mean(best_predictions)),
                'test_period': f"{test_dates.min()} to {test_dates.max()}",
                'total_samples': len(y_test)
            }
            
            logger.info(f"📊 Returning time series response with plot_image length: {len(plot_data)}")
            
            return {
                'success': True,
                'plot_image': plot_data,
                'metrics': time_series_metrics,
                'test_samples': len(y_test),
                'date_range': time_series_metrics['test_period']
            }
            
        except Exception as e:
            logger.error(f"❌ Time series plot generation failed: {str(e)}")
            return {'error': f'Time series plot generation failed: {str(e)}'} 
    
    def _apply_prediction_features(self, df):
        """Apply feature engineering to prediction data (no training, just feature creation)"""
        
        df = df.copy()
        
        # Apply temporal features similar to training but simplified
        df['day'] = df['Date'].dt.day.astype('int32')
        df['day_of_year'] = df['Date'].dt.dayofyear.astype('int32')
        df['week_of_year'] = df['Date'].dt.isocalendar().week.astype('int32')
        df['quarter'] = df['Date'].dt.quarter.astype('int32')
        df['is_weekend'] = df['DayOfWeek'].isin([5, 6]).astype('int32')
        df['is_month_start'] = df['Date'].dt.is_month_start.astype('int32')
        df['is_month_end'] = df['Date'].dt.is_month_end.astype('int32')
        
        # Cyclical encoding for temporal features - MUST match training exactly
        for col, max_val in [('Month', 12), ('DayOfWeek', 7), ('quarter', 4), ('meal_period_encoded', 3), ('meal_hour', 24)]:
            if col in df.columns:
                df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
                df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        
        # Critical: Add meal period interaction features for differentiation
        df['meal_dow_interaction'] = df['meal_period_encoded'] * df['DayOfWeek']
        df['meal_month_interaction'] = df['meal_period_encoded'] * df['Month']
        df['meal_year_interaction'] = df['meal_period_encoded'] * df['Year']
        df['meal_weekend_interaction'] = df['meal_period_encoded'] * df['is_weekend']
        
        return df