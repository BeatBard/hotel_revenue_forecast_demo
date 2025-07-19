"""
Data Loader Utility
==================

Handles loading and preparing real revenue data for the demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.revenue_data = None
        self.data_info = {}
        
    def load_revenue_data(self, data_path=None):
        """Load real hotel revenue data from CSV file"""
        
        # Define the correct data path where user copied the files
        if not data_path:
            data_path = 'revenue_center_data/RevenueCenter_1_data.csv'
        
        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"❌ Could not find data file at: {data_path}")
            logger.warning("⚠️  Real data file not found, creating sample data")
            return self._create_fallback_data()
        
        try:
            logger.info(f"📂 Loading revenue data from: {data_path}")
            
            # Load the CSV data
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Data exploration and quality checks (like original script)
            logger.info("🔍 PERFORMING DATA EXPLORATION")
            logger.info(f"   Dataset Shape: {df.shape}")
            logger.info(f"   Date Range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"   Total Days: {(df['Date'].max() - df['Date'].min()).days}")
            logger.info(f"   Revenue Range: ${df['CheckTotal'].min():.2f} - ${df['CheckTotal'].max():.2f}")
            logger.info(f"   Average Revenue: ${df['CheckTotal'].mean():.2f}")
            
            # Data quality checks
            logger.info("🔍 DATA QUALITY CHECKS:")
            logger.info(f"   Missing values: {df.isnull().sum().sum()}")
            logger.info(f"   Duplicates: {df.duplicated(['Date', 'MealPeriod']).sum()}")
            logger.info(f"   Zero revenue records: {(df['CheckTotal'] == 0).sum()}")
            
            # Revenue by meal period
            logger.info("📊 REVENUE BY MEAL PERIOD:")
            revenue_stats = df.groupby('MealPeriod')['CheckTotal'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            logger.info(f"\n{revenue_stats}")
            
            # Store data information
            self.data_info = {
                'dataset_shape': df.shape,
                'date_range': {
                    'start': df['Date'].min(),
                    'end': df['Date'].max(),
                    'total_days': (df['Date'].max() - df['Date'].min()).days
                },
                'revenue_stats': {
                    'min': float(df['CheckTotal'].min()),
                    'max': float(df['CheckTotal'].max()),
                    'mean': float(df['CheckTotal'].mean()),
                    'median': float(df['CheckTotal'].median()),
                    'std': float(df['CheckTotal'].std())
                },
                'data_quality': {
                    'missing_values': int(df.isnull().sum().sum()),
                    'duplicates': int(df.duplicated(['Date', 'MealPeriod']).sum()),
                    'zero_revenues': int((df['CheckTotal'] == 0).sum())
                },
                'meal_period_stats': revenue_stats.to_dict('index')
            }
            
            self.revenue_data = df
            logger.info("✅ Revenue data loaded successfully!")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            logger.warning("⚠️  Creating fallback sample data")
            return self._create_fallback_data()
    
    def _create_fallback_data(self):
        """Create fallback sample data if real data is not available"""
        logger.info("🔄 Creating fallback sample data...")
        
        # Create realistic sample data structure matching the original
        np.random.seed(42)
        
        # Date range (6 months of data)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 30)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Create data for each meal period
        meal_periods = ['Breakfast', 'Lunch', 'Dinner']
        data_rows = []
        
        for date in dates:
            for meal_period in meal_periods:
                # Generate realistic revenue patterns
                base_revenue = self._generate_base_revenue(date, meal_period)
                seasonal_factor = self._get_seasonal_factor(date)
                day_factor = self._get_day_factor(date)
                meal_factor = self._get_meal_factor(meal_period)
                
                # Final revenue with noise
                revenue = base_revenue * seasonal_factor * day_factor * meal_factor
                revenue = max(0, revenue + np.random.normal(0, revenue * 0.1))
                
                data_rows.append({
                    'Date': date,
                    'MealPeriod': meal_period,
                    'CheckTotal': round(revenue, 2),
                    'RevenueCenterName': 'RevenueCenter_1'
                })
        
        self.sample_data = pd.DataFrame(data_rows)
        return self.sample_data
    
    def _generate_base_revenue(self, date, meal_period):
        """Generate base revenue for a given date and meal period"""
        base_revenues = {
            'Breakfast': 800,
            'Lunch': 1200,
            'Dinner': 2000
        }
        return base_revenues[meal_period]
    
    def _get_seasonal_factor(self, date):
        """Get seasonal multiplier based on date"""
        month = date.month
        
        # Higher in winter months, lower in summer
        seasonal_factors = {
            1: 1.2, 2: 1.1, 3: 1.15, 4: 1.0, 5: 0.9, 6: 0.8,
            7: 0.75, 8: 0.8, 9: 0.95, 10: 1.05, 11: 1.1, 12: 1.25
        }
        
        return seasonal_factors[month]
    
    def _get_day_factor(self, date):
        """Get day-of-week multiplier"""
        weekday = date.weekday()
        
        # Monday=0, Sunday=6
        # Higher on weekends for some meals, weekdays for business
        if weekday < 5:  # Weekday
            return 1.1
        else:  # Weekend
            return 0.9
    
    def _get_meal_factor(self, meal_period):
        """Get meal-specific variance factor"""
        factors = {
            'Breakfast': 0.8,  # More consistent
            'Lunch': 1.0,     # Moderate variance
            'Dinner': 1.2     # Higher variance
        }
        return factors[meal_period]
    
    def get_data_summary(self):
        """Get summary statistics of the loaded data"""
        if self.sample_data is None:
            return None
            
        return {
            'total_records': len(self.sample_data),
            'date_range': {
                'start': self.sample_data['Date'].min().strftime('%Y-%m-%d'),
                'end': self.sample_data['Date'].max().strftime('%Y-%m-%d')
            },
            'revenue_stats': {
                'total': self.sample_data['CheckTotal'].sum(),
                'mean': self.sample_data['CheckTotal'].mean(),
                'median': self.sample_data['CheckTotal'].median(),
                'std': self.sample_data['CheckTotal'].std()
            },
            'meal_periods': list(self.sample_data['MealPeriod'].unique()),
                         'revenue_centers': list(
                 self.sample_data['RevenueCenterName'].unique())
        } 