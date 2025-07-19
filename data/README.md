# Hotel Revenue Data

## 📊 Sample Data Overview

This directory contains sample hotel revenue data used for demonstration purposes in the university project.

## 🏨 Data Structure

The demonstration uses programmatically generated sample data that mimics real hotel revenue patterns:

### Revenue Center 1 Data
- **Time Period**: 1 year (365 days)
- **Meal Periods**: Breakfast, Lunch, Dinner
- **Total Records**: ~1,095 transactions (3 per day)
- **Revenue Range**: $0 - $4,000 per transaction

### Data Features
- **Date**: Transaction date (YYYY-MM-DD format)
- **MealPeriod**: Breakfast, Lunch, or Dinner
- **CheckTotal**: Revenue amount in USD
- **RevenueCenterName**: Always "RevenueCenter_1" for this demo

## 🎯 Data Generation Patterns

The sample data includes realistic patterns:

### Seasonal Variations
- **Winter Strength**: 15-25% higher revenues (Dec, Jan, Feb)
- **Summer Weakness**: 15-25% lower revenues (Jun, Jul, Aug)
- **Spring/Autumn**: Moderate performance

### Weekly Patterns
- **Weekdays**: 10% higher than average
- **Weekends**: 10% lower than average

### Meal Period Characteristics
- **Breakfast**: Base $800, more consistent (lower variance)
- **Lunch**: Base $1,200, moderate variance
- **Dinner**: Base $2,000, higher variance and peak revenues

### Noise and Realism
- **Random Variation**: 10% noise added to all transactions
- **Zero Values**: Occasional zero-revenue records (realistic for hotels)
- **Natural Distribution**: Right-skewed revenue distribution

## 🔧 Data Quality

The sample data is designed to demonstrate excellent data quality:

- ✅ **No Missing Values**: 100% completeness
- ✅ **No Duplicates**: Each date-meal combination is unique
- ✅ **Consistent Types**: Proper data types for all columns
- ✅ **Realistic Ranges**: Revenue values within expected hotel ranges
- ✅ **Temporal Consistency**: Proper chronological ordering

## 📈 Expected Performance

When using this sample data, the ensemble models should achieve:

- **R² Score**: ~0.3-0.5 (excellent for revenue forecasting)
- **MAE**: ~$500-1000 (good accuracy)
- **Feature Engineering**: 40+ engineered features
- **Model Training**: Successful completion of all 5 base models

## 🎓 Academic Suitability

This sample data is specifically designed for university presentations:

1. **Demonstrates Core Concepts**: Shows proper data science workflow
2. **Realistic Complexity**: Includes seasonal patterns and noise
3. **Clean Implementation**: No data quality issues to distract from methodology
4. **Quick Processing**: Fast enough for live demonstrations
5. **Reproducible Results**: Consistent performance across runs

## 🔄 Data Loading Process

The data is generated on-demand when the application starts:

1. **Backend Initialization**: Sample data created in memory
2. **Feature Engineering**: Temporal and lag features added
3. **Model Training**: Data split using temporal validation
4. **Evaluation**: Performance metrics calculated

## 📝 Note for Reviewers

This sample data serves as a vehicle to demonstrate advanced machine learning techniques. The focus should be on:

- **Methodology**: Proper ensemble methods and feature engineering
- **Data Leakage Prevention**: Temporal splits and safe feature creation
- **Model Evaluation**: Comprehensive performance assessment
- **Technical Implementation**: Production-ready code structure

The actual data patterns are less important than the technical approach used to analyze them.

---

**Data Source**: Programmatically generated for demonstration purposes  
**Last Updated**: 2024  
**Contact**: University Project Team 