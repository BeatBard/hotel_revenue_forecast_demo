#!/usr/bin/env python3
"""
Hotel Revenue Forecasting Demo - Flask Backend
=============================================

University project demonstration backend with endpoints for:
- EDA analysis and visualizations
- Feature engineering demonstrations
- Model training and ensemble methods
- Performance evaluation and predictions

Author: University Project Demo
"""

import warnings
import logging
import sys
import os
import base64
import io
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

from flask import Flask, request, jsonify
from flask_cors import CORS

# Set matplotlib backend before other imports
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Import our custom model implementations
from models.ensemble_model import HotelRevenueEnsemble
from models.feature_engineering import FeatureEngineer
from utils.data_loader import DataLoader

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for storing models and data
ensemble_model = None
feature_engineer = None
data_loader = None
current_data = None

# Helper function to convert matplotlib plots to base64
def plot_to_base64():
    """Convert current matplotlib plot to base64 encoded string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_data = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(plot_data).decode()

# Helper function to load all revenue center data
def load_all_revenue_data():
    """Load data from all revenue centers and combine"""
    data_frames = []
    data_dir = 'revenue_center_data'
    
    for i in range(1, 10):  # RevenueCenter_1 to RevenueCenter_9
        file_path = f"{data_dir}/RevenueCenter_{i}_data.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['RevenueCenterID'] = i
            data_frames.append(df)
    
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        return combined_df
    return None

def init_app():
    """Initialize the application with required components"""
    global ensemble_model, feature_engineer, data_loader
    
    logger.info("🚀 Initializing Hotel Revenue Demo Backend...")
    


    
    logger.info("🔧 Loading Feature Engineer...")
    feature_engineer = FeatureEngineer()
    
    logger.info("🤖 Loading Ensemble Model Framework...")
    ensemble_model = HotelRevenueEnsemble()
    
    logger.info("📂 Loading Data Loader...")
    data_loader = DataLoader()
    
    print("\n" + "="*60)
    print("🚀 Hotel Revenue Demo Backend Started Successfully!")
    print("="*60)
    print("📊 EDA Analysis: ✅ Ready")
    print("🔧 Feature Engineering: ✅ Ready") 
    print("🤖 Ensemble Models: ✅ Ready")
    print("📈 Performance Evaluation: ✅ Ready")
    print("🌐 API Server: ✅ Running on http://localhost:5000")
    print("="*60)
    logger.info("✅ Backend initialization complete!")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Hotel Revenue Demo Backend is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load real hotel revenue data from CSV files"""
    try:
        global current_data
        
        logger.info("📂 Starting data loading process...")
        logger.info("   ⏳ Loading real hotel revenue data...")
        
        # Load real revenue data from CSV
        current_data = data_loader.load_revenue_data()
        
        # Perform data exploration and create temporal splits
        explored_data = ensemble_model.load_and_explore_data(current_data)
        splits = ensemble_model.create_temporal_splits(explored_data)
        
        logger.info(f"   ✅ Data loaded successfully! Shape: {current_data.shape}")
        logger.info(f"   📅 Date range: {current_data['Date'].min()} to {current_data['Date'].max()}")
        logger.info(f"   💰 Revenue range: ${current_data['CheckTotal'].min():.2f} - ${current_data['CheckTotal'].max():.2f}")
        logger.info(f"   📊 Average revenue: ${current_data['CheckTotal'].mean():.2f}")
        
        # Store data for other operations
        ensemble_model.training_data = explored_data
        
        response = {
            'success': True,
            'message': 'Real revenue data loaded successfully',
            'data_info': {
                'shape': current_data.shape,
                'columns': list(current_data.columns),
                'date_range': {
                    'start': current_data['Date'].min().isoformat(),
                    'end': current_data['Date'].max().isoformat()
                },
                'revenue_stats': {
                    'mean': float(current_data['CheckTotal'].mean()),
                    'median': float(current_data['CheckTotal'].median()),
                    'std': float(current_data['CheckTotal'].std()),
                    'min': float(current_data['CheckTotal'].min()),
                    'max': float(current_data['CheckTotal'].max())
                },
                'splits_info': {
                    'train_samples': len(splits['train']),
                    'validation_samples': len(splits['validation']),
                    'test_samples': len(splits['test'])
                }
            }
        }
        
        logger.info("✅ Data loading completed successfully!")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# EDA overview route removed - functionality integrated into model training

# EDA analysis routes removed - core functionality available through model training and predictions

@app.route('/api/feature-engineering/create-features', methods=['POST'])
def create_features():
    """Demonstrate feature engineering process"""
    try:
        if current_data is None:
            logger.warning("⚠️ Feature engineering requested but no data loaded")
            return jsonify({'error': 'No data loaded'}), 400
            
        config = request.get_json() or {}
        
        logger.info("🔧 Starting temporal feature engineering...")
        logger.info("   📅 Creating temporal features (Year, Month, Day, etc.)...")
        logger.info("   🔄 Applying cyclical encoding (sin/cos transforms)...")
        logger.info("   🏷️ Encoding categorical variables...")
        logger.info("   ⚙️ Creating interaction features...")
        
        features_info = feature_engineer.create_temporal_features(
            current_data, 
            config
        )
        
        logger.info(f"✅ Created {features_info['feature_count']} temporal features successfully!")
        return jsonify(features_info)
        
    except Exception as e:
        logger.error(f"❌ Error creating features: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-engineering/lag-features', methods=['POST'])
def create_lag_features():
    """Create and demonstrate lag features"""
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
            
        config = request.get_json() or {}
        lags = config.get('lags', [1, 7, 14, 30])
        
        lag_info = feature_engineer.create_lag_features(current_data, lags)
        
        return jsonify(lag_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-engineering/rolling-features', methods=['POST'])
def create_rolling_features():
    """Create and demonstrate rolling window features"""
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
            
        config = request.get_json() or {}
        windows = config.get('windows', [7, 14, 30])
        
        rolling_info = feature_engineer.create_rolling_features(current_data, windows)
        
        return jsonify(rolling_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-engineering/leakage-prevention', methods=['GET'])
def demonstrate_leakage_prevention():
    """Demonstrate data leakage prevention techniques"""
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
            
        leakage_demo = feature_engineer.demonstrate_leakage_prevention(current_data)
        
        return jsonify(leakage_demo)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feature-engineering/feature-importance', methods=['GET'])
def analyze_feature_importance_for_revenue():
    """Analyze feature importance for predicting CheckTotal using real data"""
    try:
        logger.info("🎯 Starting feature importance analysis for CheckTotal prediction...")
        
        # Load all revenue center data
        df = load_all_revenue_data()
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        logger.info(f"   📊 Loaded {df.shape[0]} records from all revenue centers")
        
        # Prepare features for importance analysis
        
        # Create a copy for feature engineering
        feature_df = df.copy()
        
        # Remove target column and non-predictive columns
        target = 'CheckTotal'
        exclude_cols = ['Date', 'RevenueCenterName', target, 'is_zero']
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
        
        # Encode categorical variables
        label_encoders = {}
        for col in feature_cols:
            if feature_df[col].dtype == 'object':
                le = LabelEncoder()
                feature_df[col] = le.fit_transform(feature_df[col].astype(str))
                label_encoders[col] = le
        
        # Prepare features and target
        X = feature_df[feature_cols]
        y = feature_df[target]
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        y = y.fillna(0)
        
        logger.info(f"   🔧 Using {len(feature_cols)} features to predict revenue")
        
        # Train Random Forest for feature importance
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            n_jobs=-1
        )
        
        rf_model.fit(X, y)
        
        # Get feature importance scores
        feature_importance = rf_model.feature_importances_
        
        # Create feature importance dictionary
        importance_dict = {}
        for i, col in enumerate(feature_cols):
            importance_dict[col] = float(feature_importance[i])
        
        # Sort by importance (descending)
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Create visualization data for top features
        top_features = dict(list(sorted_importance.items())[:15])  # Top 15 features
        
        # Create importance plot
        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(features)), importances, color='steelblue', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance (Random Forest)')
        plt.title('Feature Importance for Predicting Revenue (CheckTotal)')
        plt.gca().invert_yaxis()  # Highest importance at top
        
        # Add importance values on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        importance_plot = plot_to_base64()
        
        # Calculate model performance metrics
        y_pred = rf_model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Categorize features by type for better understanding
        feature_categories = {
            'temporal': [],
            'events': [],
            'operational': [],
            'location': []
        }
        
        for feature in feature_cols:
            if any(word in feature.lower() for word in ['month', 'year', 'day', 'date']):
                feature_categories['temporal'].append(feature)
            elif feature.startswith('Is') or 'event' in feature.lower():
                feature_categories['events'].append(feature)
            elif 'center' in feature.lower() or 'meal' in feature.lower():
                feature_categories['operational'].append(feature)
            else:
                feature_categories['location'].append(feature)
        
        logger.info(f"✅ Feature importance analysis completed")
        logger.info(f"   🎯 Model R² score: {r2:.3f}")
        logger.info(f"   📊 Top feature: {list(sorted_importance.keys())[0]} ({list(sorted_importance.values())[0]:.3f})")
        
        return jsonify({
            'success': True,
            'feature_importance_plot': importance_plot,
            'feature_importance': sorted_importance,
            'top_features': top_features,
            'model_performance': {
                'r2_score': float(r2),
                'mae': float(mae),
                'features_count': len(feature_cols),
                'data_points': len(X)
            },
            'feature_categories': feature_categories,
            'analysis_summary': {
                'total_features': len(feature_cols),
                'top_feature': list(sorted_importance.keys())[0],
                'top_importance': float(list(sorted_importance.values())[0]),
                'model_quality': 'Good' if r2 > 0.7 else 'Moderate' if r2 > 0.5 else 'Fair'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error in feature importance analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/individual-training', methods=['POST'])
def train_individual_models():
    """Train individual base models"""
    try:
        if current_data is None:
            logger.warning("⚠️ Model training requested but no data loaded")
            return jsonify({'error': 'No data loaded'}), 400
            
        config = request.get_json() or {}
        model_types = config.get('models', ['ridge', 'xgboost', 'lightgbm'])
        
        logger.info("🤖 Starting individual model training...")
        logger.info(f"   📋 Models to train: {', '.join(model_types)}")
        logger.info("   🔧 Preparing features and splitting data...")
        logger.info("   ⏱️ Beginning training process (this may take a moment)...")
        
        results = ensemble_model.train_individual_models(current_data, model_types)
        
        logger.info("✅ Individual model training completed!")
        logger.info(f"   📊 Trained {len(results['models_trained'])} models successfully")
        logger.info(f"   🎯 Used {results['features_used']} features")
        logger.info(f"   📈 Training samples: {results['training_samples']}")
        logger.info(f"   🧪 Validation samples: {results['validation_samples']}")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"❌ Error training individual models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/ensemble-training', methods=['POST'])
def train_ensemble_models():
    """Train ensemble models with different strategies"""
    try:
        if current_data is None:
            logger.warning("⚠️ Ensemble training requested but no data loaded")
            return jsonify({'error': 'No data loaded'}), 400
            
        config = request.get_json() or {}
        ensemble_types = config.get('ensembles', ['simple', 'weighted', 'top3'])
        
        logger.info("🎯 Starting ensemble model training...")
        logger.info(f"   📋 Ensemble strategies: {', '.join(ensemble_types)}")
        logger.info("   🔄 Generating predictions from individual models...")
        logger.info("   ⚖️ Calculating ensemble weights...")
        logger.info("   🧮 Creating ensemble combinations...")
        
        results = ensemble_model.train_ensemble(current_data, ensemble_types)
        
        logger.info("✅ Ensemble training completed!")
        logger.info(f"   🎯 Created {len(results['ensemble_strategies'])} ensemble strategies")
        logger.info(f"   🏆 Best ensemble: {results['best_ensemble'][0]} with R² = {results['best_ensemble'][1]['r2']:.3f}")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"❌ Error training ensemble models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/performance-comparison', methods=['GET'])
def model_performance_comparison():
    """Get comprehensive model performance comparison"""
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
            
        comparison = ensemble_model.compare_model_performance()
        
        return jsonify(comparison)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/predictions', methods=['POST'])
def generate_predictions():
    """Generate predictions using trained models"""
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
            
        config = request.get_json() or {}
        model_name = config.get('model', 'ensemble')
        
        predictions = ensemble_model.generate_predictions(current_data, model_name)
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from trained models"""
    try:
        model_name = request.args.get('model', 'xgboost')
        logger.info(f"🔍 Requesting feature importance for model: {model_name}")
        
        importance = ensemble_model.get_feature_importance(model_name)
        
        # Check if there's an error in the response
        if 'error' in importance:
            logger.warning(f"⚠️ Feature importance error: {importance['error']}")
            return jsonify(importance), 400
        
        logger.info(f"✅ Feature importance retrieved successfully for {model_name}")
        logger.info(f"   - Total features: {len(importance.get('features', []))}")
        logger.info(f"   - Top feature: {importance.get('sorted_features', [('N/A', 0)])[0][0]}")
        
        return jsonify(importance)
        
    except Exception as e:
        logger.error(f"❌ Error getting feature importance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/original-data-features', methods=['GET'])
def analyze_original_data_features():
    """Analyze feature importance on the original dataset"""
    try:
        logger.info("🔍 Analyzing original data features...")
        
        analysis = ensemble_model.analyze_original_data_features()
        
        # Check if there's an error in the response
        if 'error' in analysis:
            logger.warning(f"⚠️ Original data analysis error: {analysis['error']}")
            return jsonify(analysis), 400
        
        logger.info("✅ Original data feature analysis completed successfully")
        logger.info(f"   - Analyzed {len(analysis.get('feature_correlations', {}))} features")
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"❌ Error analyzing original data features: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluation/metrics', methods=['GET'])
def evaluation_metrics():
    """Get comprehensive evaluation metrics"""
    try:
        metrics = ensemble_model.get_evaluation_metrics()
        return jsonify(metrics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluation/visualizations', methods=['GET'])
def evaluation_visualizations():
    """Generate evaluation visualizations"""
    try:
        viz_type = request.args.get('type', 'predictions_vs_actual')
        
        visualization = ensemble_model.create_evaluation_visualization(viz_type)
        
        return jsonify(visualization)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demonstration/full-pipeline', methods=['POST'])
def run_full_demonstration():
    """Run the complete demonstration pipeline"""
    try:
        if current_data is None:
            logger.warning("⚠️ Full pipeline requested but no data loaded")
            return jsonify({'error': 'No data loaded'}), 400
            
        logger.info("🚀 Starting FULL DEMONSTRATION PIPELINE...")
        logger.info("="*60)
        
        # Step 1: Data Summary
        logger.info("📊 STEP 1: Generating Data Summary...")
        eda_results = {
            'total_records': len(current_data),
            'date_range': {
                'start': current_data['Date'].min().strftime('%Y-%m-%d'),
                'end': current_data['Date'].max().strftime('%Y-%m-%d')
            },
            'meal_periods': list(current_data['MealPeriod'].unique()),
            'revenue_stats': {
                'mean': float(current_data['CheckTotal'].mean()),
                'std': float(current_data['CheckTotal'].std()),
                'min': float(current_data['CheckTotal'].min()),
                'max': float(current_data['CheckTotal'].max())
            }
        }
        logger.info("   ✅ Data summary completed!")
        
        # Step 2: Feature Engineering
        logger.info("🔧 STEP 2: Engineering Features...")
        feature_results = feature_engineer.create_temporal_features(current_data, {})
        logger.info(f"   ✅ Created {feature_results['feature_count']} features!")
        
        # Step 3: Model Training
        logger.info("🤖 STEP 3: Training Individual Models...")
        model_results = ensemble_model.train_individual_models(current_data, 
                                                             ['ridge', 'xgboost', 'lightgbm'])
        logger.info(f"   ✅ Trained {len(model_results['models_trained'])} models!")
        
        # Step 4: Ensemble Creation
        logger.info("🎯 STEP 4: Creating Ensemble Strategies...")
        ensemble_results = ensemble_model.train_ensemble(current_data, ['simple', 'weighted'])
        logger.info("   ✅ Ensemble strategies created!")
        
        # Step 5: Evaluation
        logger.info("📈 STEP 5: Generating Evaluation Metrics...")
        evaluation_results = ensemble_model.get_evaluation_metrics()
        logger.info("   ✅ Evaluation completed!")
        
        logger.info("="*60)
        logger.info("🎉 FULL DEMONSTRATION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        pipeline_results = {
            'eda': eda_results,
            'feature_engineering': feature_results,
            'model_training': ensemble_results,
            'evaluation': evaluation_results
        }
        
        return jsonify({
            'success': True,
            'results': pipeline_results,
            'message': 'Complete demonstration pipeline executed successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error in full pipeline: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/single', methods=['POST'])
def predict_single_revenue():
    """Predict revenue for a specific date and meal period"""
    try:
        data = request.get_json()
        date = data.get('date')
        meal_period = data.get('meal_period')
        model_type = data.get('model_type', 'best_ensemble')
        
        if not date or not meal_period:
            return jsonify({'error': 'Date and meal_period are required'}), 400
        
        logger.info(f"🔮 Predicting revenue for {date}, {meal_period}")
        
        prediction = ensemble_model.predict_single(date, meal_period, model_type)
        
        if 'error' in prediction:
            logger.error(f"❌ Prediction error: {prediction['error']}")
            return jsonify(prediction), 400
        
        logger.info(f"✅ Prediction successful: ${prediction['predicted_revenue']}")
        logger.info(f"   📅 Date: {prediction['date']}")
        logger.info(f"   🍽️ Meal: {prediction['meal_period']}")
        logger.info(f"   🏆 Best model: {prediction['best_model']}")
        logger.info(f"   🕌 Islamic period: {prediction['islamic_period']}")
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"❌ Error in single prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/90-days', methods=['POST'])
def predict_90_days_forecast():
    """Generate 90-day revenue forecast"""
    try:
        data = request.get_json() or {}
        start_date = data.get('start_date')
        
        logger.info(f"📈 Generating 90-day forecast starting from {start_date or 'tomorrow'}")
        
        forecast = ensemble_model.predict_90_days(start_date)
        
        if 'error' in forecast:
            logger.error(f"❌ Forecast error: {forecast['error']}")
            return jsonify(forecast), 400
        
        logger.info(f"✅ 90-day forecast completed successfully")
        logger.info(f"   📊 Total predictions: {forecast['total_predictions']}")
        logger.info(f"   💰 Total forecast revenue: ${forecast['total_forecast_revenue']:,.2f}")
        logger.info(f"   📅 Forecast period: {forecast['forecast_period']}")
        logger.info(f"   📈 Daily average: ${forecast['daily_average_revenue']:,.2f}")
        
        return jsonify(forecast)
        
    except Exception as e:
        logger.error(f"❌ Error in 90-day forecast: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/accuracy-plots', methods=['GET'])
def get_accuracy_plots():
    """Generate model accuracy visualization plots"""
    try:
        logger.info("📊 Generating model accuracy plots...")
        
        plots = ensemble_model.create_accuracy_plots()
        
        if 'error' in plots:
            logger.error(f"❌ Plot generation error: {plots['error']}")
            return jsonify(plots), 400
        
        logger.info(f"✅ Accuracy plots generated successfully")
        logger.info(f"   🧪 Test samples: {plots['test_samples']}")
        logger.info(f"   🏆 Best model: {plots['best_model']}")
        logger.info(f"   📈 Models analyzed: {len(plots['metrics_summary'])}")
        
        return jsonify(plots)
        
    except Exception as e:
        logger.error(f"❌ Error generating accuracy plots: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/validation-plots', methods=['GET'])
def get_validation_plots():
    """Get validation and test performance plots"""
    try:
        logger.info("📊 Generating validation plots...")
        
        # This will create plots showing predicted vs actual for validation/test data
        plots = ensemble_model.create_accuracy_plots()
        
        if 'error' in plots:
            logger.error(f"❌ Validation plot error: {plots['error']}")
            return jsonify(plots), 400
        
        logger.info(f"✅ Validation plots generated successfully")
        
        return jsonify({
            'success': True,
            'validation_plots': plots,
            'message': 'Validation plots showing predicted vs actual values'
        })
        
    except Exception as e:
        logger.error(f"❌ Error generating validation plots: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/time-series-plots', methods=['GET'])
def get_time_series_plots():
    """Generate time series plots showing actual vs predicted values over time"""
    try:
        logger.info("📈 Generating time series actual vs predicted plots...")
        
        plots = ensemble_model.create_time_series_plots()
        
        if 'error' in plots:
            logger.error(f"❌ Time series plot error: {plots['error']}")
            return jsonify(plots), 400
        
        logger.info(f"✅ Time series plots generated successfully")
        logger.info(f"   📅 Date range: {plots['date_range']}")
        logger.info(f"   🧪 Test samples: {plots['test_samples']}")
        logger.info(f"   🏆 Best model: {plots['metrics']['best_model']}")
        logger.info(f"   📊 R² score: {plots['metrics']['r2_score']:.3f}")
        
        return jsonify(plots)
        
    except Exception as e:
        logger.error(f"❌ Error generating time series plots: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ===========================
# EDA ANALYSIS ENDPOINTS
# ===========================

@app.route('/api/eda/data-overview', methods=['GET'])
def get_data_overview():
    """Get comprehensive data overview and basic statistics"""
    try:
        logger.info("🔍 Starting EDA data overview analysis...")
        
        # Load all revenue center data
        df = load_all_revenue_data()
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        # Basic info
        data_shape = df.shape
        missing_values = df.isnull().sum().to_dict()
        data_types = df.dtypes.astype(str).to_dict()
        
        # Date range
        date_range = {
            'start_date': df['Date'].min().strftime('%Y-%m-%d'),
            'end_date': df['Date'].max().strftime('%Y-%m-%d'),
            'total_days': (df['Date'].max() - df['Date'].min()).days
        }
        
        # Revenue centers
        revenue_centers = df['RevenueCenterName'].unique().tolist()
        meal_periods = df['MealPeriod'].unique().tolist()
        
        # Revenue statistics
        revenue_stats = {
            'total_revenue': float(df['CheckTotal'].sum()),
            'mean_revenue': float(df['CheckTotal'].mean()),
            'median_revenue': float(df['CheckTotal'].median()),
            'min_revenue': float(df['CheckTotal'].min()),
            'max_revenue': float(df['CheckTotal'].max()),
            'std_revenue': float(df['CheckTotal'].std())
        }
        
        logger.info(f"✅ Data overview completed successfully")
        logger.info(f"   📊 Shape: {data_shape}")
        logger.info(f"   💰 Total Revenue: ${revenue_stats['total_revenue']:,.2f}")
        
        return jsonify({
            'success': True,
            'data_shape': data_shape,
            'missing_values': missing_values,
            'data_types': data_types,
            'date_range': date_range,
            'revenue_centers': revenue_centers,
            'meal_periods': meal_periods,
            'revenue_stats': revenue_stats
        })
        
    except Exception as e:
        logger.error(f"❌ Error in data overview: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/eda/revenue-distributions', methods=['GET'])
def get_revenue_distributions():
    """Generate revenue distribution plots and analysis"""
    try:
        logger.info("📈 Starting revenue distribution analysis...")
        
        df = load_all_revenue_data()
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        plots = {}
        
        # 1. Overall revenue distribution
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(df['CheckTotal'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Overall Revenue Distribution')
        plt.xlabel('Revenue ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. Revenue by meal period
        plt.subplot(2, 2, 2)
        df.boxplot(column='CheckTotal', by='MealPeriod', ax=plt.gca())
        plt.title('Revenue Distribution by Meal Period')
        plt.xlabel('Meal Period')
        plt.ylabel('Revenue ($)')
        plt.suptitle('')
        
        # 3. Revenue by day of week
        plt.subplot(2, 2, 3)
        revenue_by_dow = df.groupby('DayOfWeek')['CheckTotal'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        plt.bar(range(7), [revenue_by_dow.get(i, 0) for i in range(7)], color='lightcoral')
        plt.title('Average Revenue by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Revenue ($)')
        plt.xticks(range(7), days)
        plt.grid(True, alpha=0.3)
        
        # 4. Revenue by revenue center
        plt.subplot(2, 2, 4)
        revenue_by_center = df.groupby('RevenueCenterName')['CheckTotal'].mean().sort_values(ascending=False)
        plt.bar(range(len(revenue_by_center)), revenue_by_center.values, color='lightgreen')
        plt.title('Average Revenue by Revenue Center')
        plt.xlabel('Revenue Center')
        plt.ylabel('Average Revenue ($)')
        # Use actual revenue center names from sorted data, not generic RC_1, RC_2...
        center_labels = [name.replace('RevenueCenter_', 'RC_') for name in revenue_by_center.index]
        plt.xticks(range(len(revenue_by_center)), center_labels, rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['revenue_distributions'] = plot_to_base64()
        
        # Statistical summary by categories
        meal_period_stats = df.groupby('MealPeriod')['CheckTotal'].agg(['mean', 'median', 'std']).to_dict()
        center_stats = df.groupby('RevenueCenterName')['CheckTotal'].agg(['mean', 'median', 'std']).to_dict()
        
        logger.info(f"✅ Revenue distribution analysis completed successfully")
        
        return jsonify({
            'success': True,
            'plots': plots,
            'meal_period_stats': meal_period_stats,
            'center_stats': center_stats
        })
        
    except Exception as e:
        logger.error(f"❌ Error in revenue distribution analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/eda/time-series-analysis', methods=['GET'])
def get_time_series_analysis():
    """Analyze revenue patterns over time"""
    try:
        logger.info("⏰ Starting time series analysis...")
        
        df = load_all_revenue_data()
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        plots = {}
        
        # Aggregate daily revenue
        daily_revenue = df.groupby('Date')['CheckTotal'].sum().reset_index()
        
        # 1. Daily revenue time series
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(daily_revenue['Date'], daily_revenue['CheckTotal'], linewidth=1, alpha=0.7)
        plt.title('Daily Revenue Time Series')
        plt.xlabel('Date')
        plt.ylabel('Daily Revenue ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Monthly revenue trends
        df['YearMonth'] = df['Date'].dt.to_period('M')
        monthly_revenue = df.groupby('YearMonth')['CheckTotal'].sum()
        
        plt.subplot(3, 1, 2)
        monthly_revenue.plot(kind='bar', alpha=0.8, color='orange')
        plt.title('Monthly Revenue Trends')
        plt.xlabel('Month')
        plt.ylabel('Monthly Revenue ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Revenue by hour pattern (if available)
        plt.subplot(3, 1, 3)
        meal_period_revenue = df.groupby('MealPeriod')['CheckTotal'].mean()
        meal_period_revenue.plot(kind='bar', alpha=0.8, color='green')
        plt.title('Average Revenue by Meal Period')
        plt.xlabel('Meal Period')
        plt.ylabel('Average Revenue ($)')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['time_series'] = plot_to_base64()
        
        # Seasonal patterns
        seasonal_stats = {
            'daily_stats': daily_revenue['CheckTotal'].describe().to_dict(),
            'monthly_stats': monthly_revenue.describe().to_dict(),
            'meal_period_stats': meal_period_revenue.to_dict()
        }
        
        logger.info(f"✅ Time series analysis completed successfully")
        
        return jsonify({
            'success': True,
            'plots': plots,
            'seasonal_stats': seasonal_stats
        })
        
    except Exception as e:
        logger.error(f"❌ Error in time series analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/eda/correlation-analysis', methods=['GET'])
def get_correlation_analysis():
    """Analyze correlations with CheckTotal (Revenue)"""
    try:
        logger.info("🔗 Starting correlation analysis with revenue (CheckTotal)...")
        
        df = load_all_revenue_data()
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        # Check if CheckTotal column exists
        if 'CheckTotal' not in df.columns:
            return jsonify({'error': 'CheckTotal column not found in data'}), 400
        
        # Select numerical columns (excluding CheckTotal itself for predictor analysis)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        predictor_cols = [col for col in numerical_cols if col != 'CheckTotal']
        
        if len(predictor_cols) < 1:
            return jsonify({'error': 'Not enough predictor columns for correlation analysis'}), 400
        
        # Filter out columns with zero variance (constant columns)
        valid_cols = []
        for col in predictor_cols:
            if df[col].std() > 0:  # Only include columns with variation
                valid_cols.append(col)
        
        if len(valid_cols) < 1:
            return jsonify({'error': 'Not enough variable columns for correlation analysis (all predictor columns are constant)'}), 400
        
        logger.info(f"   📊 Analyzing correlations with CheckTotal using {len(valid_cols)} predictor variables")
        
        # Calculate correlations with CheckTotal
        revenue_correlations = {}
        for col in valid_cols:
            corr_val = df['CheckTotal'].corr(df[col])
            if not pd.isna(corr_val) and not np.isinf(corr_val):
                revenue_correlations[col] = float(corr_val)
            else:
                revenue_correlations[col] = 0.0
        
        # Sort correlations by absolute value (strongest first)
        sorted_correlations = dict(sorted(revenue_correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        
        # Create correlation visualization (bar plot showing correlations with revenue)
        plt.figure(figsize=(14, 8))
        
        # All correlations with CheckTotal
        variables = list(sorted_correlations.keys())
        correlations = list(sorted_correlations.values())
        colors = ['green' if corr > 0 else 'red' for corr in correlations]
        
        bars = plt.barh(range(len(variables)), correlations, color=colors, alpha=0.7)
        plt.yticks(range(len(variables)), variables)
        plt.xlabel('Correlation with Revenue (CheckTotal)')
        plt.title('Correlation of Variables with Revenue')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            plt.text(corr + (0.02 if corr >= 0 else -0.02), i, f'{corr:.3f}', 
                    va='center', ha='left' if corr >= 0 else 'right', fontsize=8)
        
        plt.tight_layout()
        correlation_plot = plot_to_base64()
        
        logger.info(f"✅ Revenue correlation analysis completed")
        logger.info(f"   🔗 Analyzed correlations with revenue for {len(valid_cols)} variables")
        
        return jsonify({
            'success': True,
            'correlation_plot': correlation_plot,
            'revenue_correlations': sorted_correlations,
            'analyzed_variables': valid_cols,
            'total_variables': len(valid_cols)
        })
        
    except Exception as e:
        logger.error(f"❌ Error in correlation analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/eda/categorical-analysis', methods=['GET'])
def get_categorical_analysis():
    """Analyze categorical variables and their impact on revenue"""
    try:
        logger.info("📋 Starting categorical analysis...")
        
        df = load_all_revenue_data()
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        plots = {}
        categorical_stats = {}
        
        # Key categorical columns to analyze
        categorical_cols = ['MealPeriod', 'DayOfWeek', 'Month', 'IslamicPeriod', 
                           'TourismIntensity', 'RevenueImpact']
        
        # Remove columns that don't exist in the data
        available_categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # 1. Revenue by categorical variables
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(available_categorical_cols[:6]):
            if i < len(axes):
                revenue_by_category = df.groupby(col)['CheckTotal'].mean().sort_values(ascending=False)
                categorical_stats[col] = revenue_by_category.to_dict()
                
                axes[i].bar(range(len(revenue_by_category)), revenue_by_category.values, alpha=0.8)
                axes[i].set_title(f'Average Revenue by {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Average Revenue ($)')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Set x-tick labels
                if len(revenue_by_category) <= 10:
                    axes[i].set_xticks(range(len(revenue_by_category)))
                    axes[i].set_xticklabels(revenue_by_category.index, rotation=45, ha='right')
                else:
                    axes[i].set_xticks(range(0, len(revenue_by_category), max(1, len(revenue_by_category)//5)))
                    axes[i].set_xticklabels([revenue_by_category.index[j] for j in range(0, len(revenue_by_category), max(1, len(revenue_by_category)//5))], rotation=45, ha='right')
        
        # Hide unused subplots
        for i in range(len(available_categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plots['categorical_analysis'] = plot_to_base64()
        
        # Event impact analysis
        event_columns = [col for col in df.columns if col.startswith('Is') and col != 'is_zero']
        event_impact = {}
        
        for event_col in event_columns:
            if df[event_col].dtype == 'object':
                # Convert boolean-like strings to actual booleans
                df[event_col] = df[event_col].astype(str).str.lower().isin(['true', '1', 'yes'])
            
            event_revenue = df.groupby(event_col)['CheckTotal'].mean()
            if len(event_revenue) == 2:  # Binary event
                impact = event_revenue.get(True, event_revenue.get(1, 0)) - event_revenue.get(False, event_revenue.get(0, 0))
                event_impact[event_col] = {
                    'with_event': float(event_revenue.get(True, event_revenue.get(1, 0))),
                    'without_event': float(event_revenue.get(False, event_revenue.get(0, 0))),
                    'impact': float(impact)
                }
        
        logger.info(f"✅ Categorical analysis completed")
        logger.info(f"   📋 Analyzed {len(available_categorical_cols)} categorical variables")
        logger.info(f"   🎯 Analyzed {len(event_impact)} event impacts")
        
        return jsonify({
            'success': True,
            'plots': plots,
            'categorical_stats': categorical_stats,
            'event_impact': event_impact,
            'analyzed_columns': available_categorical_cols
        })
        
    except Exception as e:
        logger.error(f"❌ Error in categorical analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/eda/outlier-analysis', methods=['GET'])
def get_outlier_analysis():
    """Detect and analyze outliers in revenue data"""
    try:
        logger.info("🎯 Starting outlier analysis...")
        
        df = load_all_revenue_data()
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        # Focus on revenue outliers
        revenue = df['CheckTotal']
        
        # Calculate outlier thresholds using IQR method
        Q1 = revenue.quantile(0.25)
        Q3 = revenue.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = df[(revenue < lower_bound) | (revenue > upper_bound)]
        
        # Create outlier visualization
        plt.figure(figsize=(15, 10))
        
        # 1. Box plot
        plt.subplot(2, 2, 1)
        plt.boxplot(revenue, vert=True)
        plt.title('Revenue Box Plot with Outliers')
        plt.ylabel('Revenue ($)')
        plt.grid(True, alpha=0.3)
        
        # 2. Histogram with outliers highlighted
        plt.subplot(2, 2, 2)
        plt.hist(revenue, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower bound: ${lower_bound:.2f}')
        plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper bound: ${upper_bound:.2f}')
        plt.title('Revenue Distribution with Outlier Bounds')
        plt.xlabel('Revenue ($)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Outliers by meal period
        plt.subplot(2, 2, 3)
        outlier_by_meal = outliers['MealPeriod'].value_counts()
        outlier_by_meal.plot(kind='bar', alpha=0.8, color='orange')
        plt.title('Outliers by Meal Period')
        plt.xlabel('Meal Period')
        plt.ylabel('Number of Outliers')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. Outliers over time
        plt.subplot(2, 2, 4)
        outliers_daily = outliers.groupby('Date').size()
        plt.plot(outliers_daily.index, outliers_daily.values, marker='o', alpha=0.7)
        plt.title('Outliers Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Outliers')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        outlier_plot = plot_to_base64()
        
        # Outlier statistics
        outlier_stats = {
            'total_records': len(df),
            'total_outliers': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR),
            'max_outlier': float(outliers['CheckTotal'].max()) if len(outliers) > 0 else None,
            'min_outlier': float(outliers['CheckTotal'].min()) if len(outliers) > 0 else None
        }
        
        # Outlier details (top 10 highest and lowest)
        top_outliers = outliers.nlargest(10, 'CheckTotal')[['Date', 'MealPeriod', 'RevenueCenterName', 'CheckTotal']].to_dict('records')
        bottom_outliers = outliers.nsmallest(10, 'CheckTotal')[['Date', 'MealPeriod', 'RevenueCenterName', 'CheckTotal']].to_dict('records')
        
        logger.info(f"✅ Outlier analysis completed")
        logger.info(f"   🎯 Found {len(outliers)} outliers ({outlier_stats['outlier_percentage']:.2f}%)")
        
        return jsonify({
            'success': True,
            'outlier_plot': outlier_plot,
            'outlier_stats': outlier_stats,
            'top_outliers': top_outliers,
            'bottom_outliers': bottom_outliers
        })
        
    except Exception as e:
        logger.error(f"❌ Error in outlier analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000) 