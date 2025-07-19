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
from datetime import datetime
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

if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000) 