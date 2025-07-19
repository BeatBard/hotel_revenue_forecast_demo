#!/usr/bin/env python3
"""
Test script to verify predictions work without retraining
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.models.ensemble_model import HotelRevenueEnsemble
from backend.utils.data_loader import DataLoader
import warnings
warnings.filterwarnings('ignore')

def test_prediction_without_retrain():
    """Test that predictions use trained models without retraining"""
    
    print("🧪 TESTING PREDICTION WITHOUT RETRAINING")
    print("=" * 50)
    
    # Step 1: Load data and train models (ONE TIME)
    print("📊 Step 1: Loading data...")
    data_loader = DataLoader()
    data = data_loader.load_data()
    
    if data is None:
        print("❌ Failed to load data")
        return False
    
    print(f"✅ Data loaded: {len(data)} records")
    
    ensemble_model = HotelRevenueEnsemble()
    
    # Step 2: Train models (ONE TIME)
    print("\n🔧 Step 2: Training models (one time)...")
    splits = ensemble_model.create_temporal_splits(data)
    results = ensemble_model.train_ensemble_models(splits)
    
    if 'error' in results:
        print(f"❌ Training failed: {results['error']}")
        return False
    
    print("✅ Models trained successfully")
    print(f"   📊 Training samples: {results.get('training_samples', 'N/A')}")
    print(f"   🏆 Models available: {list(ensemble_model.models.keys())}")
    
    # Step 3: Test multiple predictions (NO RETRAINING)
    print("\n🔮 Step 3: Testing multiple predictions...")
    
    test_cases = [
        ("2024-05-15", "Breakfast"),
        ("2024-05-15", "Lunch"),
        ("2024-05-15", "Dinner"),
        ("2024-05-20", "Breakfast"),
        ("2024-05-25", "Dinner")
    ]
    
    predictions = []
    
    for date, meal in test_cases:
        print(f"   🍽️ Predicting {date} {meal}...")
        pred = ensemble_model.predict_single(date, meal)
        
        if pred.get('success'):
            predictions.append({
                'date': date,
                'meal': meal,
                'revenue': pred['predicted_revenue']
            })
            print(f"      💰 ${pred['predicted_revenue']:.2f}")
        else:
            print(f"      ❌ Error: {pred.get('error', 'Unknown')}")
            return False
    
    # Step 4: Check predictions are different (not all zeros)
    print("\n📈 Step 4: Validating predictions...")
    
    revenues = [p['revenue'] for p in predictions]
    unique_revenues = len(set(revenues))
    
    print(f"   📊 Revenue values: {revenues}")
    print(f"   🔢 Unique values: {unique_revenues}")
    print(f"   💰 Range: ${min(revenues):.2f} - ${max(revenues):.2f}")
    
    if all(r == 0 for r in revenues):
        print("   ❌ All predictions are zero - model not working")
        return False
    elif unique_revenues == 1:
        print("   ⚠️  All predictions are identical - may need better features")
        return False
    else:
        print("   ✅ Predictions show variation - model working!")
    
    # Step 5: Test 90-day forecast (should be fast since no retraining)
    print("\n📅 Step 5: Testing 90-day forecast...")
    
    forecast = ensemble_model.predict_90_days("2024-05-01")
    
    if 'error' in forecast:
        print(f"   ❌ Forecast failed: {forecast['error']}")
        return False
    else:
        print(f"   ✅ Forecast successful: {forecast['total_predictions']} predictions")
        print(f"   💰 Total forecast: ${forecast['total_forecast_revenue']:,.2f}")
    
    return True

if __name__ == "__main__":
    success = test_prediction_without_retrain()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("Models train once and predict many times without retraining.")
    else:
        print("\n❌ FAILED!")
        print("There are still issues with the prediction pipeline.")
    
    sys.exit(0 if success else 1) 