#!/usr/bin/env python3
"""
Debug script to test individual predictions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.models.ensemble_model import HotelRevenueEnsemble
from backend.utils.data_loader import DataLoader
import warnings
warnings.filterwarnings('ignore')

def debug_single_prediction():
    """Debug a single prediction to understand what's happening"""
    
    print("🐛 DEBUGGING SINGLE PREDICTION")
    print("=" * 50)
    
    # Step 1: Load data
    print("📊 Loading data...")
    data_loader = DataLoader()
    data = data_loader.load_data()
    print(f"✅ Data loaded: {len(data)} records")
    
    # Step 2: Train model
    print("\n🔧 Training model...")
    ensemble_model = HotelRevenueEnsemble()
    splits = ensemble_model.create_temporal_splits(data)
    
    print(f"   Training split: {len(splits['train'])} records")
    print(f"   Validation split: {len(splits['validation'])} records")
    print(f"   Test split: {len(splits['test'])} records")
    
    if len(splits['train']) < 10:
        print("   ⚠️ WARNING: Very small training set!")
    
    results = ensemble_model.train_ensemble_models(splits)
    
    if 'error' in results:
        print(f"❌ Training failed: {results['error']}")
        return False
    
    print("✅ Training completed")
    print(f"   Models trained: {list(ensemble_model.models.keys())}")
    print(f"   Features: {len(ensemble_model.feature_names)}")
    
    # Step 3: Test single prediction with debug info
    print("\n🔮 Testing single prediction...")
    test_date = "2024-05-15"
    test_meal = "Lunch"
    
    # Enable debug logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    prediction = ensemble_model.predict_single(test_date, test_meal)
    
    print(f"\n📋 PREDICTION RESULT:")
    if prediction.get('success'):
        print(f"   ✅ Success: ${prediction['predicted_revenue']}")
        print(f"   🏆 Best model: {prediction['best_model']}")
        print(f"   🤖 Individual predictions:")
        for model, pred in prediction.get('individual_predictions', {}).items():
            print(f"      {model}: ${pred:.2f}")
        
        if prediction.get('ensemble_predictions'):
            print(f"   🎯 Ensemble predictions:")
            for ensemble, pred in prediction['ensemble_predictions'].items():
                print(f"      {ensemble}: ${pred:.2f}")
    else:
        print(f"   ❌ Failed: {prediction.get('error', 'Unknown error')}")
        return False
    
    # Step 4: Test multiple predictions to see variation
    print(f"\n🍽️ Testing meal period variation...")
    meals = ['Breakfast', 'Lunch', 'Dinner']
    for meal in meals:
        pred = ensemble_model.predict_single(test_date, meal)
        if pred.get('success'):
            print(f"   {meal}: ${pred['predicted_revenue']:.2f}")
        else:
            print(f"   {meal}: ERROR")
    
    return True

if __name__ == "__main__":
    success = debug_single_prediction()
    
    if success:
        print("\n🎉 Debug completed successfully!")
    else:
        print("\n❌ Debug revealed issues!")
    
    sys.exit(0 if success else 1) 