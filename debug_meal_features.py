#!/usr/bin/env python3
"""
Debug script to verify meal period feature differentiation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.models.ensemble_model import HotelRevenueEnsemble
from backend.utils.data_loader import DataLoader
import warnings
import logging
warnings.filterwarnings('ignore')

def debug_meal_features():
    """Debug meal period feature differentiation"""
    
    print("🔍 DEBUGGING MEAL PERIOD FEATURES")
    print("=" * 50)
    
    # Enable detailed logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data and train model
    print("📊 Loading data and training model...")
    data_loader = DataLoader()
    data = data_loader.load_data()
    
    ensemble_model = HotelRevenueEnsemble()
    splits = ensemble_model.create_temporal_splits(data)
    results = ensemble_model.train_ensemble_models(splits)
    
    if 'error' in results:
        print(f"❌ Training failed: {results['error']}")
        return False
    
    print("✅ Model trained successfully")
    
    # Test feature differentiation for same date, different meals
    test_date = "2024-05-15"
    meals = ['Breakfast', 'Lunch', 'Dinner']
    
    print(f"\n🍽️ Testing feature differentiation for {test_date}:")
    print("=" * 60)
    
    predictions = {}
    
    for meal in meals:
        print(f"\n📋 {meal.upper()} PREDICTION:")
        print("-" * 30)
        
        # This will log detailed meal features due to our debug code
        pred = ensemble_model.predict_single(test_date, meal)
        
        if pred.get('success'):
            predictions[meal] = pred['predicted_revenue']
            print(f"Final prediction: ${pred['predicted_revenue']:.2f}")
        else:
            print(f"❌ Failed: {pred.get('error', 'Unknown error')}")
            return False
    
    # Analyze differentiation
    print(f"\n📊 PREDICTION ANALYSIS:")
    print("=" * 40)
    
    for meal, revenue in predictions.items():
        print(f"{meal:>10}: ${revenue:>8.2f}")
    
    # Check if predictions are properly differentiated
    breakfast_lunch_diff = abs(predictions['Breakfast'] - predictions['Lunch'])
    breakfast_dinner_diff = abs(predictions['Breakfast'] - predictions['Dinner'])
    lunch_dinner_diff = abs(predictions['Lunch'] - predictions['Dinner'])
    
    print(f"\n🔍 DIFFERENTIATION CHECK:")
    print("-" * 30)
    print(f"Breakfast vs Lunch:  ${breakfast_lunch_diff:>8.2f}")
    print(f"Breakfast vs Dinner: ${breakfast_dinner_diff:>8.2f}")
    print(f"Lunch vs Dinner:     ${lunch_dinner_diff:>8.2f}")
    
    # Acceptable difference threshold
    min_difference = 100.0  # $100 minimum difference expected
    
    is_differentiated = (
        breakfast_lunch_diff > min_difference or
        breakfast_dinner_diff > min_difference or
        lunch_dinner_diff > min_difference
    )
    
    if is_differentiated:
        print(f"\n✅ SUCCESS: Meal periods show good differentiation!")
        print(f"   Largest difference: ${max(breakfast_lunch_diff, breakfast_dinner_diff, lunch_dinner_diff):.2f}")
    else:
        print(f"\n❌ ISSUE: Meal periods still too similar")
        print(f"   Maximum difference: ${max(breakfast_lunch_diff, breakfast_dinner_diff, lunch_dinner_diff):.2f}")
        print(f"   Expected minimum: ${min_difference}")
        
        print(f"\n🔧 TROUBLESHOOTING SUGGESTIONS:")
        print("1. Check if meal period features are being created correctly")
        print("2. Verify model is trained with sufficient meal period variation")
        print("3. Check if lag/rolling features are overwhelming meal signals")
        print("4. Consider if training data has enough meal period diversity")
    
    return is_differentiated

if __name__ == "__main__":
    success = debug_meal_features()
    
    if success:
        print("\n🎉 Meal period features are working correctly!")
    else:
        print("\n❌ Meal period features need fixing!")
    
    sys.exit(0 if success else 1) 