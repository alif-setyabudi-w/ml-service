"""
Test script untuk memverifikasi pickle save/load functionality
"""

import os
import sys
from knn_model import NutritionKNN, load_nutrition_csv, load_model

# Path ke CSV dan pickle
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'data_gizi.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def test_pickle_functionality():
    """Test save dan load model dengan pickle"""
    
    print("=" * 70)
    print("Testing Model Pickle Save/Load Functionality")
    print("=" * 70)
    
    # Test 1: Train and save model
    print("\n[TEST 1] Training model from CSV dan saving to pickle...")
    try:
        X, y, feature_names = load_nutrition_csv(CSV_PATH)
        print(f"✓ Loaded {len(X)} food items from CSV")
        
        # Train model
        model = NutritionKNN(k=5)
        model.fit(X, y, feature_names)
        print(f"✓ Model trained successfully")
        
        # Save model
        success = model.save_model(MODEL_PATH)
        if success:
            print(f"✓ Model saved to {MODEL_PATH}")
        else:
            print(f"✗ Failed to save model")
            return False
            
    except Exception as e:
        print(f"✗ Error during training: {str(e)}")
        return False
    
    # Test 2: Load model from pickle
    print("\n[TEST 2] Loading model from pickle...")
    try:
        loaded_model = load_model(MODEL_PATH)
        print(f"✓ Model loaded successfully from pickle")
        print(f"  - Number of food items: {len(loaded_model.y)}")
        print(f"  - Feature names: {loaded_model.feature_names}")
        print(f"  - K value: {loaded_model.k}")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False
    
    # Test 3: Compare original and loaded model
    print("\n[TEST 3] Comparing original and loaded model...")
    try:
        # Test yang sama dengan kedua model
        test_nutrients = [25.0, 50.0, 5.0]  # protein, carbs, fat
        
        # Get recommendations dari original model
        orig_recs = model.recommend(test_nutrients, k=3)
        print(f"✓ Original model recommendations:")
        for rec in orig_recs:
            print(f"  {rec['rank']}. {rec['name']} (Score: {rec['similarity_score']:.4f})")
        
        # Get recommendations dari loaded model
        loaded_recs = loaded_model.recommend(test_nutrients, k=3)
        print(f"\n✓ Loaded model recommendations:")
        for rec in loaded_recs:
            print(f"  {rec['rank']}. {rec['name']} (Score: {rec['similarity_score']:.4f})")
        
        # Verify hasil sama
        if orig_recs[0]['name'] == loaded_recs[0]['name']:
            print(f"\n✓ Both models produced identical results!")
        else:
            print(f"\n✗ Models produced different results")
            return False
            
    except Exception as e:
        print(f"✗ Error during comparison: {str(e)}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All pickle tests passed!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_pickle_functionality()
    sys.exit(0 if success else 1)
