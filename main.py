"""
Main script untuk menjalankan KNN recommendation system
"""

import numpy as np
import pandas as pd
from knn_model import NutritionKNN, load_nutrition_csv, load_model
import json
import os

# Path ke CSV file
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data_gizi.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def main():
    """
    Contoh penggunaan KNN untuk rekomendasi nutrisi
    """
    print("=" * 60)
    print("K-Nearest Neighbors Nutrition Recommendation System")
    print("=" * 60)
    
    # 1. Load or train model
    print("\n[1] Initializing model...")
    model = None
    feature_names = None
    
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from pickle: {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            feature_names = model.feature_names
            print(f"✓ Loaded {len(model.y)} food items from pickle")
        except Exception as e:
            print(f"✗ Failed to load pickle: {str(e)}")
            print(f"Training from CSV instead...")
            model = None
    
    # If pickle not available or failed, train from CSV
    if model is None:
        print(f"\nLoading nutrition data from CSV...")
        try:
            X, y, feature_names = load_nutrition_csv(CSV_PATH)
            print(f"✓ Loaded {len(X)} food items")
            print(f"✓ Features: {feature_names}")
            
            # Train model
            print("\n[2] Training KNN model...")
            model = NutritionKNN(k=5)
            model.fit(X, y, feature_names)
            print("✓ Model trained successfully")
            
            # Save model to pickle
            print(f"\nSaving model to pickle: {MODEL_PATH}")
            model.save_model(MODEL_PATH)
            print("✓ Model saved for faster startup next time")
        except FileNotFoundError:
            print(f"✗ Error: CSV file not found at {CSV_PATH}")
            return
    
    # 3. Contoh rekomendasi untuk user
    print("\n[3] Generating recommendations...")
    
    # User requirement example: tinggi protein, rendah lemak
    user_requirements = np.array([25.0, 50.0, 5.0])  # [protein, carbs, fat]
    
    print(f"\nUser requirements: {dict(zip(feature_names, user_requirements))}")
    
    # Get recommendations
    recommendations = model.recommend(user_requirements, k=10)
    
    print("\nTop 10 Food Recommendations:")
    print("-" * 80)
    for rec in recommendations:
        print(f"\n{rec['rank']}. {rec['name']}")
        print(f"   Similarity Score: {rec['similarity_score']:.4f}")
        print(f"   Nutrients: {rec['nutrients']}")
    
    # 4. Analisis profil nutrisi
    print("\n[4] Nutritional Profile Analysis:")
    print("-" * 80)
    profile = model.get_nutritional_profile(user_requirements)
    
    print("\nUser Nutrients vs Average Dataset:")
    for feature in feature_names:
        user_val = profile['user_nutrients'][feature]
        avg_val = profile['average_nutrients'][feature]
        diff = profile['differences'][feature]
        
        print(f"\n{feature}:")
        print(f"  User Value: {user_val:.2f}g")
        print(f"  Average:    {avg_val:.2f}g")
        print(f"  Difference: {diff:+.2f}g")
    
    # 5. Test dengan beberapa kombinasi
    print("\n[5] Testing with different requirements...")
    print("-" * 80)
    
    test_cases = [
        {"name": "High Protein (Muscle Building)", 
         "nutrients": np.array([35.0, 40.0, 8.0])},
        {"name": "Low Fat (Calorie Control)", 
         "nutrients": np.array([20.0, 60.0, 2.0])},
        {"name": "Balanced Nutrition", 
         "nutrients": np.array([15.0, 55.0, 3.0])},
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print(f"Requirements: {dict(zip(feature_names, test_case['nutrients']))}")
        
        recs = model.recommend(test_case['nutrients'], k=3)
        for rec in recs:
            print(f"  {rec['rank']}. {rec['name']} (Similarity: {rec['similarity_score']:.4f})")


def interactive_mode():
    """
    Interactive mode untuk input user
    """
    print("\n" + "=" * 60)
    print("Interactive KNN Recommendation System")
    print("=" * 60)
    
    # Load or train model
    print("\nInitializing model...")
    model = None
    feature_names = None
    
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from pickle: {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            feature_names = model.feature_names
            print(f"✓ Model loaded from pickle")
        except Exception as e:
            print(f"✗ Failed to load pickle: {str(e)}")
            print(f"Training from CSV instead...")
            model = None
    
    # If pickle not available or failed, train from CSV
    if model is None:
        print(f"Loading nutrition data from CSV...")
        try:
            X, y, feature_names = load_nutrition_csv(CSV_PATH)
            print(f"✓ Loaded {len(X)} food items")
            
            # Train model
            model = NutritionKNN(k=5)
            model.fit(X, y, feature_names)
            
            # Save model to pickle
            print(f"\nSaving model to pickle: {MODEL_PATH}")
            model.save_model(MODEL_PATH)
        except FileNotFoundError:
            print(f"✗ Error: CSV file not found at {CSV_PATH}")
            return
    
    while True:
        print("\n" + "-" * 60)
        print("Enter your nutritional requirements (or 'quit' to exit):")
        
        try:
            user_input = []
            for feature in feature_names:
                value = input(f"Enter {feature} (in grams): ")
                
                if value.lower() == 'quit':
                    print("\nGoodbye!")
                    return
                
                user_input.append(float(value))
            
            user_nutrients = np.array(user_input)
            k = int(input("How many recommendations do you want? (default: 5): ") or "5")
            
            # Get recommendations
            recommendations = model.recommend(user_nutrients, k=k)
            
            print(f"\nTop {k} Food Recommendations for you:")
            print("-" * 60)
            for rec in recommendations:
                print(f"\n{rec['rank']}. {rec['name']}")
                print(f"   Similarity Score: {rec['similarity_score']:.4f}")
                print(f"   Nutrients:")
                for nutrient, value in rec['nutrients'].items():
                    print(f"     - {nutrient}: {value:.2f}g")
        
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run demo
    main()
    
    # Uncomment untuk interactive mode
    # interactive_mode()
