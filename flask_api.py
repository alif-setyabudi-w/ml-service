"""
Flask API untuk KNN Nutrition Recommendation System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import pandas as pd
from knn_model import NutritionKNN, load_nutrition_csv, load_model

app = Flask(__name__)
CORS(app)

# Path ke CSV file
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data_gizi.csv')

# Path ke model pickle file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Initialize model globally
model = None
feature_names = None
nutrition_df = None


def init_model():
    """Initialize KNN model on startup"""
    global model, feature_names
    
    print("\n" + "="*70)
    print("[MODEL INITIALIZATION] Starting at server startup...")
    print("="*70)
    
    try:
        # Coba load dari pickle file terlebih dahulu
        if os.path.exists(MODEL_PATH):
            print(f"\n[STEP 1] Loading model from pickle file: {MODEL_PATH}")
            loaded = load_model(MODEL_PATH)
            feature_names = loaded.feature_names

            # Cek apakah model punya atribut 'categories' (model lama tidak punya)
            has_categories = hasattr(loaded, 'categories') and loaded.categories is not None
            if not has_categories:
                print("[WARN] Loaded model does NOT have categories attribute. Retraining from CSV...")
                os.remove(MODEL_PATH)
            else:
                model = loaded
                num_items = len(model.y) if model.y is not None else 0
                print(f"[OK] MODEL LOADED from pickle with {num_items} food items")
                print(f"[OK] Features: {feature_names}")
                print(f"[OK] Categories available: YES ({len(model.categories)} items)")
                print("\n[INFO] Model will be REUSED for ALL incoming requests")
                return True

        # Jika pickle tidak ada atau perlu retrain, train dari CSV
        print(f"\n[STEP 1] Training model from CSV: {CSV_PATH}")
        X, y, feature_names, categories = load_nutrition_csv(CSV_PATH)
        model = NutritionKNN(k=5)
        model.fit(X, y, feature_names, categories)
        print(f"[OK] KNN Model trained successfully with {len(X)} food items")
        if categories:
            unique_cats = list(set(categories))
            print(f"[OK] Categories found: {unique_cats}")
        
        # Simpan model ke pickle untuk next startup
        print(f"\n[STEP 2] Saving model to pickle file: {MODEL_PATH}")
        model.save_model(MODEL_PATH)
        print(f"[OK] Model saved for faster startup next time")
        print("\n[INFO] Model will be REUSED for ALL incoming requests")
        print("="*70 + "\n")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error initializing model: {str(e)}")
        print("="*70 + "\n")
        return False


@app.route('/api/knn/health', methods=['GET'])
def health():
    """Health check endpoint"""
    global model
    model_status = "[OK] LOADED AND READY" if model is not None else "[ERROR] NOT LOADED"
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_status": model_status,
        "feature_names": feature_names or []
    })


@app.route('/api/knn/recommend', methods=['POST'])
def recommend():
    """
    Rekomendasi makanan berdasarkan kebutuhan nutrisi user
    LEVEL 1 Implementation: Weighted KNN Distance dengan prioritas energi
    
    Request body:
    {
        "nutrients": [energi_kal, protein_g, lemak_g, karbohidrat_g],  // 4 features untuk weighted
        "k": 7,
        "distance_metric": "euclidean",
        "use_weighted": true  // Optional, default true jika 4 features
    }
    
    Backward Compatible:
    {
        "nutrients": [protein_g, carbohydrate_g, fat_g],  // 3 features untuk unweighted
        "k": 5
    }
    """
    try:
        if model is None:
            return jsonify({
                "success": False,
                "message": "Model not loaded"
            }), 500
        
        # Model sudah di-load saat server startup, cukup gunakan
        # Tidak ada re-loading yang terjadi di setiap request!

        data = request.get_json()
        
        if not data or 'nutrients' not in data:
            return jsonify({
                "success": False,
                "message": "Missing 'nutrients' field"
            }), 400
        
        nutrients = data.get('nutrients')
        k = data.get('k', 7)
        distance_metric = data.get('distance_metric', 'euclidean')
        distance_metric = data.get('distance_metric', 'euclidean')
        
        # Validate input - accept 3 features
        if not isinstance(nutrients, list) or len(nutrients) != 3:
            return jsonify({
                "success": False,
                "message": f"Nutrients should be a list of 3 values. Got {len(nutrients)} values."
            }), 400
        
        # Get recommendations
        user_nutrients = np.array(nutrients, dtype=float)
        recommendations = model.recommend(user_nutrients, k=k, distance_metric=distance_metric)
        
        # Add mode info to response
        mode_info = "UNWEIGHTED (Standard Euclidean)"
        
        return jsonify({
            "success": True,
            "mode": mode_info,
            "feature_count": len(nutrients),
            "user_requirements": dict(zip(feature_names, nutrients)),
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/api/knn/predict', methods=['POST'])
def predict():
    """
    Prediksi makanan berdasarkan fitur nutrisi
    
    Request body:
    {
        "features": [[protein, carbs, fat], ...],
        "distance_metric": "euclidean"  # optional
    }
    """
    try:
        if model is None:
            return jsonify({
                "success": False,
                "message": "Model not loaded"
            }), 500

        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                "success": False,
                "message": "Missing 'features' field"
            }), 400
        
        features = data.get('features')
        distance_metric = data.get('distance_metric', 'euclidean')
        
        # Validate input
        if not isinstance(features, list) or len(features) == 0:
            return jsonify({
                "success": False,
                "message": "Features should be a non-empty list"
            }), 400
        
        # Get predictions
        predictions = model.predict(features, distance_metric=distance_metric)
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "count": len(predictions)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/api/knn/profile', methods=['POST'])
def get_profile():
    """
    Analisis profil nutrisi user
    
    Request body:
    {
        "nutrients": [protein, carbohydrate, fat]
    }
    """
    try:
        if model is None:
            return jsonify({
                "success": False,
                "message": "Model not loaded"
            }), 500

        data = request.get_json()
        
        if not data or 'nutrients' not in data:
            return jsonify({
                "success": False,
                "message": "Missing 'nutrients' field"
            }), 400
        
        nutrients = data.get('nutrients')
        
        # Validate input
        if not isinstance(nutrients, list) or len(nutrients) != 4:
            return jsonify({
                "success": False,
                "message": "Nutrients should be a list of 4 values"
            }), 400
        
        user_nutrients = np.array(nutrients, dtype=float)
        profile = model.get_nutritional_profile(user_nutrients)
        
        return jsonify({
            "success": True,
            "profile": profile
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/api/knn/features', methods=['GET'])
def get_features():
    """Get list of features/nutrients"""
    return jsonify({
        "success": True,
        "features": feature_names or []
    })


# ==================== NUTRITION DATA ENDPOINTS ====================

@app.route('/api/nutrition', methods=['GET'])
def get_nutrition_data():
    """Get all nutrition data from CSV"""
    global nutrition_df
    
    try:
        # Load CSV jika belum dimuat - dengan delimiter semicolon
        if nutrition_df is None:
            nutrition_df = pd.read_csv(CSV_PATH, lineterminator='\n')
            # Clean up column names - remove any trailing whitespace/\r
            nutrition_df.columns = nutrition_df.columns.str.strip()
            # Clean up data - remove leading/trailing whitespace from string columns
            for col in nutrition_df.select_dtypes(include=['object']).columns:
                nutrition_df[col] = nutrition_df[col].str.strip().str.replace(r'[\r\n]', '', regex=True)
            print(f"[OK] Loaded {len(nutrition_df)} nutrition items from CSV")
            print(f"[OK] CSV columns: {list(nutrition_df.columns)}")
        
        # Transform ke format yang diinginkan
        data = []
        for idx, row in nutrition_df.iterrows():
            # Clean numeric fields - remove any leftover CR/LF
            def clean_numeric(val):
                if pd.isna(val):
                    return 0.0
                s = str(val).strip().replace('\r', '').replace('\n', '')
                try:
                    return float(s)
                except:
                    return 0.0
            
            item = {
                'kode': str(row.get('kode', '')).strip(),
                'nama_bahan': str(row.get('nama_bahan', '')).strip(),
                'energi_kal': clean_numeric(row.get('energi_kal', 0)),
                'protein_g': clean_numeric(row.get('protein_g', 0)),
                'lemak_g': clean_numeric(row.get('lemak_g', 0)),
                'karbohidrat_g': clean_numeric(row.get('karbohidrat_g', 0)),
                'kategori': str(row.get('kategori', 'all')).strip(),
            }
            data.append(item)
        
        return jsonify({
            'success': True,
            'count': len(data),
            'data': data
        })
    except Exception as e:
        print(f"[ERROR] Error loading nutrition data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/nutrition/search', methods=['GET'])
def search_nutrition_data():
    """Search nutrition data by name"""
    global nutrition_df
    
    try:
        query = request.args.get('query', '').lower().strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter is required'
            }), 400
        
        # Load CSV jika belum dimuat
        if nutrition_df is None:
            nutrition_df = pd.read_csv(CSV_PATH, lineterminator='\n')
            # Clean up column names
            nutrition_df.columns = nutrition_df.columns.str.strip()
            # Clean up data
            for col in nutrition_df.select_dtypes(include=['object']).columns:
                nutrition_df[col] = nutrition_df[col].str.strip()
        
        # Search by nama_bahan
        filtered = nutrition_df[
            nutrition_df['nama_bahan'].str.lower().str.contains(query, na=False)
        ]
        
        # Transform ke format yang diinginkan
        data = []
        for idx, row in filtered.iterrows():
            item = {
                'kode': str(row.get('kode', '')).strip(),
                'nama_bahan': str(row.get('nama_bahan', '')).strip(),
                'energi_kal': float(row.get('energi_kal', 0)) if pd.notna(row.get('energi_kal')) else 0,
                'protein_g': float(row.get('protein_g', 0)) if pd.notna(row.get('protein_g')) else 0,
                'lemak_g': float(row.get('lemak_g', 0)) if pd.notna(row.get('lemak_g')) else 0,
                'karbohidrat_g': float(row.get('karbohidrat_g', 0)) if pd.notna(row.get('karbohidrat_g')) else 0,
            }
            data.append(item)
        
        return jsonify({
            'success': True,
            'count': len(data),
            'data': data
        })
    except Exception as e:
        print(f"[ERROR] Error searching nutrition data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "Endpoint not found"
    }), 404


def calculate_target_nutrients(user_profile: dict) -> dict:
    """
    Menghitung target nutrisi berdasarkan profil pengguna dan tujuan gizi
    Fokus pada KALORI PER MAKAN, bukan target kalori per hari
    
    Args:
        user_profile: Dictionary dengan user info dan calorie target
        
    Returns:
        Dictionary dengan target nutrients (protein, carbs, fat) untuk 1 makan
    """
    # Gunakan calories_per_meal jika ada, jika tidak hitung dari target_calories
    if 'calories_per_meal' in user_profile:
        target_calories = user_profile.get('calories_per_meal')
    else:
        target_calories = user_profile.get('target_calories', 2000)
        # Jika masih nilai besar (> 500), berarti kalori per hari, bagi 4
        if target_calories > 500:
            target_calories = target_calories / 4
    
    tujuan = user_profile.get('tujuan', 'seimbang')
    
    # Kalori per gram untuk setiap nutrisi
    CAL_PER_PROTEIN = 4
    CAL_PER_CARB = 4
    CAL_PER_FAT = 9
    
    # Hitung distribusi nutrisi berdasarkan tujuan
    # Menggunakan pendekatan persen kalori
    if tujuan == 'naik':  # Menaikkan berat badan
        # Lebih banyak energi, focus pada fat dan carbs
        protein_pct = 0.15  # 15% kalori dari protein
        carb_pct = 0.55     # 55% kalori dari karbohidrat
        fat_pct = 0.30      # 30% kalori dari lemak
    elif tujuan == 'turun':  # Menurunkan berat badan
        # Lebih banyak protein, kurangi fat
        protein_pct = 0.35  # 35% kalori dari protein
        carb_pct = 0.45     # 45% kalori dari karbohidrat
        fat_pct = 0.20      # 20% kalori dari lemak
    else:  # seimbang
        # Balanced distribution
        protein_pct = 0.25  # 25% kalori dari protein
        carb_pct = 0.50     # 50% kalori dari karbohidrat
        fat_pct = 0.25      # 25% kalori dari lemak
    
    # Hitung gram untuk setiap nutrisi
    protein_g = (target_calories * protein_pct) / CAL_PER_PROTEIN
    carbed_g = (target_calories * carb_pct) / CAL_PER_CARB
    fat_g = (target_calories * fat_pct) / CAL_PER_FAT
    
    return {
        "protein_g": protein_g,
        "karbohidrat_g": carbed_g,
        "lemak_g": fat_g,
        "target_calories_per_meal": target_calories,
        "target_distribution": {
            "protein_pct": protein_pct * 100,
            "carb_pct": carb_pct * 100,
            "fat_pct": fat_pct * 100
        }
    }


def apply_feature_weights(nutrients: np.ndarray, user_profile: dict, feature_names: list) -> np.ndarray:
    """
    Menerapkan bobot fitur berdasarkan tujuan pengguna
    Mengubah nutrient profile sesuai prioritas goal
    
    Args:
        nutrients: Array nutrisi [protein, carbs, fat]
        user_profile: Dictionary dengan informasi tujuan
        feature_names: Nama fitur nutrisi
        
    Returns:
        Array nutrisi dengan bobot yang diterapkan
    """
    tujuan = user_profile.get('tujuan', 'seimbang')
    weighted = nutrients.copy().astype(float)
    
    # Tentukan bobot berdasarkan tujuan untuk setiap nutrisi
    # Format weights sesuai dengan feature_names yang ada di CSV: ['protein_g', 'lemak_g', 'karbohidrat_g']
    if tujuan == 'naik':  # Menaikkan BB - prioritas energi
        # Bobot untuk [protein_g, lemak_g, karbohidrat_g]
        weights = np.array([0.15, 0.30, 0.55])  # Lemak dan karbo lebih penting
    elif tujuan == 'turun':  # Menurunkan BB - prioritas protein
        weights = np.array([0.55, 0.10, 0.35])  # Protein lebih penting, lemak minimal
    else:  # seimbang - balanced bobot
        weights = np.array([0.25, 0.25, 0.50])  # Seimbang
    
    # Terapkan bobot
    weighted = nutrients * weights
    
    return weighted


@app.route('/api/recommend', methods=['POST'])
def recommend_from_profile():
    """
    Rekomendasi makanan berdasarkan profil user dari Node.js backend
    Endpoint ini untuk backward compatibility dengan Node.js controller
    
    Request body:
    {
        "user_profile": {
            "usia": 25,
            "jenis_kelamin": "pria",
            "bmi": 26.1,
            "tdee": 2500,
            "target_calories": 2125,
            "aktivitas": "sedang",
            "tujuan": "turun",
            "kategori": "all"
        },
        "k": 10
    }
    """
    try:
        if model is None:
            return jsonify({
                "success": False,
                "recommendations": [],
                "message": "Model not loaded"
            }), 500

        data = request.get_json()
        
        if not data or 'user_profile' not in data:
            return jsonify({
                "success": False,
                "recommendations": [],
                "message": "Missing 'user_profile' field"
            }), 400
        
        user_profile = data.get('user_profile')
        k = data.get('k', 10)
        distance_metric = data.get('distance_metric', 'euclidean')
        
        # Validate user_profile
        if not isinstance(user_profile, dict):
            return jsonify({
                "success": False,
                "recommendations": [],
                "message": "user_profile should be a dictionary"
            }), 400
        
        # Calculate target nutrients based on user profile
        # Menggunakan calories_per_meal untuk fokus pada nutrisi per makan
        target_nutrients_calc = calculate_target_nutrients(user_profile)
        
        # Create nutrient array [protein, carbs, fat] matching feature order in CSV
        # CSV features are: ['protein_g', 'lemak_g', 'karbohidrat_g']
        user_nutrients = np.array([
            target_nutrients_calc['protein_g'],
            target_nutrients_calc['lemak_g'],
            target_nutrients_calc['karbohidrat_g']
        ], dtype=float)
        
        calories_per_meal = user_profile.get('calories_per_meal', user_profile.get('target_calories', 2000) / 4)
        
        print(f"[/api/recommend] Calories per meal: {calories_per_meal} kal")
        print(f"[/api/recommend] Target nutrients per meal: {user_nutrients}")
        print(f"[/api/recommend] Total calories per day: {user_profile.get('target_calories')}")
        print(f"[/api/recommend] Tujuan: {user_profile.get('tujuan')}")
        kategori_filter = user_profile.get('kategori', 'all')
        print(f"[/api/recommend] Kategori filter: '{kategori_filter}'")
        has_cats = hasattr(model, 'categories') and model.categories is not None
        print(f"[/api/recommend] Model has categories: {has_cats}")
        
        # Get recommendations from KNN model
        recommendations = model.recommend(
            user_nutrients, 
            k=k,
            distance_metric=distance_metric,
            category_filter=user_profile.get('kategori', 'all')
        )
        
        # Enhance recommendations dengan additional data
        for rec in recommendations:
            rec['user_target'] = {
                'calories_total_per_day': user_profile.get('target_calories'),
                'calories_per_meal': calories_per_meal,
                'meals_per_day': user_profile.get('meals_per_day', 4),
                'protein_g': target_nutrients_calc['protein_g'],
                'lemak_g': target_nutrients_calc['lemak_g'],
                'karbohidrat_g': target_nutrients_calc['karbohidrat_g'],
                'tujuan': user_profile.get('tujuan')
            }
        
        return jsonify({
            "success": True,
            "user_profile": user_profile,
            "calories_per_meal": calories_per_meal,
            "meals_per_day": user_profile.get('meals_per_day', 4),
            "target_nutrients": target_nutrients_calc,
            "recommendation_note": f"Rekomendasi untuk 1 makan (~{calories_per_meal} kal)",
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    
    except Exception as e:
        print(f"[ERROR] Error in /api/recommend: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "recommendations": [],
            "message": str(e)
        }), 500


@app.route('/api/gizi', methods=['GET'])
def get_gizi_data():
    """
    Endpoint untuk mendapatkan data gizi dari CSV
    Sesuai dengan request dari frontend JS
    """
    try:
        # Load nutrition data
        global nutrition_df
        
        if nutrition_df is None:
            nutrition_df = pd.read_csv(CSV_PATH, lineterminator='\n', encoding='latin-1')
            nutrition_df.columns = nutrition_df.columns.str.strip()
            
            # Konversi kolom numerik
            numeric_cols = ['energi_kal', 'protein_g', 'lemak_g', 'karbohidrat_g']
            for col in numeric_cols:
                if col in nutrition_df.columns:
                    nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce')
        
        # Transform ke format yang diinginkan
        data = []
        for idx, row in nutrition_df.iterrows():
            item = {
                'nama_bahan': str(row.get('nama_bahan', '')).strip(),
                'energi_kal': float(row.get('energi_kal', 0)) if pd.notna(row.get('energi_kal')) else 0,
                'protein_g': float(row.get('protein_g', 0)) if pd.notna(row.get('protein_g')) else 0,
                'lemak_g': float(row.get('lemak_g', 0)) if pd.notna(row.get('lemak_g')) else 0,
                'karbohidrat_g': float(row.get('karbohidrat_g', 0)) if pd.notna(row.get('karbohidrat_g')) else 0,
            }
            data.append(item)
        
        return jsonify({
            'success': True,
            'count': len(data),
            'data': data
        })
    
    except Exception as e:
        print(f"[ERROR] Error in get_gizi_data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "Internal server error"
    }), 500


# Inisialisasi model saat module pertama kali di-load
# Berlaku baik untuk `python app.py` maupun saat gunicorn import module ini
init_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, port=port, host='0.0.0.0')
