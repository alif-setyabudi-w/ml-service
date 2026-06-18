"""
K-Nearest Neighbors (KNN) Algorithm for Nutrition Recommendation System
Sistem rekomendasi makanan berdasarkan kebutuhan nutrisi pengguna
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import math
import pickle
import os


class NutritionKNN:
    """
    K-Nearest Neighbors classifier untuk rekomendasi makanan berdasarkan nutrisi
    """

    def __init__(self, k: int = 7):
        """
        Inisialisasi KNN model
        
        Args:
            k (int): Jumlah tetangga terdekat yang akan dipertimbangkan (optimal = 7 untuk dataset 1125 items)
        """
        self.k = k
        self.X = None  # Feature data (nutrient values)
        self.y = None  # Labels (food names)
        self.feature_names = None
        self.categories = None  # Labels (categories)

    def fit(self, X: np.ndarray, y: List[str], feature_names: List[str] = None, categories: List[str] = None):
        """
        Fit model dengan training data
        
        Args:
            X: Array of shape (n_samples, n_features) dengan nilai nutrisi
            y: List of food names
            feature_names: Nama-nama fitur (nutrient names)
            categories: List of food categories
        """
        self.X = np.array(X, dtype=float)
        self.y = np.array(y)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]
        self.categories = np.array(categories) if categories is not None else np.array(["all"] * len(y))
        print(f"Model fitted with {len(self.X)} food items")

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Hitung Euclidean distance antara dua titik
        
        d = sqrt((x1-x2)^2 + (y1-y2)^2 + ...)
        """
        return math.sqrt(np.sum((x1 - x2) ** 2))

    def manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Hitung Manhattan distance (L1 distance) antara dua titik
        
        d = |x1-x2| + |y1-y2| + ...
        """
        return np.sum(np.abs(x1 - x2))

    def weighted_euclidean_distance(self, x1: np.ndarray, x2: np.ndarray, 
                                    weights: np.ndarray = None) -> float:
        """
        Hitung Weighted Euclidean distance dengan feature weighting
        Untuk prioritas kalori dalam rekomendasi nutrisi
        
        Features: [energi, protein, lemak, karbohidrat] (jika 4 features)
                  [protein, carbs, fat] (jika 3 features - backward compatible)
        
        Default weights untuk 4 features: [0.50, 0.15, 0.15, 0.20]
                                           [energi, protein, fat, carbs]
        
        Args:
            x1: User nutrients array
            x2: Food nutrients array
            weights: Feature weights. Default: auto-set based on feature count
            
        Returns:
            float: Weighted Euclidean distance
        """
        # Set default weights based on feature count
        if weights is None:
            if len(x1) == 4:
                # Features: [energi, protein, lemak, karbohidrat]
                weights = np.array([0.50, 0.15, 0.15, 0.20])
            elif len(x1) == 3:
                # Features: [protein, carbs, fat] - backward compatible (unweighted)
                weights = np.array([0.333, 0.333, 0.334])
            else:
                # Default: equal weights
                weights = np.ones(len(x1)) / len(x1)
        
        weights = np.array(weights)
        weighted_diff = weights * (x1 - x2)
        return math.sqrt(np.sum(weighted_diff ** 2))

    def weighted_manhattan_distance(self, x1: np.ndarray, x2: np.ndarray,
                                    weights: np.ndarray = None) -> float:
        """
        Hitung Weighted Manhattan distance dengan feature weighting
        
        Default weights untuk 4 features: [0.50, 0.15, 0.15, 0.20]
        """
        if weights is None:
            if len(x1) == 4:
                weights = np.array([0.50, 0.15, 0.15, 0.20])
            elif len(x1) == 3:
                weights = np.array([0.333, 0.333, 0.334])
            else:
                weights = np.ones(len(x1)) / len(x1)
        
        weights = np.array(weights)
        weighted_diff = weights * np.abs(x1 - x2)
        return np.sum(weighted_diff)

    def cosine_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Hitung Cosine similarity antara dua titik
        Menggunakan dot product dibagi dengan magnitude vectors
        
        Hasil antara -1 dan 1, dimana 1 artinya identical direction
        """
        # Hindari division by zero
        magnitude_x1 = np.linalg.norm(x1)
        magnitude_x2 = np.linalg.norm(x2)
        
        if magnitude_x1 == 0 or magnitude_x2 == 0:
            return 0
        
        dot_product = np.dot(x1, x2)
        cosine_sim = dot_product / (magnitude_x1 * magnitude_x2)
        
        # Return nilai positif (convert dari -1,1 range ke 0,1 range)
        return (cosine_sim + 1) / 2

    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalisasi fitur menggunakan Min-Max scaling
        
        X_normalized = (X - X_min) / (X_max - X_min)
        """
        X_min = np.min(self.X, axis=0)
        X_max = np.max(self.X, axis=0)
        
        # Hindari division by zero
        denominator = X_max - X_min
        denominator[denominator == 0] = 1
        
        return (X - X_min) / denominator

    def predict(self, X_query: np.ndarray, distance_metric: str = "euclidean") -> List[str]:
        """
        Prediksi label untuk query data
        
        Args:
            X_query: Array of shape (n_queries, n_features)
            distance_metric: 'euclidean' atau 'manhattan'
            
        Returns:
            List of predicted labels
        """
        # Normalisasi both training dan query data
        X_train_normalized = self.normalize_features(self.X)
        X_query_normalized = self.normalize_features(np.array(X_query, dtype=float))

        predictions = []

        for query in X_query_normalized:
            # Hitung distance ke semua training points
            distances = []
            for i, train_point in enumerate(X_train_normalized):
                if distance_metric == "euclidean":
                    dist = self.euclidean_distance(query, train_point)
                elif distance_metric == "manhattan":
                    dist = self.manhattan_distance(query, train_point)
                else:
                    raise ValueError(f"Unknown distance metric: {distance_metric}")

                distances.append((dist, self.y[i]))

            # Sort by distance dan ambil k terdekat
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[: self.k]

            # Hitung voting untuk prediksi
            food_votes = {}
            for dist, food in k_nearest:
                food_votes[food] = food_votes.get(food, 0) + 1

            # Prediksi = yang paling sering muncul
            predicted = max(food_votes, key=food_votes.get)
            predictions.append(predicted)

        return predictions

    def recommend(self, user_nutrients: np.ndarray, k: int = None, 
                  distance_metric: str = "euclidean", category_filter: str = None,
                  use_weighted: bool = True) -> List[Dict]:
        """
        Rekomendasi top-k makanan yang paling mirip dengan kebutuhan nutrisi user
        LEVEL 1 Implementation: Menggunakan Weighted Distance dengan prioritas energi
        
        Args:
            user_nutrients: Array nutrisi user
                           Jika 4 features: [energi_kal, protein, lemak, karbohidrat]
                           Jika 3 features: [protein, carbs, fat]
            k: Jumlah rekomendasi (default: self.k)
            distance_metric: 'euclidean' atau 'manhattan'
            category_filter: Filter berdasarkan kategori makanan
            use_weighted: True untuk gunakan weighted distance (LEVEL 1), False untuk unweighted
            
        Returns:
            List of dictionaries dengan informasi makanan yang direkomendasikan
        """
        k = k or self.k
        
        # Validasi feature count
        n_features = len(user_nutrients)
        if n_features not in [3, 4]:
            raise ValueError(f"Expected 3 or 4 features, got {n_features}")
        
        # Normalisasi user nutrients
        X_train_normalized = self.normalize_features(self.X)
        user_normalized = self.normalize_features(np.array([user_nutrients]))[0]
        user_original = np.array(user_nutrients, dtype=float)

        # Hitung distance dan similarity ke semua food items
        distances = []
        max_distance = 0
        min_distance = float('inf')
        
        # Set weights berdasarkan feature count
        if use_weighted and n_features == 4:
            # LEVEL 1: Weighted distance dengan energi sebagai prioritas
            weights = np.array([0.50, 0.15, 0.15, 0.20])  # [energy, protein, fat, carbs]
            print("[INFO] Using WEIGHTED distance (LEVEL 1): Energy 50%, Protein 15%, Fat 15%, Carbs 20%")
        else:
            # Backward compatible: unweighted distance
            weights = np.ones(n_features) / n_features
            print(f"[INFO] Using UNWEIGHTED distance (equal weights for {n_features} features)")
        
        # First pass: collect distances dan find min/max untuk normalisasi
        temp_distances = []
        for i, food_nutrients in enumerate(X_train_normalized):
            if distance_metric == "euclidean":
                if use_weighted:
                    dist = self.weighted_euclidean_distance(user_normalized, food_nutrients, weights)
                else:
                    dist = self.euclidean_distance(user_normalized, food_nutrients)
            elif distance_metric == "manhattan":
                if use_weighted:
                    dist = self.weighted_manhattan_distance(user_normalized, food_nutrients, weights)
                else:
                    dist = self.manhattan_distance(user_normalized, food_nutrients)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            
            # Hitung cosine similarity menggunakan data original (bukan normalized)
            food_original = np.array(self.X[i], dtype=float)
            cosine_sim = self.cosine_similarity(user_original, food_original)
            
            item_category = self.categories[i] if hasattr(self, 'categories') and self.categories is not None else "all"

            # Check category filter (only process if matches filter)
            if category_filter and category_filter.lower() != "all" and category_filter.lower() != "semua kategori":
                if not isinstance(item_category, str) or item_category.lower() != category_filter.lower():
                    continue

            temp_distances.append({
                "index": i,
                "distance": dist,
                "cosine_sim": cosine_sim,
                "kategori": item_category
            })
            
            max_distance = max(max_distance, dist)
            min_distance = min(min_distance, dist)
        
        # Normalize distances untuk similarity score
        distance_range = max_distance - min_distance if max_distance > min_distance else 1
        
        # Second pass: hitung final similarity score dengan kombinasi metrics
        for item in temp_distances:
            i = item["index"]
            dist = item["distance"]
            cosine_sim = item["cosine_sim"]
            
            # Normalized distance similarity (0 to 1, dimana 1 adalah paling mirip)
            # Min distance -> similarity = 1, max distance -> similarity = 0
            distance_sim = 1 - ((dist - min_distance) / distance_range)
            
            # Kombinasi dari distance-based similarity dan cosine similarity
            # Weight: 60% distance-based, 40% cosine similarity
            combined_similarity = (distance_sim * 0.6) + (cosine_sim * 0.4)
            
            # Normalize ke 0-100 percent
            similarity_percent = combined_similarity * 100
            
            distances.append({
                "rank": len(distances) + 1,
                "name": self.y[i],
                "kategori": item["kategori"],
                "distance": dist,
                "cosine_similarity": round(cosine_sim, 4),
                "distance_similarity": round(distance_sim, 4),
                "nutrients": {
                    name: self.X[i][j] 
                    for j, name in enumerate(self.feature_names)
                },
                "similarity_score": round(combined_similarity, 4),
                "similarity_percent": round(similarity_percent, 1)
            })

        # Sort by similarity (descending) dan ambil top k
        distances.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_k = distances[:k]

        # Update ranking
        for idx, item in enumerate(top_k):
            item["rank"] = idx + 1

        return top_k

    def get_nutritional_profile(self, user_nutrients: np.ndarray) -> Dict:
        """
        Analisis profil nutrisi user dan bandingkan dengan rata-rata dataset
        
        Args:
            user_nutrients: Array nutrisi user
            
        Returns:
            Dictionary dengan analisis profil
        """
        avg_nutrients = np.mean(self.X, axis=0)
        
        profile = {
            "user_nutrients": {
                name: user_nutrients[i] 
                for i, name in enumerate(self.feature_names)
            },
            "average_nutrients": {
                name: avg_nutrients[i] 
                for i, name in enumerate(self.feature_names)
            },
            "differences": {
                name: user_nutrients[i] - avg_nutrients[i]
                for i, name in enumerate(self.feature_names)
            }
        }
        
        return profile

    def save_model(self, filepath: str) -> bool:
        """
        Simpan model ke file pickle
        
        Args:
            filepath: Path untuk menyimpan file model.pkl
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"[OK] Model saved successfully to {filepath}")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving model: {str(e)}")
            return False


def load_nutrition_csv(filepath: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    try:
        # Baca CSV dengan delimiter semicolon dan handle line endings
        # Use lineterminator='\n' untuk handle CRLF line endings properly
        df = pd.read_csv(
            filepath,
            lineterminator='\n'
        )
        
        # Clean up column names - remove any trailing whitespace/\r
        df.columns = df.columns.str.strip()
        
        # Clean up data - remove leading/trailing whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        print(f"[OK] CSV loaded successfully. Shape: {df.shape}")
        print(f"[OK] Columns: {list(df.columns)}")
        
        # Extract features - dengan energi sebagai feature pertama (LEVEL 1 Weighted KNN)
        # Priority: Include energy jika ada untuk weighted distance calculation
        feature_cols = []
        feature_names_list = []
        
        if 'energi_kal' in df.columns:
            feature_cols.append('energi_kal')
            feature_names_list.append('energi_kal')
            print("[OK] Energy column found - will use weighted distance (LEVEL 1)")
        
        feature_cols.extend(['protein_g', 'lemak_g', 'karbohidrat_g'])
        feature_names_list.extend(['protein_g', 'lemak_g', 'karbohidrat_g'])
        
        # Verify kolom ada
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"[ERROR] Missing columns: {missing_cols}")
            print(f"[ERROR] Available columns: {list(df.columns)}")
            # Fallback: gunakan tanpa energi
            feature_cols = ['protein_g', 'lemak_g', 'karbohidrat_g']
            feature_names_list = ['protein_g', 'lemak_g', 'karbohidrat_g']
            print("[WARN] Falling back to 3 features (no energy)")
        
        # Convert features to float and handle any non-numeric values
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        X = df[feature_cols].fillna(0).values.astype(float)
        
        # Extract food names - gunakan nama_bahan
        y = df['nama_bahan'].values.tolist()
        
        # Extract categories if exists
        categories = df['kategori'].fillna('all').values.tolist() if 'kategori' in df.columns else None
        
        print(f"[OK] Loaded {len(X)} food items with {len(feature_cols)} features: {feature_names_list}")
        return X, y, feature_names_list, categories
    except Exception as e:
        print(f"[ERROR] Error loading CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_model(model: NutritionKNN, X_test: np.ndarray, 
                  y_test: List[str], k_values: List[int] = None) -> Dict:
    """
    Evaluasi performa model dengan berbagai nilai k
    
    Args:
        model: KNN model yang sudah di-fit
        X_test: Test features
        y_test: Test labels
        k_values: List of k values untuk dievaluasi
        
    Returns:
        Dictionary dengan hasil evaluasi
    """
    k_values = k_values or [3, 5, 7, 9]
    results = {}

    for k in k_values:
        model.k = k
        predictions = model.predict(X_test)
        
        # Hitung accuracy
        correct = sum(1 for pred, actual in zip(predictions, y_test) if pred == actual)
        accuracy = correct / len(y_test) if y_test else 0
        
        results[k] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(y_test)
        }

    return results

def load_model(filepath: str) -> 'NutritionKNN':
    """
    Load model dari file pickle
    
    Args:
        filepath: Path ke file model.pkl
        
    Returns:
        NutritionKNN: Model yang sudah di-load
        
    Raises:
        FileNotFoundError: Jika file tidak ditemukan
        Exception: Jika ada error saat loading
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] Model loaded successfully from {filepath}")
        return model
    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}")
        raise
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        raise
