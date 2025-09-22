# models.py
# ============================================================
# Baseline models for Personalized Learning Recommender
# ============================================================

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. POPULARITY-BASED MODEL
# ------------------------------------------------------------
class PopularityRecommender:
    def __init__(self, item_col, rating_col=None):
        self.item_col = item_col
        self.rating_col = rating_col
        self.popularity = None

    def fit(self, df):
        if self.rating_col and self.rating_col in df.columns:
            # Popularity = tổng điểm rating
            self.popularity = (
                df.groupby(self.item_col)[self.rating_col]
                .sum()
                .sort_values(ascending=False)
            )
        else:
            # Popularity = số lần xuất hiện
            self.popularity = df[self.item_col].value_counts()

        # 🔹 Log thêm số lượng item
        print(f"[Popularity] Trained on {len(self.popularity)} items. Top item: {self.popularity.index[0]}")
        return self

    def recommend(self, user_id=None, top_k=10):
        return self.popularity.head(top_k).index.tolist()


# ------------------------------------------------------------
# 2. MATRIX FACTORIZATION (SVD)
# ------------------------------------------------------------
class MFRecommender:
    def __init__(self, user_col, item_col, rating_col, n_components=50):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.n_components = n_components
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.rev_user_mapping = {}
        self.rev_item_mapping = {}

    def fit(self, df):
        # Map user/item ids → index
        users = df[self.user_col].unique()
        items = df[self.item_col].unique()
        self.user_mapping = {u: i for i, u in enumerate(users)}
        self.item_mapping = {it: j for j, it in enumerate(items)}
        self.rev_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.rev_item_mapping = {j: it for it, j in self.item_mapping.items()}

        # Build sparse matrix
        row = df[self.user_col].map(self.user_mapping)
        col = df[self.item_col].map(self.item_mapping)
        data = df[self.rating_col] if self.rating_col in df.columns else np.ones(len(df))
        matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))

        # 🔹 Log thêm shape của ma trận
        print(f"[MF] Training on matrix shape: {matrix.shape} (users × items)")

        # 🔹 Auto-fix n_components
        n_items = matrix.shape[1]
        if self.n_components > n_items:
            print(f"⚠️ n_components={self.n_components} > n_items={n_items}. Reducing to {n_items}.")
            self.n_components = n_items

        # SVD decomposition
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_factors = svd.fit_transform(matrix)
        self.item_factors = svd.components_.T
        return self


    def recommend(self, user_id, top_k=10):
        if user_id not in self.user_mapping:
            return []
        user_idx = self.user_mapping[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        top_items_idx = np.argsort(scores)[::-1][:top_k]
        return [self.rev_item_mapping[i] for i in top_items_idx]

# 3. CONTENT-BASED MODEL
# ------------------------------------------------------------
class ContentBasedRecommender:
    def __init__(self, user_col, item_col, feature_cols):
        self.user_col = user_col
        self.item_col = item_col
        self.feature_cols = feature_cols
        self.user_profiles = None
        self.item_profiles = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.rev_item_mapping = {}

    def fit(self, df):
        # Tạo profile cho item
        items = df[self.item_col].unique()
        self.item_mapping = {it: j for j, it in enumerate(items)}
        self.rev_item_mapping = {j: it for it, j in self.item_mapping.items()}

        # Item features
        item_features = df.groupby(self.item_col)[self.feature_cols].mean()
        scaler = StandardScaler()
        self.item_profiles = scaler.fit_transform(item_features)

        # User profile = trung bình các item đã tương tác
        self.user_profiles = {}
        for user, group in df.groupby(self.user_col):
            interacted_items = group[self.item_col].map(self.item_mapping)
            if len(interacted_items) > 0:
                profile = self.item_profiles[interacted_items].mean(axis=0)
                self.user_profiles[user] = profile

        print(f"[Content-Based] Trained on {len(self.item_mapping)} items with {len(self.feature_cols)} features")
        return self

    def recommend(self, user_id, top_k=10):
        if user_id not in self.user_profiles:
            return []
        user_vector = self.user_profiles[user_id].reshape(1, -1)
        scores = cosine_similarity(user_vector, self.item_profiles)[0]
        top_items_idx = np.argsort(scores)[::-1][:top_k]
        return [self.rev_item_mapping[i] for i in top_items_idx]

# ------------------------------------------------------------
# 4. HYBRID MODEL
# ------------------------------------------------------------
class HybridRecommender:
    def __init__(self, models, weights=None):
        """
        models: dict { "pop": model1, "mf": model2, "cb": model3 }
        weights: dict { "pop": 0.3, "mf": 0.4, "cb": 0.3 }
        """
        self.models = models
        self.weights = weights if weights else {k: 1.0 for k in models}

    def recommend(self, user_id, top_k=10):
        scores = {}
        for name, model in self.models.items():
            if hasattr(model, "recommend"):
                recs = model.recommend(user_id=user_id, top_k=top_k * 2)  # lấy nhiều hơn để gộp
                for rank, item in enumerate(recs):
                    score = (top_k - rank) / top_k  # điểm dựa trên rank
                    scores[item] = scores.get(item, 0) + score * self.weights.get(name, 1.0)

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked_items[:top_k]]


# ------------------------------------------------------------
# WRAPPER FUNCTIONS
# ------------------------------------------------------------
def train_popularity(train_df, item_col, rating_col=None):
    print("[Popularity] Training popularity-based recommender...")
    model = PopularityRecommender(item_col=item_col, rating_col=rating_col)
    return model.fit(train_df)

def train_mf(train_df, user_col, item_col, rating_col, n_components=50):
    print("[MF] Training matrix factorization recommender...")
    model = MFRecommender(user_col, item_col, rating_col, n_components=n_components)
    return model.fit(train_df)

def train_content_based(train_df, user_col, item_col, feature_cols):
    print("[Content-Based] Training content-based recommender...")
    model = ContentBasedRecommender(user_col, item_col, feature_cols)
    return model.fit(train_df)

def train_hybrid(models, weights=None):
    print("[Hybrid] Combining models with weights:", weights)
    return HybridRecommender(models, weights)
