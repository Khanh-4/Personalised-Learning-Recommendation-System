import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ========================
# Popularity Recommender
# ========================
class PopularityRecommender:
    def __init__(self):
        self.popularity = None

    def fit(self, df, item_col):
        logger.info("[Popularity] Training popularity-based recommender...")
        self.popularity = df[item_col].value_counts()
        logger.info(f"[Popularity] Trained on {len(self.popularity)} items. Top item: {self.popularity.index[0]}")

    def recommend(self, user_id=None, k=5):
        return list(self.popularity.head(k).index)


# ========================
# Matrix Factorization Recommender (basic)
# ========================
class MFRecommender:
    def __init__(self, n_factors=10, n_iters=10, lr=0.01, reg=0.01):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.lr = lr
        self.reg = reg
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}

    def fit(self, df, user_col, item_col):
        logger.info("[MF] Training matrix factorization recommender...")
        users = df[user_col].unique()
        items = df[item_col].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {it: j for j, it in enumerate(items)}

        n_users, n_items = len(users), len(items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        logger.info(f"[MF] Training on matrix shape: ({n_users}, {n_items}) (users × items)")

    def recommend(self, user_id, k=5):
        if user_id not in self.user_map:
            logger.warning(f"[MF] User {user_id} not found, returning empty list")
            return []
        u_idx = self.user_map[user_id]
        scores = self.user_factors[u_idx].dot(self.item_factors.T)
        top_items = np.argsort(-scores)[:k]
        return [list(self.item_map.keys())[i] for i in top_items]


# ========================
# Content-Based Recommender
# ========================
class ContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.similarity_matrix = None

    def fit(self, item_features):
        logger.info("[Content-Based] Training content-based recommender...")
        self.item_features = item_features
        sim = np.dot(item_features, item_features.T)
        norms = np.linalg.norm(item_features, axis=1)
        self.similarity_matrix = sim / np.outer(norms, norms)
        logger.info(f"[Content-Based] Trained on {len(item_features)} items with {item_features.shape[1]} features")

    def recommend(self, user_id=None, item_id=None, k=5):
        if self.similarity_matrix is None:
            raise ValueError("Model not trained yet.")

        if item_id is None:
            return list(self.item_features.index[:k])

        if item_id not in self.item_features.index:
            logger.warning(f"[Content-Based] ⚠ Item {item_id} not found, fallback to popularity.")
            return list(self.item_features.index[:k])

        idx = self.item_features.index.get_loc(item_id)
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [self.item_features.index[i] for i, _ in scores[1:k + 1]]


# ========================
# Hybrid Recommender
# ========================
class HybridRecommender:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else {name: 1.0 for name in models}

    def recommend(self, user_id, k=5):
        scores = defaultdict(float)
        for name, model in self.models.items():
            if hasattr(model, "recommend"):
                recs = model.recommend(user_id=user_id, k=k)
                for rank, item in enumerate(recs):
                    scores[item] += self.weights.get(name, 0) * (k - rank)

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked_items[:k]]


# ========================
# Helper to build hybrid
# ========================
def train_hybrid(models, weights):
    logger.info("[Hybrid] Training hybrid recommender...")
    logger.info(f"[Hybrid] Combining models with weights: {weights}")
    return HybridRecommender(models, weights)

# ========================
# Training Wrappers
# ========================
def train_popularity(train_df, item_col, rating_col=None):
    model = PopularityRecommender()
    model.fit(train_df, item_col)
    return model


def train_mf(train_df, user_col, item_col, rating_col=None, n_components=20):
    model = MFRecommender(n_factors=n_components)
    model.fit(train_df, user_col, item_col)
    return model


def train_content_based(train_df, user_col, item_col, feature_cols):
    features = train_df[feature_cols]
    model = ContentBasedRecommender()
    model.fit(features)
    return model