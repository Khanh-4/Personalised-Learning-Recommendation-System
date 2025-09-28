import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ========================
# Popularity Recommender
# ========================
class PopularityRecommender:
    def __init__(self):
        self.popularity = None

    def fit(self, df, item_col="content_type", rating_col="engagement_score"):
        self.popularity = (
            df.groupby(item_col)[rating_col]
            .sum()
            .sort_values(ascending=False)
        )
        print(f"[PopularityRecommender] Trained on {len(self.popularity)} items")

    def recommend(self, user_id=None, top_k=None, k=None):
        if k is not None:
            top_k = k
        if top_k is None:
            top_k = 10
        return list(self.popularity.head(top_k).index)


# ========================
# MF Recommender
# ========================
class MFRecommender:
    def __init__(self, num_factors=50, learning_rate=0.01, reg=0.01, epochs=20):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}

    def fit(self, df, user_col="learner_id", item_col="content_type", rating_col="engagement_score"):
        users = df[user_col].unique()
        items = df[item_col].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {it: j for j, it in enumerate(items)}

        n_users, n_items = len(users), len(items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.num_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.num_factors))

        data = df[[user_col, item_col, rating_col]].values

        print(f"[MFRecommender] Training on matrix shape: ({n_users}, {n_items})")
        for epoch in range(self.epochs):
            np.random.shuffle(data)
            total_loss = 0
            for u, it, r in data:
                if u not in self.user_map or it not in self.item_map:
                    continue
                ui, ii = self.user_map[u], self.item_map[it]
                pred = self.user_factors[ui, :].dot(self.item_factors[ii, :])
                err = r - pred

                # SGD update
                self.user_factors[ui, :] += self.learning_rate * (err * self.item_factors[ii, :] - self.reg * self.user_factors[ui, :])
                self.item_factors[ii, :] += self.learning_rate * (err * self.user_factors[ui, :] - self.reg * self.item_factors[ii, :])

                total_loss += err**2

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"[MFRecommender] Epoch {epoch+1}/{self.epochs}, Loss={total_loss/len(data):.4f}")

        print(f"[MFRecommender] Factors learned: {self.user_factors.shape}, {self.item_factors.shape}")

    def recommend(self, user_id, top_k=10, k=None):
        if k is not None:
            top_k = k
        if user_id not in self.user_map:
            print(f"[MFRecommender] ⚠ User {user_id} not found in training data.")
            return []

        u_idx = self.user_map[user_id]
        scores = self.user_factors[u_idx, :].dot(self.item_factors.T)
        ranked_items = np.argsort(scores)[::-1][:top_k]
        rev_item_map = {v: k for k, v in self.item_map.items()}
        return [rev_item_map[i] for i in ranked_items]

    def score_items(self, user_id):
        if user_id not in self.user_map:
            return {}
        u_idx = self.user_map[user_id]
        scores = self.user_factors[u_idx, :].dot(self.item_factors.T)
        return {item: scores[idx] for item, idx in self.item_map.items()}


# ========================
# Content-Based Recommender
# ========================
class ContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.user_profiles = {}
        self.item_index = {}
        self.user_index = {}

    def fit(self, df, user_col="learner_id", item_col="content_type", feature_cols=None):
        self.item_index = {item: idx for idx, item in enumerate(df[item_col].unique())}
        self.user_index = {user: idx for idx, user in enumerate(df[user_col].unique())}
        self.item_features = np.zeros((len(self.item_index), len(feature_cols)))

        for item, idx in self.item_index.items():
            item_feats = df[df[item_col] == item][feature_cols].mean().values
            self.item_features[idx] = item_feats

        self.user_profiles = {}
        for user, u_idx in self.user_index.items():
            user_items = df[df[user_col] == user][item_col].map(self.item_index)
            if len(user_items) > 0:
                self.user_profiles[u_idx] = self.item_features[user_items].mean(axis=0)

        print(f"[Content-Based] Trained on {len(self.item_index)} items with {len(feature_cols)} features")
        print(f"[Content-Based] Built profiles for {len(self.user_profiles)} users")

    def recommend(self, user_id, top_k=None, k=None):
        if k is not None:
            top_k = k
        if top_k is None:
            top_k = 10

        if user_id not in self.user_index:
            print(f"[Content-Based] ⚠ User {user_id} not in training data. Falling back to popularity.")
            return list(self.item_index.keys())[:top_k]

        u_idx = self.user_index[user_id]
        if u_idx not in self.user_profiles:
            return list(self.item_index.keys())[:top_k]

        profile = self.user_profiles[u_idx]
        scores = cosine_similarity([profile], self.item_features)[0]
        ranked_items = np.argsort(scores)[::-1][:top_k]
        rev_item_index = {v: k for k, v in self.item_index.items()}
        return [rev_item_index[i] for i in ranked_items]


# ========================
# Hybrid Recommender (MF)
# ========================
class HybridRecommender:
    def __init__(self, pop_model, mf_model, cb_model, weights):
        self.pop = pop_model
        self.mf = mf_model
        self.cb = cb_model
        self.weights = weights

    def recommend(self, user_id, top_k=None, k=None):
        if k is not None:
            top_k = k
        if top_k is None:
            top_k = 10

        scores = {}

        if self.pop:
            pop_recs = self.pop.recommend(user_id, top_k=len(self.pop.popularity))
            for rank, item in enumerate(pop_recs):
                scores[item] = scores.get(item, 0) + self.weights['pop'] * (len(pop_recs) - rank)

        if self.mf and user_id in self.mf.user_map:
            mf_recs = self.mf.recommend(user_id, top_k=len(self.mf.item_map))
            for rank, item in enumerate(mf_recs):
                scores[item] = scores.get(item, 0) + self.weights['mf'] * (len(mf_recs) - rank)

        if self.cb and user_id in self.cb.user_index:
            cb_recs = self.cb.recommend(user_id, top_k=len(self.cb.item_index))
            for rank, item in enumerate(cb_recs):
                scores[item] = scores.get(item, 0) + self.weights['cb'] * (len(cb_recs) - rank)

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked_items[:top_k]]


# ========================
# Hybrid NCF Recommender
# ========================
class HybridNCFRecommender:
    def __init__(self, pop_model, ncf_model, cb_model, weights):
        self.pop = pop_model
        self.ncf = ncf_model
        self.cb = cb_model
        self.weights = weights

    def recommend(self, user_id, top_k=None, k=None):
        if k is not None:
            top_k = k
        if top_k is None:
            top_k = 10

        scores = {}

        if self.pop:
            pop_recs = self.pop.recommend(user_id, top_k=len(self.pop.popularity))
            for rank, item in enumerate(pop_recs):
                scores[item] = scores.get(item, 0) + self.weights['pop'] * (len(pop_recs) - rank)

        if self.ncf:
            ncf_recs = self.ncf.recommend(user_id, k=top_k*5)
            for rank, item in enumerate(ncf_recs):
                scores[item] = scores.get(item, 0) + self.weights['ncf'] * (len(ncf_recs) - rank)

        if self.cb and user_id in self.cb.user_index:
            cb_recs = self.cb.recommend(user_id, top_k=len(self.cb.item_index))
            for rank, item in enumerate(cb_recs):
                scores[item] = scores.get(item, 0) + self.weights['cb'] * (len(cb_recs) - rank)

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked_items[:top_k]]


# ========================
# Training Wrappers
# ========================
def train_popularity(train_df, item_col, rating_col=None):
    model = PopularityRecommender()
    model.fit(train_df, item_col=item_col, rating_col=rating_col)
    return model


def train_mf(train, user_col, item_col, rating_col, n_components=20, **kwargs):
    model = MFRecommender(num_factors=n_components, **kwargs)
    model.fit(train, user_col=user_col, item_col=item_col, rating_col=rating_col)
    return model


def train_content_based(train_df, user_col, item_col, feature_cols):
    model = ContentBasedRecommender()
    model.fit(train_df, user_col=user_col, item_col=item_col, feature_cols=feature_cols)
    return model


def train_hybrid(models, weights):
    pop_model = models.get("pop")
    mf_model = models.get("mf")
    cb_model = models.get("cb")
    return HybridRecommender(pop_model, mf_model, cb_model, weights)


def train_hybrid_ncf(models, weights):
    pop_model = models.get("pop")
    ncf_model = models.get("ncf")
    cb_model = models.get("cb")
    return HybridNCFRecommender(pop_model, ncf_model, cb_model, weights)
