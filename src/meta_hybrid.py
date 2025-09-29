# src/meta_hybrid.py
# ============================================================
# Meta-Hybrid Recommender (stacking hoặc fixed weights)
# ============================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class MetaHybridRecommender:
    def __init__(self, base_scores, ground_truth=None, weights=None):
        """
        base_scores: dict gồm các score từ nhiều model
                     {"mf": scores_mf, "ncf": scores_ncf, "kg": scores_kg}
                     scores_xxx[user][item] = float

        ground_truth: dict[user] = set(items_interacted)
        weights: dict optional → nếu truyền vào thì sẽ bỏ qua meta-model
                 {"mf": 0.4, "ncf": 0.4, "kg": 0.2}
        """
        self.base_scores = base_scores
        self.ground_truth = ground_truth
        self.meta_model = LogisticRegression(max_iter=1000) if ground_truth else None
        self.weights = weights

    def build_dataset(self):
        X, y = [], []
        users = list(self.ground_truth.keys())
        first_model = list(self.base_scores.keys())[0]
        for user in users:
            items = list(self.base_scores[first_model][user].keys())
            for item in items:
                features = [self.base_scores[m][user][item] for m in self.base_scores]
                label = 1 if item in self.ground_truth[user] else 0
                X.append(features)
                y.append(label)
        return np.array(X), np.array(y)

    def fit(self):
        if self.weights:
            print(f"[Meta-Hybrid] Using fixed weights: {self.weights}")
            return
        if not self.ground_truth:
            raise ValueError("Need ground_truth to train meta-learner")
        X, y = self.build_dataset()
        self.meta_model.fit(X, y)
        auc = roc_auc_score(y, self.meta_model.predict_proba(X)[:, 1])
        print(f"[Meta-Hybrid] Meta-learner trained (AUC={auc:.4f})")

    def recommend(self, user=None, user_id=None, top_k=10):
        if user_id is not None:
            user = user_id
        if user is None:
            raise ValueError("Must provide user or user_id")

        first_model = list(self.base_scores.keys())[0]
        items = list(self.base_scores[first_model][user].keys())
        scores = []
        for item in items:
            if self.weights:
                score = sum(self.base_scores[m][user][item] * self.weights.get(m, 0) for m in self.base_scores)
            else:
                features = [self.base_scores[m][user][item] for m in self.base_scores]
                score = self.meta_model.predict_proba([features])[0, 1]
            scores.append((item, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scores[:top_k]]

