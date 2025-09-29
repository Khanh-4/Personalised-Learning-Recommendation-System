import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class DecisionTreeRecommender:
    """
    Simple recommender using Decision Tree (CART).
    Predicts whether a user will like an item based on training data.
    """
    def __init__(self, user_col="user_id", item_col="item_id", rating_col="engagement_score", max_depth=None):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.item_features = None

    def fit(self, df: pd.DataFrame):
        # One-hot encode items
        X = pd.get_dummies(df[self.item_col])
        y = (df[self.rating_col] > 0.5).astype(int)  # binary: like/dislike
        self.model.fit(X, y)
        self.item_features = X.columns
        print(f"[DecisionTree] Trained with max_depth={self.max_depth}, items={len(self.item_features)}")

    def recommend(self, user_id, top_k=10):
        # Build item matrix (one-hot for each item)
        if self.item_features is None:
            raise ValueError("Model not fitted yet.")
        item_matrix = pd.DataFrame(np.eye(len(self.item_features)), columns=self.item_features)
        preds = self.model.predict_proba(item_matrix)[:, 1]  # probability of "like"
        ranked_idx = np.argsort(-preds)[:top_k]
        return list(self.item_features[ranked_idx])


class RandomForestRecommender:
    """
    Simple recommender using Random Forest.
    Aggregates multiple decision trees for better generalization.
    """
    def __init__(self, user_col="user_id", item_col="item_id", rating_col="engagement_score", n_estimators=100, max_depth=None):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        self.item_features = None

    def fit(self, df: pd.DataFrame):
        X = pd.get_dummies(df[self.item_col])
        y = (df[self.rating_col] > 0.5).astype(int)
        self.model.fit(X, y)
        self.item_features = X.columns
        print(f"[RandomForest] Trained with n_estimators={self.n_estimators}, items={len(self.item_features)}")

    def recommend(self, user_id, top_k=10):
        if self.item_features is None:
            raise ValueError("Model not fitted yet.")
        item_matrix = pd.DataFrame(np.eye(len(self.item_features)), columns=self.item_features)
        preds = self.model.predict_proba(item_matrix)[:, 1]
        ranked_idx = np.argsort(-preds)[:top_k]
        return list(self.item_features[ranked_idx])
