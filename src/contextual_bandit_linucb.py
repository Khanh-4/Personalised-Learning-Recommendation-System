# src/contextual_bandit_linucb.py
# ============================================================
# LinUCB Contextual Bandit Implementation
# ============================================================

import numpy as np


class LinUCB:
    """
    Implementation of the LinUCB (Linear Upper Confidence Bound) algorithm.

    Used for contextual bandits — selecting arms (items) based on context vectors.
    """

    def __init__(self, n_arms, n_features, alpha=0.25):
        """
        Parameters
        ----------
        n_arms : int
            Number of actions (e.g., items).
        n_features : int
            Size of feature vector (context dimension).
        alpha : float
            Exploration parameter — higher means more exploration.
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        # Each arm has its own covariance matrix A and bias vector b
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    # --------------------------------------------------------
    # Select arm given contexts
    # --------------------------------------------------------
    def select_arm(self, contexts):
        """
        Select the best arm based on UCB policy.

        Parameters
        ----------
        contexts : list of np.array
            List of context vectors, one per arm.

        Returns
        -------
        int
            Index of the chosen arm.
        """
        p = []

        for a in range(self.n_arms):
            x = contexts[a].reshape(-1, 1)
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            # Predict reward + exploration bonus
            p_t = (theta.T @ x) + self.alpha * np.sqrt(x.T @ A_inv @ x)
            p.append(p_t.item())

        return int(np.argmax(p))

    # --------------------------------------------------------
    # Update model after observing reward
    # --------------------------------------------------------
    def update(self, arm, reward, context):
        """
        Update parameters of the selected arm.

        Parameters
        ----------
        arm : int
            Index of the selected arm.
        reward : float
            Observed reward.
        context : np.array
            Context vector for the selected arm.
        """
        x = context.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x

    # --------------------------------------------------------
    # Get learned weight vector for each arm
    # --------------------------------------------------------
    def get_theta(self, arm):
        """Return estimated weight vector θ for a given arm."""
        A_inv = np.linalg.inv(self.A[arm])
        return (A_inv @ self.b[arm]).flatten()

    # --------------------------------------------------------
    # Predict expected reward for a given arm & context
    # --------------------------------------------------------
    def predict_reward(self, arm, context):
        x = context.reshape(-1, 1)
        theta = self.get_theta(arm)
        return float(theta @ x)
