"""
Contextual bandit helpers for offline simulation.

Includes:
 - LinUCB (shared linear model)
 - EpsilonGreedy baseline
 - simulate_bandit(...) to run an offline online-style loop on test data
 - make_context_default(...) helper to build context vectors using:
     - KG embeddings (dict keys like 'u:<id>'/'i:<id>')
     - MF user/item latent factors (if available on mf_model)
     - item features (optional)
 - candidate prefilter helpers
"""

import numpy as np
import random
import logging
from typing import Dict, List, Callable, Tuple, Any, Optional
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Bandit Algorithms
# ---------------------------
class LinUCB:
    """Shared LinUCB (single linear model across all arms)."""
    def __init__(self, dim: int, alpha: float = 1.0, regularization: float = 1.0):
        self.d = dim
        self.alpha = float(alpha)
        self.reg = float(regularization)
        self.A = np.eye(self.d) * self.reg
        self.b = np.zeros((self.d,), dtype=float)

    def _theta(self) -> np.ndarray:
        try:
            return np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.A, self.b, rcond=None)[0]

    def score(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        theta = self._theta()
        mean = float(theta.dot(x))
        try:
            A_inv_x = np.linalg.solve(self.A, x)
            var = float(np.sqrt(x.dot(A_inv_x)))
        except np.linalg.LinAlgError:
            var = float(np.sqrt(x.dot(np.linalg.pinv(self.A)).dot(x)))
        return mean + self.alpha * var

    def select(self, contexts: List[np.ndarray], items: Optional[List[Any]] = None) -> Any:
        scores = [self.score(x) for x in contexts]
        idx = int(np.argmax(scores))
        return items[idx] if items is not None else idx

    def update(self, x: np.ndarray, reward: float):
        x = np.asarray(x, dtype=float)
        self.A += np.outer(x, x)
        self.b += reward * x

    def get_params(self):
        return {"A": self.A.copy(), "b": self.b.copy(), "alpha": self.alpha, "reg": self.reg}


class EpsilonGreedy:
    """Simple epsilon-greedy baseline over average item reward."""
    def __init__(self, epsilon: float = 0.1):
        self.eps = float(epsilon)
        self.counts = defaultdict(int)
        self.value = defaultdict(float)

    def select(self, contexts: List[np.ndarray], items: List[Any]) -> Any:
        if random.random() < self.eps:
            return random.choice(items)
        best_item = None
        best_val = -np.inf
        for it in items:
            v = self.value.get(it, 0.0)
            if v > best_val:
                best_val = v
                best_item = it
        return best_item if best_item is not None else random.choice(items)

    def update(self, item, reward: float):
        self.counts[item] += 1
        n = self.counts[item]
        self.value[item] += (reward - self.value[item]) / n


# ---------------------------
# Context builder / helpers
# ---------------------------
def flatten_array(a):
    return np.asarray(a, dtype=float).ravel()

def concat_features(*parts) -> np.ndarray:
    arrs = [flatten_array(p) for p in parts if p is not None]
    return np.concatenate(arrs) if arrs else np.zeros(1, dtype=float)

def binarize_reward(score: float, threshold: float = 0.5) -> int:
    return 1 if score >= threshold else 0

def safe_get_embed(emb_dict, key):
    """Avoid ambiguous truth value errors when embedding exists but is array."""
    if emb_dict is None:
        return None
    val = emb_dict.get(key)
    return None if val is None else np.asarray(val, dtype=float)

def get_mf_embeddings_from_model(mf_model) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
    user_embs, item_embs = {}, {}
    if mf_model is None:
        return user_embs, item_embs
    possible_user_attrs = ["user_factors", "U", "user_embeddings", "user_latent"]
    possible_item_attrs = ["item_factors", "V", "item_embeddings", "item_latent"]
    user_map = getattr(mf_model, "user_map", None)
    item_map = getattr(mf_model, "item_map", None)
    for attr in possible_user_attrs:
        if hasattr(mf_model, attr):
            arr = getattr(mf_model, attr)
            if isinstance(arr, np.ndarray):
                if user_map:
                    for uid, idx in user_map.items():
                        user_embs[uid] = arr[int(idx)]
                else:
                    for i in range(arr.shape[0]):
                        user_embs[i] = arr[i]
            break
    for attr in possible_item_attrs:
        if hasattr(mf_model, attr):
            arr = getattr(mf_model, attr)
            if isinstance(arr, np.ndarray):
                if item_map:
                    for iid, idx in item_map.items():
                        item_embs[iid] = arr[int(idx)]
                else:
                    for i in range(arr.shape[0]):
                        item_embs[i] = arr[i]
            break
    return user_embs, item_embs


def make_context_default(
    user_id,
    item_id,
    *,
    kg_embeddings: Optional[Dict[str, np.ndarray]] = None,
    mf_model=None,
    item_feature_map: Optional[Dict[Any, np.ndarray]] = None,
    session_feats: Optional[Dict[str, float]] = None,
    concat_scale: Optional[StandardScaler] = None,
    fallback_dim: int = 16
) -> np.ndarray:
    """Construct context vector safely combining KG, MF, and features."""
    parts = []

    # 1. KG embeddings
    if kg_embeddings is not None:
        for key in [f"u:{user_id}", str(user_id)]:
            v = safe_get_embed(kg_embeddings, key)
            if v is not None:
                parts.append(v)
                break
        for key in [f"i:{item_id}", str(item_id)]:
            v = safe_get_embed(kg_embeddings, key)
            if v is not None:
                parts.append(v)
                break

    # 2. MF embeddings
    if mf_model is not None:
        user_embs, item_embs = get_mf_embeddings_from_model(mf_model)
        uvec = user_embs.get(user_id)
        ivec = item_embs.get(item_id)
        if uvec is not None:
            parts.append(np.asarray(uvec))
        if ivec is not None:
            parts.append(np.asarray(ivec))

    # 3. Item features
    if item_feature_map is not None and item_id in item_feature_map:
        parts.append(np.asarray(item_feature_map[item_id]))

    # 4. Session features
    if session_feats is not None and isinstance(session_feats, dict):
        sf = np.asarray([float(session_feats.get(k, 0.0)) for k in sorted(session_feats.keys())], dtype=float)
        parts.append(sf)

    if not parts:
        return np.zeros(fallback_dim, dtype=float)

    vec = concat_features(*parts)
    if concat_scale is not None:
        try:
            vec = concat_scale.transform(vec.reshape(1, -1)).ravel()
        except Exception:
            pass
    return vec


# ---------------------------
# Simulation
# ---------------------------
def simulate_bandit(
    bandit,
    test_df,
    make_context_fn: Callable[[Any, Any], np.ndarray],
    user_col: str,
    item_col: str,
    reward_fn: Callable[[Any, Any, dict], float],
    candidate_selector: Optional[Callable[[Any, dict], List[Any]]] = None,
    binarize: bool = False,
    binary_threshold: float = 0.5,
    top_k: int = 1,
    verbose: bool = False,
    return_logs: bool = False   # ✅ thêm tham số
):
    """Offline simulation for contextual bandit."""
    import pandas as pd
    users = test_df[user_col].unique()
    all_items = test_df[item_col].unique().tolist()

    if candidate_selector is None:
        def candidate_selector_default(u, row): return all_items
        candidate_selector = candidate_selector_default

    total_reward = 0.0
    total_binary_hits = 0
    total_rounds = 0
    reward_logs = []   # ✅ log reward từng vòng

    for idx, row in test_df.iterrows():
        u = row[user_col]
        candidates = candidate_selector(u, row)
        if len(candidates) == 0:
            continue

        contexts = [make_context_fn(u, i) for i in candidates]
        chosen = bandit.select(contexts, candidates)
        reward = reward_fn(u, chosen, row)

        r = float(binarize_reward(reward, threshold=binary_threshold)) if binarize else float(reward)

        # update
        try:
            ch_idx = candidates.index(chosen)
            bandit.update(contexts[ch_idx], r)
        except Exception:
            pass

        total_reward += r
        if binarize and r > 0:
            total_binary_hits += 1
        total_rounds += 1

        reward_logs.append(r)   # ✅ lưu reward

        if verbose and total_rounds % 200 == 0:
            print(f"[Bandit] rounds={total_rounds}, cum_reward={total_reward:.3f}")

    avg_reward = total_reward / total_rounds if total_rounds else 0.0
    ctr = (total_binary_hits / total_rounds) if total_rounds and binarize else None

    results = {
        "rounds": total_rounds,
        "cumulative_reward": total_reward,
        "avg_reward": avg_reward,
        "ctr": ctr
    }
    if return_logs:
        results["logs"] = reward_logs   # ✅ trả thêm log reward để vẽ biểu đồ

    return results


# ---------------------------
# Prefilter helper
# ---------------------------
def prefilter_top_k_by_popularity(pop_model, user_id, row, top_k=50):
    try:
        recs = prefilter_top_k_by_popularity_inner(pop_model, user_id, top_k)
        return recs
    except Exception:
        return []

def prefilter_top_k_by_popularity_inner(pop_model, user_id, top_k):
    try:
        recs = pop_model.recommend(user_id=user_id, top_k=top_k)
    except TypeError:
        try:
            recs = pop_model.recommend(user_id, top_k)
        except Exception:
            recs = pop_model.recommend(user_id)[:top_k]
    if isinstance(recs, list):
        return recs
    elif isinstance(recs, dict):
        return [i for i, _ in sorted(recs.items(), key=lambda x: -x[1])][:top_k]
    else:
        return list(recs)[:top_k]
