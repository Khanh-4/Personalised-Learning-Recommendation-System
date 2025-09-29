# src/kg_recommender.py
import numpy as np
import random

class KGRecommender:
    """
    Simple KG-based recommender wrapper that uses node embeddings produced
    by kg_features.build_and_embed(...) (emb: dict[node_id -> vector]).
    Usage:
        kg = KGRecommender(user_col="learner_id", item_col="content_type", emb=emb, metric='cosine')
        kg.fit(train_df)
        recs = kg.recommend(user_id, top_k=10)
    """
    def __init__(self, user_col, item_col, emb, metric="dot", seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.user_col = user_col
        self.item_col = item_col
        # convert embeddings to numpy arrays for safety
        self.emb_dict = {k: np.array(v, dtype=float) for k, v in (emb or {}).items()}
        self.metric = metric  # 'dot' or 'cosine'
        self.user_map = {}
        self.item_map = {}
        self.users_list = []
        self.items_list = []
        self.user_emb = None
        self.item_emb = None
        self.user_interactions = {}  # user_id -> set(item_id)

    def fit(self, df):
        """Build maps and embedding matrices from the training dataframe."""
        users = list(df[self.user_col].unique())
        items = list(df[self.item_col].unique())

        self.user_map = {u: idx for idx, u in enumerate(users)}
        self.item_map = {it: idx for idx, it in enumerate(items)}
        self.users_list = users
        self.items_list = items

        # dimension inference
        if len(self.emb_dict) > 0:
            dim = next(iter(self.emb_dict.values())).shape[0]
        else:
            # fallback dim
            dim = 32

        # prepare matrices
        self.user_emb = np.zeros((len(users), dim), dtype=float)
        self.item_emb = np.zeros((len(items), dim), dtype=float)

        # build user_interactions
        self.user_interactions = {}
        for u in users:
            user_items = df.loc[df[self.user_col] == u, self.item_col].unique()
            self.user_interactions[u] = set(user_items)

        for u, idx in self.user_map.items():
            vec = self._get_node_vec(u, dim)
            self.user_emb[idx] = vec

        for it, idx in self.item_map.items():
            vec = self._get_node_vec(it, dim)
            self.item_emb[idx] = vec

        if self.metric == "cosine":
            self.user_emb = self._normalize(self.user_emb)
            self.item_emb = self._normalize(self.item_emb)

    def _get_node_vec(self, node, dim):
        """Try to retrieve embedding for node, with fallback strategies."""
        # direct match
        if node in self.emb_dict:
            v = self.emb_dict[node]
            return self._resize_or_pad(v, dim)

        # try str/int conversion
        if isinstance(node, (int, np.integer)) and str(node) in self.emb_dict:
            v = self.emb_dict[str(node)]
            return self._resize_or_pad(v, dim)
        if isinstance(node, str) and node.isdigit() and int(node) in self.emb_dict:
            v = self.emb_dict[int(node)]
            return self._resize_or_pad(v, dim)

        # fallback: small random vector
        return np.random.normal(0, 1e-2, size=dim)

    def _resize_or_pad(self, arr, dim):
        arr = np.array(arr, dtype=float)
        if arr.shape[0] == dim:
            return arr
        if arr.shape[0] > dim:
            return arr[:dim]
        # pad with zeros
        out = np.zeros(dim, dtype=float)
        out[:arr.shape[0]] = arr
        return out

    def _normalize(self, mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def recommend(self, user_id, top_k=10, k=None, exclude_known=True, **kwargs):
        """
        Return top-k item IDs for `user_id`.
        Accepts both top_k and k (k overrides).
        exclude_known=True will filter out items the user already interacted with (if fit was called).
        """
        if k is not None:
            top_k = k

        # quick fallback if not fit or user unknown
        if self.item_emb is None or self.user_emb is None:
            # return random items
            items = list(self.item_map.keys())
            return random.sample(items, min(top_k, len(items)))

        if user_id not in self.user_map:
            # unseen user -> random (or could use popularity fallback)
            items = list(self.item_map.keys())
            return random.sample(items, min(top_k, len(items)))

        u_idx = self.user_map[user_id]
        u_vec = self.user_emb[u_idx]

        # compute scores
        if self.metric == "dot":
            scores = self.item_emb.dot(u_vec)
        else:  # cosine (embs should be normalized already)
            scores = self.item_emb.dot(u_vec)

        ranked_idx = np.argsort(-scores)  # descending

        # build ordered item ids, optionally excluding known items
        out = []
        known = self.user_interactions.get(user_id, set()) if exclude_known else set()
        for idx in ranked_idx:
            it = self.items_list[idx]
            if it in known:
                continue
            out.append(it)
            if len(out) >= top_k:
                break

        # if we filtered too many and need more, fill with random unseen items
        if len(out) < top_k:
            all_items = set(self.items_list)
            candidates = list(all_items - set(out) - known)
            if candidates:
                need = top_k - len(out)
                sampled = random.sample(candidates, min(need, len(candidates)))
                out.extend(sampled)

        return out
