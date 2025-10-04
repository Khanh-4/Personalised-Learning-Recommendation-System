# src/kg_features.py
"""
Knowledge Graph helper utilities for the recommendation pipeline.
"""

from typing import Optional, Dict, List, Tuple, Any
import os
import random
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# optional imports
try:
    from gensim.models import Word2Vec
    _HAS_GENSIM = True
except Exception:
    Word2Vec = None
    _HAS_GENSIM = False

# reproducibility
RNG = random.Random(42)
np.random.seed(42)

DEFAULT_MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)


# ---------------------------
# Graph building
# ---------------------------
def build_knowledge_graph(df, user_col: str, item_col: str,
                          concept_col: Optional[str] = None,
                          interaction_edge_attr: Optional[str] = None,
                          directed: bool = False) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    for _, row in df.iterrows():
        u = f"u:{row[user_col]}"
        i = f"i:{row[item_col]}"
        G.add_node(u, _type="user", raw_id=row[user_col])
        G.add_node(i, _type="item", raw_id=row[item_col])

        attrs = {}
        if interaction_edge_attr is not None and interaction_edge_attr in row:
            attrs[interaction_edge_attr] = row[interaction_edge_attr]
        G.add_edge(u, i, **attrs)

        if concept_col is not None and pd_notna(row.get(concept_col)):
            c = f"c:{row[concept_col]}"
            G.add_node(c, _type="concept", raw_id=row[concept_col])
            G.add_edge(i, c)
    return G


def pd_notna(x):
    try:
        import pandas as pd
        return pd.notna(x)
    except Exception:
        return x is not None


def save_graph(graph: nx.Graph, path: Optional[str] = None):
    if path is None:
        path = os.path.join(DEFAULT_MODELS_DIR, "kg_graph.graphml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_graphml(graph, path)


def load_graph(path: Optional[str] = None) -> nx.Graph:
    if path is None:
        path = os.path.join(DEFAULT_MODELS_DIR, "kg_graph.graphml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    return nx.read_graphml(path)


# ---------------------------
# Random walks
# ---------------------------
def _random_walk(graph: nx.Graph, start_node: str, walk_length: int, rng: random.Random) -> List[str]:
    walk = [start_node]
    for _ in range(walk_length - 1):
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if len(neighbors) == 0:
            break
        walk.append(rng.choice(neighbors))
    return walk


def generate_random_walks(graph: nx.Graph, num_walks: int = 10,
                          walk_length: int = 40, seed: int = 42) -> List[List[str]]:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            w = _random_walk(graph, node, walk_length, rng)
            walks.append(w)
    return walks


# ---------------------------
# Embedding generation
# ---------------------------
def generate_kg_embeddings(graph: nx.Graph, method: str = "node2vec", dim: int = 64,
                           num_walks: int = 10, walk_length: int = 40, window: int = 5,
                           epochs: int = 5, workers: int = 1, seed: int = 42,
                           save_path: Optional[str] = None,
                           device: Optional[str] = None) -> Dict[str, np.ndarray]:
    method = method.lower()
    if method in ("node2vec", "deepwalk"):
        if not _HAS_GENSIM:
            raise ImportError("gensim is required for node2vec/deepwalk embeddings")
        walks = generate_random_walks(graph, num_walks=num_walks,
                                      walk_length=walk_length, seed=seed)
        model = Word2Vec(
            sentences=walks,
            vector_size=dim,
            window=window,
            min_count=0,
            sg=1,
            workers=workers,
            epochs=epochs,
            seed=seed,
        )
        embeddings = {node: model.wv[node] for node in graph.nodes()}

    elif method == "gnn":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        mapping = {n: i for i, n in enumerate(graph.nodes())}
        edges = [[mapping[u], mapping[v]] for u, v in graph.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        x = torch.eye(len(mapping), dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index).to(device)

        class GraphSAGE(torch.nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim):
                super().__init__()
                self.conv1 = SAGEConv(in_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, out_dim)
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index)
                return x

        model = GraphSAGE(x.size(1), dim, dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = (out.norm(dim=1) - 1).pow(2).mean()
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print(f"[GNN] Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index).cpu().numpy()
        embeddings = {node: emb[idx] for node, idx in mapping.items()}

    else:
        raise ValueError("Unsupported method: choose 'node2vec'|'deepwalk'|'gnn'")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **{str(k): v for k, v in embeddings.items()})

    return embeddings


def load_embeddings(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    if path is None:
        path = os.path.join(DEFAULT_MODELS_DIR, "kg_embeddings.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ---------------------------
# Mapping embeddings into DataFrame
# ---------------------------
def _expand_embedding_to_cols(prefix: str, vec: np.ndarray) -> Dict[str, Any]:
    return {f"{prefix}_{i}": float(vec[i]) for i in range(len(vec))}


def map_embeddings_to_df(df, user_col: str, item_col: str,
                         embeddings: Dict[str, np.ndarray],
                         user_prefix: str = "kg_user",
                         item_prefix: str = "kg_item",
                         fill_value: float = 0.0):
    import pandas as pd
    df_out = df.copy().reset_index(drop=True)
    if len(embeddings) == 0:
        raise ValueError("Empty embeddings dict")
    example = next(iter(embeddings.values()))
    dim = len(example)

    user_cols = [f"{user_prefix}_{i}" for i in range(dim)]
    item_cols = [f"{item_prefix}_{i}" for i in range(dim)]

    for c in user_cols + item_cols:
        df_out[c] = fill_value

    for idx, row in df_out.iterrows():
        ukey = f"u:{row[user_col]}"
        ikey = f"i:{row[item_col]}"
        uvec = embeddings.get(ukey)
        ivec = embeddings.get(ikey)
        if uvec is not None:
            for i in range(dim):
                df_out.at[idx, user_cols[i]] = float(uvec[i])
        if ivec is not None:
            for i in range(dim):
                df_out.at[idx, item_cols[i]] = float(ivec[i])
    return df_out


def get_node_embedding(node_id: str, embeddings: Dict[str, np.ndarray], dim: int = 64) -> np.ndarray:
    if node_id in embeddings:
        return embeddings[node_id]
    raw = node_id.split(":", 1)[-1]
    if raw in embeddings:
        return embeddings[raw]
    return np.zeros(dim, dtype=np.float32)


# ---------------------------
# Convenience
# ---------------------------
def build_and_embed(df, user_col: str, item_col: str,
                    concept_col: Optional[str] = None,
                    method: str = "node2vec", dim: int = 64,
                    save_graph_path: Optional[str] = None,
                    save_embeddings_path: Optional[str] = None,
                    **embed_kwargs) -> Tuple[nx.Graph, Dict[str, np.ndarray]]:
    G = build_knowledge_graph(df, user_col, item_col, concept_col=concept_col)
    if save_graph_path is None:
        save_graph_path = os.path.join(DEFAULT_MODELS_DIR, "kg_graph.graphml")
    save_graph(G, save_graph_path)

    embeddings = generate_kg_embeddings(
        G, method=method, dim=dim,
        save_path=save_embeddings_path or os.path.join(DEFAULT_MODELS_DIR, "kg_embeddings.npz"),
        **embed_kwargs
    )
    return G, embeddings



# ---------------------------
# small test / example usage
# ---------------------------
if __name__ == "__main__":
    # quick smoke test when running the module directly (requires pandas + gensim)
    try:
        import pandas as pd
        print("Running kg_features.py self-test...")

        # build toy dataset
        df = pd.DataFrame([
            {"learner_id": "u1", "content_type": "cA", "topic": "T1"},
            {"learner_id": "u1", "content_type": "cB", "topic": "T2"},
            {"learner_id": "u2", "content_type": "cA", "topic": "T1"},
            {"learner_id": "u3", "content_type": "cC", "topic": "T3"},
        ])

        G = build_knowledge_graph(df, user_col="learner_id", item_col="content_type", concept_col="topic")
        print("Graph nodes:", len(G.nodes()), "edges:", len(G.edges()))
        if _HAS_GENSIM:
            emb = generate_kg_embeddings(G, method="node2vec", dim=16, num_walks=5, walk_length=10, epochs=10, workers=1)
            print("Sample embedding keys:", list(emb.keys())[:6])
            df2 = map_embeddings_to_df(df, user_col="learner_id", item_col="content_type", embeddings=emb)
            print("Expanded df shape:", df2.shape)
            print(df2.columns.tolist()[:10])
        else:
            print("gensim not installed; skip embedding generation (pip install gensim to enable).")

        print("Self-test done.")

    except Exception as e:
        print("Self-test failed:", e)
