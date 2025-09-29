# src/kg_features.py
"""
Knowledge Graph helper utilities for the recommendation pipeline.

Features:
- build_knowledge_graph(df, user_col, item_col, concept_col=None)
- random walks (deepwalk/node2vec style)
- generate_kg_embeddings(graph, method="node2vec"/"deepwalk"/"gnn")
  -> uses gensim.Word2Vec for node2vec/deepwalk walks
- map_embeddings_to_df(df, user_col, item_col, user_prefix="kg_user", item_prefix="kg_item")
  -> expands embeddings to many columns so DataFrame models can use them
- save/load graph and embeddings

Save locations:
- default: ../models/kg_graph.graphml and ../models/kg_embeddings.npz

Notes:
- If you want GNN-based embeddings, you'll need torch_geometric or DGL; the function
  has a placeholder and will raise a descriptive error if you pick 'gnn' without libs.
"""

from typing import Optional, Dict, List, Tuple, Any
import os
import random
import json
import numpy as np
import networkx as nx

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

# Default save paths (outside src/)
DEFAULT_MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)


# ---------------------------
# Graph building / utilities
# ---------------------------
def build_knowledge_graph(
    df,
    user_col: str,
    item_col: str,
    concept_col: Optional[str] = None,
    interaction_edge_attr: Optional[str] = None,
    directed: bool = False
) -> nx.Graph:
    """
    Build a simple KG from DataFrame.

    Nodes:
      - prefix 'u:' + user_id
      - prefix 'i:' + item_id
      - prefix 'c:' + concept_id (if provided)

    Edges:
      - (u:USER) -- engaged_with --> (i:ITEM)  (edge attr interaction_edge_attr if provided)
      - (i:ITEM) -- belongs_to --> (c:CONCEPT)

    Returns networkx Graph (or DiGraph if directed=True).
    """
    G = nx.DiGraph() if directed else nx.Graph()

    for _, row in df.iterrows():
        u = f"u:{row[user_col]}"
        i = f"i:{row[item_col]}"
        G.add_node(u, _type="user", raw_id=row[user_col])
        G.add_node(i, _type="item", raw_id=row[item_col])

        # edge user-item (may carry rating or timestamp)
        attrs = {}
        if interaction_edge_attr is not None and interaction_edge_attr in row:
            attrs[interaction_edge_attr] = row[interaction_edge_attr]
        G.add_edge(u, i, **attrs)

        # optional concept mapping
        if concept_col is not None and pd_notna(row.get(concept_col)):
            c = f"c:{row[concept_col]}"
            G.add_node(c, _type="concept", raw_id=row[concept_col])
            G.add_edge(i, c)

    return G


def pd_notna(x):
    # helper to abstract pandas NA without importing pandas here
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
# Random walks (DeepWalk / Node2Vec style)
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


def generate_random_walks(
    graph: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 40,
    seed: int = 42
) -> List[List[str]]:
    """
    Generate random walks for each node.
    Returns list of token lists (walks) suitable for Word2Vec.
    """
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
# Embedding generation (extended with GNN via torch_geometric)
# ---------------------------
def generate_kg_embeddings(
    graph: nx.Graph,
    method: str = "node2vec",
    dim: int = 64,
    num_walks: int = 10,
    walk_length: int = 40,
    window: int = 5,
    epochs: int = 5,
    workers: int = 1,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate node embeddings for the graph.

    method: "node2vec" | "deepwalk" | "gnn"
    - node2vec/deepwalk: random walks + Word2Vec
    - gnn: GraphSAGE (PyTorch Geometric)
    """
    method = method.lower()
    if method in ("node2vec", "deepwalk"):
        ...
        # (giữ nguyên code Node2Vec/DeepWalk ở bản trước)
        ...

    elif method == "gnn":
        try:
            import torch
            from torch_geometric.data import Data
            from torch_geometric.nn import SAGEConv
        except ImportError as e:
            raise ImportError(
                "PyTorch Geometric is required for method='gnn'.\n"
                "Install with: pip install torch torch_geometric"
            ) from e

        # build edge_index
        mapping = {n: i for i, n in enumerate(graph.nodes())}
        edges = []
        for u, v in graph.edges():
            edges.append([mapping[u], mapping[v]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # features: use identity (one-hot) or constant
        x = torch.eye(len(mapping), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index)

        # define GraphSAGE model
        class GraphSAGE(torch.nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim):
                super().__init__()
                self.conv1 = SAGEConv(in_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, out_dim)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index)
                return x

        model = GraphSAGE(x.size(1), dim, dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # unsupervised loss: force embeddings to not collapse (L2 norm reg)
            loss = (out.norm(dim=1) - 1).pow(2).mean()
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print(f"[GNN] Epoch {epoch}/{epochs}, loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index).cpu().numpy()

        embeddings = {node: emb[idx] for node, idx in mapping.items()}

        if save_path is None:
            save_path = os.path.join(DEFAULT_MODELS_DIR, "kg_embeddings_gnn.npz")
        np.savez_compressed(save_path, **{str(k): v for k, v in embeddings.items()})

        return embeddings

    else:
        raise ValueError("Unsupported method: choose 'node2vec'|'deepwalk'|'gnn'.")



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


def map_embeddings_to_df(
    df,
    user_col: str,
    item_col: str,
    embeddings: Dict[str, np.ndarray],
    user_prefix: str = "kg_user",
    item_prefix: str = "kg_item",
    fill_value: float = 0.0
):
    """
    For each row in df, find embeddings for user (u:<id>) and item (i:<id>)
    and expand them into columns (kg_user_0 ... kg_user_{D-1}) and
    (kg_item_0 ... kg_item_{D-1}).

    Returns new DataFrame (copy) with columns added.
    """
    import pandas as pd
    df_out = df.copy().reset_index(drop=True)

    # infer dim from embeddings dict
    if len(embeddings) == 0:
        raise ValueError("Empty embeddings dict")
    example = next(iter(embeddings.values()))
    dim = len(example)

    # prepare column names
    user_cols = [f"{user_prefix}_{i}" for i in range(dim)]
    item_cols = [f"{item_prefix}_{i}" for i in range(dim)]

    # initialize columns
    for c in user_cols + item_cols:
        df_out[c] = fill_value

    for idx, row in df_out.iterrows():
        ukey = f"u:{row[user_col]}"
        ikey = f"i:{row[item_col]}"

        uvec = embeddings.get(ukey)
        ivec = embeddings.get(ikey)
        if uvec is None:
            # fallback try raw id str
            if ukey not in embeddings and str(row[user_col]) in embeddings:
                uvec = embeddings[str(row[user_col])]
        if ivec is None:
            if ikey not in embeddings and str(row[item_col]) in embeddings:
                ivec = embeddings[str(row[item_col])]

        if uvec is not None:
            for i in range(dim):
                df_out.at[idx, user_cols[i]] = float(uvec[i])
        if ivec is not None:
            for i in range(dim):
                df_out.at[idx, item_cols[i]] = float(ivec[i])

    return df_out


def get_node_embedding(node_id: str, embeddings: Dict[str, np.ndarray], dim: int = 64) -> np.ndarray:
    """
    Return embedding for node_id (string like 'u:123' or 'i:456').
    If not present, return zero-vector.
    """
    if node_id in embeddings:
        return embeddings[node_id]
    else:
        # also try without prefix
        raw = node_id.split(":", 1)[-1]
        if raw in embeddings:
            return embeddings[raw]
        return np.zeros(dim, dtype=np.float32)


# ---------------------------
# Convenience: integrate end-to-end
# ---------------------------
def build_and_embed(
    df,
    user_col: str,
    item_col: str,
    concept_col: Optional[str] = None,
    method: str = "node2vec",
    dim: int = 64,
    save_graph_path: Optional[str] = None,
    save_embeddings_path: Optional[str] = None,
    **embed_kwargs
) -> Tuple[nx.Graph, Dict[str, np.ndarray]]:
    """
    Convenience function: build graph -> generate embeddings -> save -> return (graph, embeddings)
    """
    G = build_knowledge_graph(df, user_col, item_col, concept_col=concept_col)
    if save_graph_path is None:
        save_graph_path = os.path.join(DEFAULT_MODELS_DIR, "kg_graph.graphml")
    save_graph(G, save_graph_path)

    embeddings = generate_kg_embeddings(
        G,
        method=method,
        dim=dim,
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
