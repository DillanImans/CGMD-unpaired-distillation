from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import NearestNeighbors
from utils.run_config import (
    get_run_root,
    is_unified_config,
    load_yaml,
    phase_dir,
    resolve_path,
    write_run_config,
)

def _safe_np_str_array(xs):
    return np.array(list(xs), dtype=str)

def _softmax_rowwise(neg_d_over_temp: np.ndarray) -> np.ndarray:
    m = neg_d_over_temp.max(axis = 1, keepdims = True)
    ex = np.exp(neg_d_over_temp - m)
    return ex / (ex.sum(axis = 1, keepdims = True) + 1e-12)


def _symmetrize_weighted_edges(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    n_nodes: int,
    edge_distance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    w = edge_weight.astype(np.float32)

    valid = np.logical_and.reduce([src >= 0, src < n_nodes, dst >= 0, dst < n_nodes, src != dst])
    src = src[valid]
    dst = dst[valid]
    w = w[valid]
    d = None
    if edge_distance is not None:
        d = edge_distance.astype(np.float32)[valid]
    if src.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32), None

    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    keys = u.astype(np.int64) * np.int64(n_nodes) + v.astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    u = u[order]
    v = v[order]
    keys = keys[order]
    w = w[order]
    if d is not None:
        d = d[order]

    _, starts = np.unique(keys, return_index=True)
    ends = np.append(starts[1:], len(keys))
    w_pair = np.array([float(w[s:e].mean()) for s, e in zip(starts, ends)], dtype=np.float32)
    d_pair = None
    if d is not None:
        d_pair = np.array([float(d[s:e].mean()) for s, e in zip(starts, ends)], dtype=np.float32)
    u_pair = u[starts]
    v_pair = v[starts]

    src_sym = np.concatenate([u_pair, v_pair]).astype(np.int64)
    dst_sym = np.concatenate([v_pair, u_pair]).astype(np.int64)
    w_sym = np.concatenate([w_pair, w_pair]).astype(np.float32)

    row_sum = np.bincount(src_sym, weights=w_sym.astype(np.float64), minlength=n_nodes).astype(np.float32)
    w_norm = w_sym / np.maximum(row_sum[src_sym], 1e-12)

    d_sym = None
    if d_pair is not None:
        d_sym = np.concatenate([d_pair, d_pair]).astype(np.float32)

    return np.stack([src_sym, dst_sym], axis=0).astype(np.int64), w_norm.astype(np.float32), d_sym


def _anchor_coverage_from_edges(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    is_anchor: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    if src.size == 0:
        return np.zeros(n_nodes, dtype=np.int64), np.zeros(n_nodes, dtype=np.float32)

    anchor_hit = is_anchor[dst].astype(np.float32)
    anchor_count = np.bincount(src, weights=anchor_hit, minlength=n_nodes)
    anchor_conf = np.bincount(src, weights=edge_weight.astype(np.float32) * anchor_hit, minlength=n_nodes)
    return np.rint(anchor_count).astype(np.int64), anchor_conf.astype(np.float32)


def build_knn_graph(
    df_split: pd.DataFrame,
    feature_cols: list[str],
    k: int,
    metric: str,
    weight_mode: str,
    temperature: float,
    rbf_sigma: float,
    edge_direction: str = "directed",
) -> dict:
    
    assert metric in ["cosine", "euclidean"]
    assert weight_mode in ["softmax", "rbf"]

    patient_ids = df_split["patient_id"].astype(str).tolist()
    X = df_split[feature_cols].to_numpy(dtype=np.float32)

    N = X.shape[0]
    if N < 2:
        raise ValueError(f"Need atleast 2 patients to build a graph, got N={N}")
    
    k_eff = min(k, N - 1)

    nn = NearestNeighbors(n_neighbors = k_eff + 1, metric = metric)
    nn.fit(X)
    dists, nbrs = nn.kneighbors(X, return_distance=True)


    rows = np.arange(N)
    is_self = (nbrs == rows[:, None])
    nbrs_clean = []
    dists_clean = []

    for i in range(N):
        mask = ~is_self[i]
        nbrs_i = nbrs[i][mask]
        dists_i = dists[i][mask]

        nbrs_clean.append(nbrs_i[:k_eff])
        dists_clean.append(dists_i[:k_eff])

    nbrs = np.stack(nbrs_clean, axis = 0)
    dists = np.stack(dists_clean, axis = 0)

    if weight_mode == "softmax":
        assert temperature > 0
        w = _softmax_rowwise(-dists / temperature)
    else:
        assert rbf_sigma > 0
        w = np.exp(-(dists ** 2) / (2.0 * (rbf_sigma ** 2)))
        w = w / (w.sum(axis = 1, keepdims = True) + 1e-12)

    src = np.repeat(np.arange(N), k_eff)
    dst = nbrs.reshape(-1)
    edge_index = np.stack([src, dst], axis = 0).astype(np.int64)
    edge_weight = w.reshape(-1).astype(np.float32)
    edge_distance = dists.reshape(-1).astype(np.float32)
    if edge_direction == "undirected":
        edge_index, edge_weight, edge_distance = _symmetrize_weighted_edges(
            edge_index, edge_weight, N, edge_distance=edge_distance
        )
        nbrs = np.zeros((0, 0), dtype=np.int64)
        dists = np.zeros((0, 0), dtype=np.float32)

    return {
        "patient_ids": patient_ids,
        "X": X,
        "k_eff": k_eff,
        "neighbors": nbrs,
        "distances": dists,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_distance": edge_distance,
        "edge_direction": edge_direction,
    }


def build_knn_graph_with_pool(
    df_query: pd.DataFrame,
    df_pool: pd.DataFrame,
    feature_cols: list[str],
    k: int,
    metric: str,
    weight_mode: str,
    temperature: float,
    rbf_sigma: float,
) -> dict:
    assert metric in ["cosine", "euclidean"]
    assert weight_mode in ["softmax", "rbf"]

    query_ids = df_query["patient_id"].astype(str).tolist()
    pool_ids = df_pool["patient_id"].astype(str).tolist()
    Xq = df_query[feature_cols].to_numpy(dtype=np.float32)
    Xp = df_pool[feature_cols].to_numpy(dtype=np.float32)

    if Xq.shape[0] == 0 or Xp.shape[0] == 0:
        raise ValueError("Query or pool has no rows for KNN graph")

    k_eff = min(k, Xp.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(Xp)
    dists, nbrs = nn.kneighbors(Xq, return_distance=True)

    if weight_mode == "softmax":
        assert temperature > 0
        w = _softmax_rowwise(-dists / temperature)
    else:
        assert rbf_sigma > 0
        w = np.exp(-(dists ** 2) / (2.0 * (rbf_sigma ** 2)))
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)

    return {
        "query_ids": query_ids,
        "pool_ids": pool_ids,
        "k_eff": k_eff,
        "neighbors": nbrs,
        "distances": dists,
        "weights": w,
    }


def degree_stats(edge_index: np.ndarray, N: int) -> dict:
    src = edge_index[0]
    dst = edge_index[1]
    out_deg = np.bincount(src, minlength = N)
    in_deg = np.bincount(dst, minlength = N)

    def summ(x):
        return {
            "min": int(x.min()),
            "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
            "max": int(x.max()),
            "mean": float(x.mean()),
        }
    
    return {"out_deg": summ(out_deg), "in_deg": summ(in_deg)}

def main(config_path: str):
    cfg = load_yaml(config_path)
    unified = is_unified_config(cfg)
    if unified:
        run_root = get_run_root(cfg)
        gcfg = dict(cfg["graph"])
    else:
        run_root = None
        gcfg = cfg["graph"]

    patient_csv = Path(gcfg["patient_index_csv"])
    splits_csv = gcfg.get("splits_csv")
    if not splits_csv:
        raise ValueError("graph.splits_csv is required to build train-only graph from splits.csv")
    if unified and run_root is not None:
        default_out_dir = phase_dir(run_root, "graphs")
        gcfg["out_dir"] = resolve_path(gcfg.get("out_dir"), default_out_dir)
        write_run_config(run_root, config_path)
    out_dir = Path(gcfg.get("out_dir", "important_results/graphs"))
    out_dir.mkdir(parents=True, exist_ok = True)

    splits = gcfg.get("splits", ["train"])
    combine_splits = bool(gcfg.get("combine_splits", False))
    combined_name = gcfg.get("combined_name", "trainval")
    inductive = bool(gcfg.get("inductive", False))
    inductive_query_splits = gcfg.get("inductive_query_splits", [])
    query_split_col = gcfg.get("inductive_query_split_col")

    k = int(gcfg.get("k", 20))
    metric = gcfg.get("metric", "cosine")
    weight_mode = gcfg.get("weight_mode", "softmax")
    edge_direction = str(gcfg.get("edge_direction", "directed")).lower()
    if edge_direction not in {"directed", "undirected"}:
        raise ValueError("graph.edge_direction must be one of: directed, undirected")
    temperature = float(gcfg.get("temperature", 0.5))
    rbf_sigma = float(gcfg.get("rbf_sigma", 1.0))

    # clinical preprocessing config
    clinical_cols = gcfg.get(
        "clinical_cols",
        [
            "age","sex","sbp","dbp","dyslipidemia","smoking","cad","af","paod",
            "creatinine","bun","cholesterol","triglyceride","hdl","ldl","glucose",
        ],
    )
    keep_cols = gcfg.get("keep_cols", ["n_brain","has_brain","n_fundus","has_fundus"])
    bin_cols = gcfg.get("binary_cols", ["sex","dyslipidemia","smoking","cad","af","paod"])
    cont_cols = [c for c in clinical_cols if c not in bin_cols]
    contz_cols = [f"{c}_z" for c in cont_cols]
    feature_cols = list(bin_cols) + contz_cols

    df = pd.read_csv(patient_csv)
    df["patient_id"] = df["patient_id"].astype(str)

    df_splits = pd.read_csv(splits_csv)
    split_col = "split"
    if "brain_graph_split" in df_splits.columns:
        split_col = "brain_graph_split"
    if not {"patient_id", split_col}.issubset(df_splits.columns):
        raise ValueError("splits_csv must include columns: patient_id and split column")
    split_map = dict(zip(df_splits["patient_id"].astype(str), df_splits[split_col].astype(str)))
    df["split"] = df["patient_id"].map(split_map)

    train_mask = df["split"] == "train"
    if train_mask.sum() == 0:
        raise ValueError("No train rows found after applying splits.csv")

    scaler_stats = {}
    for c in cont_cols:
        mu = df.loc[train_mask, c].mean()
        sd = df.loc[train_mask, c].std(ddof=0)
        sd = float(sd) if sd and sd > 1e-12 else 1.0
        scaler_stats[c] = {"mean": float(mu), "std": float(sd)}
        df[f"{c}_z"] = (df[c] - mu) / sd

    for c in bin_cols:
        df[c] = df[c].astype(int)

    print(f"[PREP] train_rows={int(train_mask.sum())} total_rows={len(df)}")
    for c in cont_cols:
        stats = scaler_stats[c]
        print(f"[PREP] {c}: mean={stats['mean']:.6f} std={stats['std']:.6f}")

    emb_path = gcfg.get("brain_embeddings_pt", None)
    emb_anchor_set = None
    if emb_path is not None and str(emb_path).lower() != "null":
        emb = torch.load(emb_path, map_location="cpu")
        emb_anchor_set = set(map(str, emb["patient"]["patient_ids"]))


    if inductive:
        df_train = df[df["split"] == "train"].copy()
        df_val = df[df["split"] == "val"].copy()
        if len(df_train) == 0:
            raise ValueError("No train rows found for inductive graph")
        if len(df_val) == 0:
            raise ValueError("No val rows found for inductive graph")
        df_query = df.iloc[0:0].copy()
        if inductive_query_splits:
            if query_split_col is None:
                if "fundus_split" in df_splits.columns:
                    query_split_col = "fundus_split"
            if query_split_col is None:
                raise ValueError("inductive_query_splits set but no inductive_query_split_col found")
            if not {"patient_id", query_split_col}.issubset(df_splits.columns):
                raise ValueError("splits_csv missing inductive_query_split_col")
            query_map = dict(zip(df_splits["patient_id"].astype(str), df_splits[query_split_col].astype(str)))
            df_query = df[df["patient_id"].map(query_map).isin(inductive_query_splits)].copy()
            if not df_query.empty:
                df_query["split"] = "query"
                overlap = set(df_query["patient_id"]) & set(df_train["patient_id"]) | set(df_val["patient_id"])
                if overlap:
                    df_query = df_query[~df_query["patient_id"].isin(overlap)].copy()
                    print(f"[WARN] dropped {len(overlap)} inductive query rows already in train/val")

        train_graph = build_knn_graph_with_pool(
            df_query=df_train,
            df_pool=df_train,
            feature_cols=feature_cols,
            k=k,
            metric=metric,
            weight_mode=weight_mode,
            temperature=temperature,
            rbf_sigma=rbf_sigma,
        )

        val_graph = build_knn_graph_with_pool(
            df_query=df_val,
            df_pool=df_train,
            feature_cols=feature_cols,
            k=k,
            metric=metric,
            weight_mode=weight_mode,
            temperature=temperature,
            rbf_sigma=rbf_sigma,
        )

        if not df_query.empty:
            query_graph = build_knn_graph_with_pool(
                df_query=df_query,
                df_pool=df_train,
                feature_cols=feature_cols,
                k=k,
                metric=metric,
                weight_mode=weight_mode,
                temperature=temperature,
                rbf_sigma=rbf_sigma,
            )
        else:
            query_graph = None

        patient_ids = train_graph["query_ids"] + val_graph["query_ids"]
        if query_graph is not None:
            patient_ids += query_graph["query_ids"]
        pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

        def _edges_from_knn(query_ids, pool_ids, nbrs, dists, weights):
            src = []
            dst = []
            dist = []
            w = []
            for i, qid in enumerate(query_ids):
                qidx = pid_to_idx[qid]
                for j, nbr in enumerate(nbrs[i]):
                    pid = pool_ids[nbr]
                    src.append(qidx)
                    dst.append(pid_to_idx[pid])
                    dist.append(float(dists[i, j]))
                    w.append(float(weights[i, j]))
            return (
                np.array(src, dtype=np.int64),
                np.array(dst, dtype=np.int64),
                np.array(dist, dtype=np.float32),
                np.array(w, dtype=np.float32),
            )

        src_tr, dst_tr, d_tr, w_tr = _edges_from_knn(
            train_graph["query_ids"], train_graph["pool_ids"], train_graph["neighbors"], train_graph["distances"], train_graph["weights"]
        )
        src_val, dst_val, d_val, w_val = _edges_from_knn(
            val_graph["query_ids"], val_graph["pool_ids"], val_graph["neighbors"], val_graph["distances"], val_graph["weights"]
        )
        if query_graph is not None:
            src_q, dst_q, d_q, w_q = _edges_from_knn(
                query_graph["query_ids"], query_graph["pool_ids"], query_graph["neighbors"], query_graph["distances"], query_graph["weights"]
            )
        else:
            src_q = np.array([], dtype=np.int64)
            dst_q = np.array([], dtype=np.int64)
            d_q = np.array([], dtype=np.float32)
            w_q = np.array([], dtype=np.float32)

        edge_index = np.stack([np.concatenate([src_tr, src_val, src_q]), np.concatenate([dst_tr, dst_val, dst_q])], axis=0)
        edge_distance = np.concatenate([d_tr, d_val, d_q], axis=0)
        edge_weight = np.concatenate([w_tr, w_val, w_q], axis=0)
        if edge_direction == "undirected":
            edge_index, edge_weight, edge_distance = _symmetrize_weighted_edges(
                edge_index, edge_weight, len(patient_ids), edge_distance=edge_distance
            )

        df_sp = df[df["patient_id"].isin(patient_ids)].copy()
        df_sp = df_sp.set_index("patient_id").loc[patient_ids].reset_index(drop=True)
        sp = combined_name
    else:
        if combine_splits:
            sp = combined_name
            df_sp = df[df["split"].isin(splits)].copy()
        else:
            sp = "train"
            df_sp = df[df["split"] == sp].copy()

    if df_sp[feature_cols].isna().any().any():
        bad = df_sp[feature_cols].isna().mean().sort_values(ascending=False)
        raise ValueError(f"NaNs found in feature columns for split={sp}. Missing fractions:\n{bad}")

    if not inductive:
        graph = build_knn_graph(
            df_split=df_sp,
            feature_cols=feature_cols,
            k=k,
            metric=metric,
            weight_mode=weight_mode,
            temperature=temperature,
            rbf_sigma=rbf_sigma,
            edge_direction=edge_direction,
        )
        patient_ids = graph["patient_ids"]
        edge_index = graph["edge_index"]
        edge_weight = graph["edge_weight"]
        edge_distance = graph["edge_distance"]
        k_eff = graph["k_eff"]
        nbrs = graph["neighbors"]
        dists = graph["distances"]
    else:
        graph = None
        k_eff = min(k, len(df[df["split"] == "train"]))
        edge_distance = edge_distance if "edge_distance" in locals() else np.zeros((0,), dtype=np.float32)
        nbrs = None
        dists = None

    patient_ids = [str(pid) for pid in patient_ids]
    N = len(patient_ids)

    has_brain = df_sp["has_brain"].astype(int).to_numpy()
    has_fundus = df_sp["has_fundus"].astype(int).to_numpy()

    is_anchor = has_brain == 1
    if emb_anchor_set is not None:
        pid_arr = np.array(patient_ids, dtype=str)
        is_anchor = np.logical_and(is_anchor, np.isin(pid_arr, list(emb_anchor_set)))

    is_fundus_only = np.logical_and(has_fundus == 1, has_brain == 0)

    anchor_count, anchor_conf = _anchor_coverage_from_edges(
        edge_index=edge_index,
        edge_weight=edge_weight,
        is_anchor=is_anchor,
        n_nodes=N,
    )

    deg = degree_stats(edge_index, N)
    print(f"\n=== Graph split={sp} ===")
    print(f"N={N}  k={k}  k_eff={k_eff}  metric={metric}  weight={weight_mode}  edge_direction={edge_direction}")
    print("Out-degree stats:", deg["out_deg"])
    print("In-degree stats :", deg["in_deg"])

    if is_fundus_only.any():
        fo_counts = anchor_count[is_fundus_only]
        fo_conf   = anchor_conf[is_fundus_only]
        print(f"Fundus-only nodes: {is_fundus_only.sum()}")
        print(f"Fundus-only anchor_count min/median/max: {fo_counts.min()}/{np.median(fo_counts)}/{fo_counts.max()}")
        print(f"Fundus-only anchor_conf  min/median/max: {fo_conf.min():.4f}/{np.median(fo_conf):.4f}/{fo_conf.max():.4f}")
        print(f"Fundus-only with ZERO anchors: {(fo_counts==0).sum()}")
    else:
        print("No fundus-only nodes in this split.")

    out_path = out_dir / f"graph_{sp}.npz"
    np.savez_compressed(
        out_path,
        patient_ids=_safe_np_str_array(patient_ids),
        feature_cols=_safe_np_str_array(feature_cols),
        k=np.int64(k),
        k_eff=np.int64(k_eff),
        metric=str(metric),
        weight_mode=str(weight_mode),
        edge_direction=str(edge_direction),
        temperature=np.float32(temperature),
        rbf_sigma=np.float32(rbf_sigma),
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_distance=edge_distance,
        neighbors=(nbrs.astype(np.int64) if nbrs is not None else np.zeros((0, 0), dtype=np.int64)),
        distances=(dists.astype(np.float32) if dists is not None else np.zeros((0, 0), dtype=np.float32)),
        is_anchor=is_anchor.astype(np.int8),
        is_fundus_only=is_fundus_only.astype(np.int8),
        anchor_count=anchor_count.astype(np.int64),
        anchor_conf=anchor_conf.astype(np.float32),
    )
    print(f"[DONE] Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
