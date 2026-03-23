from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from models.graphsage_imputer import GraphSAGEImputer
from utils.phase3_imputer_data import (
    load_graph_npz,
    load_splits,
    load_clinical_features,
    seed_all,
    save_json,
)
from utils.phase4_confidence import (
    mc_predict_embeddings,
    summary_stats,
    uncertainty_to_confidence,
)
from utils.run_config import (
    get_run_root,
    is_unified_config,
    load_yaml,
    phase_dir,
    resolve_path,
    write_run_config,
)




def _load_priors_npz_with_key(path: Path, patient_ids: list[str], key: str) -> np.ndarray:
    npz = np.load(path, allow_pickle=True)
    if "patient_ids" not in npz.files or key not in npz.files:
        raise ValueError(f"Invalid priors npz: {path}")
    pids = [str(pid).strip() for pid in npz["patient_ids"].astype(str).tolist()]
    emb = npz[key].astype(np.float32)
    if pids == patient_ids:
        return emb
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    idx = [pid_to_idx.get(pid, None) for pid in patient_ids]
    if any(i is None for i in idx):
        raise ValueError(f"Priors missing patient_ids for {path}")
    return emb[idx]


def _load_priors_npz(path: Path, patient_ids: list[str]) -> np.ndarray:
    return _load_priors_npz_with_key(path, patient_ids, "embeddings")


def _infer_out_dim_from_ckpt(model_state: dict) -> int:
    last_idx = -1
    last_shape = None
    for k, v in model_state.items():
        if k.endswith(".lin.weight") and k.startswith("layers."):
            parts = k.split(".")
            try:
                idx = int(parts[1])
            except (ValueError, IndexError):
                continue
            if idx > last_idx:
                last_idx = idx
                last_shape = v.shape
    if last_shape is None:
        raise ValueError("Could not infer out_dim from checkpoint")
    return int(last_shape[0])


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    unified = is_unified_config(cfg)
    if unified:
        run_cfg = cfg["run"]
        phase_cfg = cfg["phase4"]
        run_root = get_run_root(cfg)
        run_dir = phase_dir(run_root, "phase4")
    else:
        run_cfg = cfg["run"]
        phase_cfg = cfg
        run_root = None
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = run_cfg.get("name", f"phase4_conf_{ts}")
        run_dir = Path(run_cfg.get("output_root", "logs/phase4_confidence")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if unified and run_root is not None:
        write_run_config(run_root, config_path)
    (run_dir / "config.yaml").write_text(Path(config_path).read_text())

    data_cfg = dict(phase_cfg["data"])
    model_cfg = phase_cfg["model"]
    phase3_cfg = dict(phase_cfg["phase3"])
    mc_cfg = phase_cfg["mc"]
    conf_cfg = phase_cfg["confidence"]

    if unified and run_root is not None:
        data_cfg["graph_npz"] = resolve_path(
            data_cfg.get("graph_npz"),
            phase_dir(run_root, "graphs") / "graph_trainval_inductive.npz",
        )
        phase3_cfg["ckpt_path"] = resolve_path(
            phase3_cfg.get("ckpt_path"),
            phase_dir(run_root, "phase3") / "best.pt",
        )

    seed_all(int(run_cfg.get("seed", 0)))

    graph = load_graph_npz(data_cfg["graph_npz"])
    splits = load_splits(data_cfg["splits_csv"], graph.patient_ids)

    clinical = load_clinical_features(
        path=data_cfg["clinical_csv"],
        patient_ids=graph.patient_ids,
        train_mask_all=splits.train_mask_all,
        clinical_cols=data_cfg.get("clinical_cols"),
        exclude_cols=data_cfg.get("clinical_exclude_cols"),
        binary_cols=data_cfg.get("binary_cols"),
        cont_cols=data_cfg.get("cont_cols"),
        label_col=data_cfg.get("label_col", "htn"),
    )

    ckpt = torch.load(phase3_cfg["ckpt_path"], map_location="cpu")
    model_state = ckpt["model_state"]

    out_dim_cfg = model_cfg.get("out_dim")
    out_dim = _infer_out_dim_from_ckpt(model_state)
    if out_dim_cfg is not None and int(out_dim_cfg) != int(out_dim):
        raise ValueError(f"out_dim mismatch: ckpt={out_dim} cfg={out_dim_cfg}")

    model = GraphSAGEImputer(
        in_dim=clinical.x.shape[1],
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        out_dim=int(out_dim),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_mlp=bool(model_cfg.get("use_mlp", False)),
        mlp_dim=model_cfg.get("mlp_dim"),
        normalize_output=bool(model_cfg.get("normalize_output", True)),
    )
    model.load_state_dict(model_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    k = int(mc_cfg.get("K", 30))
    if k < 2:
        raise ValueError("mc.K must be >= 2")

    mu, u = mc_predict_embeddings(
        model=model,
        x=clinical.x,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        k=k,
        device=device,
        normalize_each_pass=bool(mc_cfg.get("normalize_each_pass", True)),
        eps=float(conf_cfg.get("eps", 1e-12)),
    )

    u_np = u.cpu().numpy().astype(np.float32)
    conf, conf_meta = uncertainty_to_confidence(
        u_np,
        p_high=float(conf_cfg.get("percentile_high", 95)),
        eps=float(conf_cfg.get("eps", 1e-12)),
    )

    summary = {
        "priors": {},
        "mc": {
            "K": k,
            "normalize_each_pass": bool(mc_cfg.get("normalize_each_pass", True)),
            "dropout": float(model_cfg.get("dropout", 0.0)),
        },
        "uncertainty_stats": summary_stats(u_np),
        "confidence_stats": summary_stats(conf),
        "confidence_mapping": conf_meta,
    }

    if float(u_np.max()) <= 1e-8:
        print("[WARN] uncertainty near-zero; dropout may be disabled or ineffective")

    out_path = run_dir / "brain_priors.npz"
    np.savez_compressed(
        out_path,
        patient_ids=np.array(graph.patient_ids, dtype=str),
        embeddings=mu.cpu().numpy().astype(np.float32),
        uncertainty=u_np,
        confidence=conf.astype(np.float32),
    )

    summary["priors"]["imputed"] = {
        "path": str(out_path),
        "uncertainty_stats": summary_stats(u_np),
        "confidence_stats": summary_stats(conf),
        "confidence_mapping": conf_meta,
    }

    # baselines from phase3 priors (mean/random/shuffled/class_mean)
    priors_dir = phase3_cfg.get("priors_dir")
    if priors_dir is None:
        priors_dir = Path(phase3_cfg["ckpt_path"]).parent
    priors_dir = Path(priors_dir)

    baseline_files = {
        "mean": "imputed_brain_priors_mean.npz",
        "random": "imputed_brain_priors_random.npz",
        "shuffled": "imputed_brain_priors_shuffled.npz",
        "class_mean": "imputed_brain_priors_class_mean.npz",
        "local_pos": "imputed_brain_priors_local_pos.npz",
        "local_neg": "imputed_brain_priors_local_neg.npz",
    }

    processed = set()
    for key, fname in baseline_files.items():
        bpath = priors_dir / fname
        if not bpath.exists():
            continue
        emb = _load_priors_npz(bpath, graph.patient_ids)
        u_base = np.zeros(emb.shape[0], dtype=np.float32)
        conf_base, conf_meta_base = uncertainty_to_confidence(
            u_base,
            p_high=float(conf_cfg.get("percentile_high", 95)),
            eps=float(conf_cfg.get("eps", 1e-12)),
        )
        out_b = run_dir / f"brain_priors_{key}.npz"
        np.savez_compressed(
            out_b,
            patient_ids=np.array(graph.patient_ids, dtype=str),
            embeddings=emb.astype(np.float32),
            uncertainty=u_base,
            confidence=conf_base.astype(np.float32),
        )
        summary["priors"][key] = {
            "path": str(out_b),
            "uncertainty_stats": summary_stats(u_base),
            "confidence_stats": summary_stats(conf_base),
            "confidence_mapping": conf_meta_base,
        }
        processed.add(key)

    posneg_path = priors_dir / "imputed_brain_priors_local_posneg.npz"
    if posneg_path.exists():
        if "local_pos" not in processed:
            emb = _load_priors_npz_with_key(posneg_path, graph.patient_ids, "embeddings_pos")
            u_base = np.zeros(emb.shape[0], dtype=np.float32)
            conf_base, conf_meta_base = uncertainty_to_confidence(
                u_base,
                p_high=float(conf_cfg.get("percentile_high", 95)),
                eps=float(conf_cfg.get("eps", 1e-12)),
            )
            out_b = run_dir / "brain_priors_local_pos.npz"
            np.savez_compressed(
                out_b,
                patient_ids=np.array(graph.patient_ids, dtype=str),
                embeddings=emb.astype(np.float32),
                uncertainty=u_base,
                confidence=conf_base.astype(np.float32),
            )
            summary["priors"]["local_pos"] = {
                "path": str(out_b),
                "uncertainty_stats": summary_stats(u_base),
                "confidence_stats": summary_stats(conf_base),
                "confidence_mapping": conf_meta_base,
            }
        if "local_neg" not in processed:
            emb = _load_priors_npz_with_key(posneg_path, graph.patient_ids, "embeddings_neg")
            u_base = np.zeros(emb.shape[0], dtype=np.float32)
            conf_base, conf_meta_base = uncertainty_to_confidence(
                u_base,
                p_high=float(conf_cfg.get("percentile_high", 95)),
                eps=float(conf_cfg.get("eps", 1e-12)),
            )
            out_b = run_dir / "brain_priors_local_neg.npz"
            np.savez_compressed(
                out_b,
                patient_ids=np.array(graph.patient_ids, dtype=str),
                embeddings=emb.astype(np.float32),
                uncertainty=u_base,
                confidence=conf_base.astype(np.float32),
            )
            summary["priors"]["local_neg"] = {
                "path": str(out_b),
                "uncertainty_stats": summary_stats(u_base),
                "confidence_stats": summary_stats(conf_base),
                "confidence_mapping": conf_meta_base,
            }

    groups = splits.imaging_group
    group_summary = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.any():
            group_summary[str(g)] = {
                "count": int(mask.sum()),
                "confidence_mean": float(conf[mask].mean()),
            }
    summary["confidence_by_imaging_group"] = group_summary

    save_json(run_dir / "confidence_summary.json", summary)

    print(f"[DONE] saved: {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
