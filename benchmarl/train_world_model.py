#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""
Train MBPO-style world models (dynamics ensemble) from an offline dataset.

This script is designed to work with datasets produced by `benchmarl/collect.py`:
- a directory containing `meta.json`, `index.jsonl` (optional), and `batch_*.pt` shards
- each shard is a TensorDict containing transitions with keys like:
  (group, "observation"), (group, "action"), ("next", group, "observation"),
  ("next", group, "reward"), ("next", group, "done")

High-level behavior:
- Instantiate an Experiment from Hydra config (like `benchmarl/run.py`) BUT do not train the policy.
- Use the algorithm instance's internal world-model trainer (`_train_dynamics`) on offline batches.
- Save the world model checkpoint via `algorithm.save_world_model(...)`.

Example
-------

1) Collect data:

    python benchmarl/collect.py algorithm=mbpo_mappo task=vmas/navigation \
      ++collect.policy=random ++collect.total_frames=200000

2) Train world model:

    python benchmarl/train_world_model.py algorithm=mbpo_mappo task=vmas/navigation \
      ++world_model.dataset_dir=/path/to/run/dataset \
      ++world_model.save_path=/path/to/world_model.pt \
      ++world_model.epochs=5 ++world_model.updates_per_shard=50
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase

from benchmarl.hydra_config import load_experiment_from_hydra


@dataclass
class WorldModelTrainConfig:
    dataset_dir: str = ""
    save_path: str = "world_model.pt"

    # Training schedule
    epochs: int = 1
    updates_per_shard: int = 50
    batch_size: Optional[int] = None  # defaults to algorithm.model_batch_size

    # Optional: cap shards read (debug)
    max_shards: Optional[int] = None

    # Device for loaded shards before sampling (they'll be moved as needed by algorithm).
    load_device: str = "cpu"

    # Logging
    log_every: int = 10

    # Validation
    val_ratio: float = 0.1  # fraction of shards reserved for validation
    eval_every_updates: int = 200  # run validation every N gradient updates
    val_batches: int = 10  # number of minibatches to evaluate per group
    val_batch_size: Optional[int] = None  # defaults to batch_size
    update_elites_from_val: bool = True  # update elite indices using validation losses
    
    # Uncertainty threshold computation
    compute_uncertainty_threshold: bool = False  # compute uncertainty thresholds from validation data
    uncertainty_threshold_percentile: float = 90.0  # percentile to use as threshold (e.g., 90 = 90th percentile)
    uncertainty_metric: str = "total_rew_unc"  # which uncertainty metric to use: 'epistemic_obs_var', 'epistemic_rew_std', 'aleatoric_obs_logvar', 'aleatoric_rew_logvar', 'total_obs_unc', 'total_rew_unc'


def _get_wm_cfg(cfg: DictConfig) -> WorldModelTrainConfig:
    d = {}
    if "world_model" in cfg:
        try:
            d = OmegaConf.to_container(cfg.world_model, resolve=True)  # type: ignore[attr-defined]
        except Exception:
            d = {}
    valid = set(WorldModelTrainConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    d = {k: v for k, v in (d or {}).items() if k in valid}
    return WorldModelTrainConfig(**d)


def _list_shards(dataset_dir: Path, max_shards: Optional[int]) -> List[Path]:
    index = dataset_dir / "index.jsonl"
    shards: List[Path] = []
    if index.exists():
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            p = dataset_dir / rec["path"]
            if p.exists():
                shards.append(p)
    else:
        shards = sorted(dataset_dir.glob("batch_*.pt"))
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    return shards


def _required_keys_present(td: TensorDictBase, group: str) -> bool:
    keys = td.keys(True, True)
    # Some collectors/envs store reward/done at the top level ("next","reward"/"next","done")
    # instead of under the group. We'll accept either and later expand to group-level keys.
    has_rew = ("next", group, "reward") in keys or ("next", "reward") in keys
    has_done = ("next", group, "done") in keys or ("next", "done") in keys
    return (
        (group, "observation") in keys
        and (group, "action") in keys
        and ("next", group, "observation") in keys
        and has_rew
        and has_done
    )


def _ensure_group_reward_done(td: TensorDictBase, group: str) -> TensorDictBase:
    """Ensure group-level reward/done keys exist by expanding top-level keys when needed."""
    keys = td.keys(True, True)
    # ("next", group, "reward")
    if ("next", group, "reward") not in keys and ("next", "reward") in keys:
        try:
            rew = td.get(("next", "reward"))
            rew = rew.expand(td.get(group).shape).unsqueeze(-1)
            td.set(("next", group, "reward"), rew)
        except Exception:
            pass
    # ("next", group, "done")
    if ("next", group, "done") not in keys and ("next", "done") in keys:
        try:
            done = td.get(("next", "done"))
            done = done.expand(td.get(group).shape).unsqueeze(-1)
            td.set(("next", group, "done"), done)
        except Exception:
            pass
    return td


def _sample_minibatch(td: TensorDictBase, batch_size: int, *, generator: torch.Generator) -> TensorDictBase:
    n = int(td.batch_size[0]) if hasattr(td, "batch_size") and len(td.batch_size) else int(td.numel())
    batch_size = min(max(1, int(batch_size)), max(1, n))
    idx = torch.randint(0, n, (batch_size,), device="cpu", generator=generator)
    return td[idx]


def _split_train_val_shards(
    shards: List[Path], val_ratio: float, *, seed: int
) -> Tuple[List[Path], List[Path]]:
    val_ratio = float(val_ratio)
    val_ratio = min(max(val_ratio, 0.0), 0.9)
    if not shards:
        return [], []
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 991_733)
    perm = torch.randperm(len(shards), generator=g).tolist()
    shards_shuf = [shards[i] for i in perm]
    n_val = int(round(len(shards_shuf) * val_ratio))
    n_val = max(0, min(n_val, len(shards_shuf) - 1))  # keep at least 1 train shard
    val = shards_shuf[:n_val]
    train = shards_shuf[n_val:]
    return train, val


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_train_world_model(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    wm_cfg = _get_wm_cfg(cfg)
    if not wm_cfg.dataset_dir:
        raise ValueError("Missing required: ++world_model.dataset_dir=/path/to/dataset")

    dataset_dir = Path(wm_cfg.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {str(dataset_dir)}")

    # Make an experiment (to build env specs + algorithm world model), but disable logging and evaluation.
    # We keep this minimal to avoid side-effects and overhead.
    try:
        cfg.experiment.loggers = []
        cfg.experiment.create_json = False
        cfg.experiment.evaluation = False
        cfg.experiment.render = False
    except Exception:
        pass

    print(f"\n[train_world_model] Algorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))
    print("\nWorld-model training config:\n")
    print(OmegaConf.to_yaml(OmegaConf.structured(wm_cfg)))

    experiment = load_experiment_from_hydra(cfg, task_name=task_name, algorithm_name=algorithm_name)
    algo = experiment.algorithm

    if not hasattr(algo, "_train_dynamics"):
        raise TypeError(f"Algorithm {algorithm_name} does not expose _train_dynamics; cannot train world model.")
    if not hasattr(algo, "save_world_model"):
        raise TypeError(f"Algorithm {algorithm_name} does not expose save_world_model; cannot save world model.")

    # Batch size default
    batch_size = int(wm_cfg.batch_size or getattr(algo, "model_batch_size", 256) or 256)
    batch_size = max(1, batch_size)

    shards = _list_shards(dataset_dir, wm_cfg.max_shards)
    if not shards:
        raise ValueError(f"No shards found in {str(dataset_dir)}")
    train_shards, val_shards = _split_train_val_shards(
        shards, wm_cfg.val_ratio, seed=int(cfg.seed)
    )
    if not train_shards:
        raise ValueError("No training shards after train/val split.")

    save_path = Path(wm_cfg.save_path)
    if not save_path.is_absolute():
        save_path = (Path(HydraConfig.get().runtime.output_dir) / save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # RNG dedicated to offline world-model training for reproducibility
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(cfg.seed) + 202_601_08)

    total_updates = 0
    import random
    random.seed(int(cfg.seed) + 202_601_08)
    
    # Collect uncertainty values across all validation batches for threshold computation
    per_group_uncertainty_values: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    
    for epoch in range(max(1, int(wm_cfg.epochs))):
        random.shuffle(train_shards)
        for shard_i, shard_path in enumerate(train_shards):
            td: TensorDictBase = torch.load(shard_path, map_location=wm_cfg.load_device, weights_only=False)
            td = td.detach()
            try:
                td = td.reshape(-1)
            except Exception:
                # If a shard has unexpected structure, skip it rather than breaking the run.
                continue

            # Train per group (MBPO trains per group model ensemble).
            for group in experiment.group_map.keys():
                if not _required_keys_present(td, group):
                    continue
                for u in range(max(1, int(wm_cfg.updates_per_shard))):
                    mb = _sample_minibatch(td, batch_size, generator=rng)
                    mb = _ensure_group_reward_done(mb, group)
                    algo._train_dynamics(group, mb)  # type: ignore[attr-defined]
                    total_updates += 1

                    # Periodic validation.
                    if (
                        val_shards
                        and int(wm_cfg.eval_every_updates) > 0
                        and (total_updates % int(wm_cfg.eval_every_updates) == 0)
                        and hasattr(algo, "_world_model_eval_losses")
                    ):
                        val_bs = int(wm_cfg.val_batch_size or batch_size)
                        val_bs = max(1, val_bs)
                        n_val_batches = max(1, int(wm_cfg.val_batches))

                        per_group_losses: Dict[str, List[torch.Tensor]] = {}
                        per_group_metrics_sum: Dict[str, Dict[str, float]] = {}
                        per_group_metrics_n: Dict[str, int] = {}

                        for _ in range(n_val_batches):
                            shard_path_v = val_shards[
                                int(torch.randint(0, len(val_shards), (1,), generator=rng).item())
                            ]
                            tdv: TensorDictBase = torch.load(
                                shard_path_v, map_location=wm_cfg.load_device, weights_only=False
                            ).detach()
                            try:
                                tdv = tdv.reshape(-1)
                            except Exception:
                                continue

                            for g in experiment.group_map.keys():
                                if not _required_keys_present(tdv, g):
                                    continue
                                mbv = _sample_minibatch(tdv, val_bs, generator=rng)
                                mbv = _ensure_group_reward_done(mbv, g)
                                losses_per_model, metrics = algo._world_model_eval_losses(g, mbv)  # type: ignore[attr-defined]
                                per_group_losses.setdefault(g, []).append(losses_per_model.detach())
                                per_group_metrics_sum.setdefault(g, {})
                                for k, v in metrics.items():
                                    per_group_metrics_sum[g][k] = per_group_metrics_sum[g].get(k, 0.0) + float(v)
                                per_group_metrics_n[g] = per_group_metrics_n.get(g, 0) + 1
                                
                                # Collect per-sample uncertainty values for threshold computation
                                if (
                                    wm_cfg.compute_uncertainty_threshold
                                    and hasattr(algo, "_compute_per_sample_uncertainty")
                                ):
                                    try:
                                        unc_dict = algo._compute_per_sample_uncertainty(g, mbv)  # type: ignore[attr-defined]
                                        if g not in per_group_uncertainty_values:
                                            per_group_uncertainty_values[g] = {}
                                        for unc_key, unc_tensor in unc_dict.items():
                                            per_group_uncertainty_values[g].setdefault(unc_key, []).append(unc_tensor)
                                    except Exception as e:
                                        # Silently skip if uncertainty computation fails
                                        pass

                        for g, losses_list in per_group_losses.items():
                            losses_stack = torch.stack(losses_list, dim=0)  # [n, ensemble]
                            mean_losses = losses_stack.mean(dim=0)
                            if wm_cfg.update_elites_from_val and hasattr(algo, "_set_elites_from_losses"):
                                try:
                                    algo._set_elites_from_losses(g, mean_losses)  # type: ignore[attr-defined]
                                except Exception:
                                    pass

                            n = max(1, per_group_metrics_n.get(g, 0))
                            metrics_mean = {
                                k: v / n for k, v in per_group_metrics_sum.get(g, {}).items()
                            }
                            print(
                                f"[train_world_model][val] epoch={epoch} update={total_updates} group={g} "
                                f"loss_total={metrics_mean.get('eval/loss_total', float('nan')):.6f} "
                                f"obs_mse={metrics_mean.get('eval/mse_observation', float('nan')):.6f} "
                                f"rew_mse={metrics_mean.get('eval/mse_reward', float('nan')):.6f} "
                                f"obs_smape={metrics_mean.get('eval/smape_observation', float('nan')):.3f} "
                                f"rew_smape={metrics_mean.get('eval/smape_reward', float('nan')):.3f} "
                                f"obs_r2={metrics_mean.get('eval/r2_observation', float('nan')):.3f} "
                                f"rew_r2={metrics_mean.get('eval/r2_reward', float('nan')):.3f}"
                            )
                    if wm_cfg.log_every and (total_updates % int(wm_cfg.log_every) == 0):
                        # Best-effort: print current mean loss across ensemble if available.
                        losses = getattr(algo, "_model_losses", {}).get(group, None)
                        if losses is not None:
                            try:
                                mean_loss = float(losses.mean().detach().cpu().item())
                                print(
                                    f"[train_world_model] epoch={epoch} shard={shard_i}/{len(train_shards)} "
                                    f"group={group} updates={total_updates} mean_loss={mean_loss:.6f}"
                                )
                            except Exception:
                                pass

    # Compute uncertainty thresholds from collected validation data
    uncertainty_thresholds: Dict[str, Dict[str, float]] = {}
    if wm_cfg.compute_uncertainty_threshold and per_group_uncertainty_values:
        print("\n[train_world_model] Computing uncertainty thresholds from validation data...")
        threshold_percentile = float(wm_cfg.uncertainty_threshold_percentile)
        threshold_percentile = max(0.0, min(100.0, threshold_percentile))
        uncertainty_metric = str(wm_cfg.uncertainty_metric)
        
        for g, unc_dict in per_group_uncertainty_values.items():
            uncertainty_thresholds[g] = {}
            
            # Compute statistics for each uncertainty metric
            for unc_key, unc_tensors in unc_dict.items():
                if not unc_tensors:
                    continue
                # Concatenate all uncertainty values
                all_unc = torch.cat(unc_tensors, dim=0)
                all_unc = all_unc[torch.isfinite(all_unc)]
                
                if all_unc.numel() == 0:
                    continue
                
                # Compute statistics
                mean_unc = float(all_unc.mean().item())
                std_unc = float(all_unc.std().item())
                median_unc = float(all_unc.median().item())
                
                # Compute percentiles
                percentiles = [50.0, 75.0, 90.0, 95.0, 99.0]
                percentile_values = {}
                for p in percentiles:
                    q = torch.quantile(all_unc, p / 100.0, interpolation="linear")
                    percentile_values[f"p{p}"] = float(q.item())
                
                # Use the specified percentile as the threshold
                threshold = torch.quantile(all_unc, threshold_percentile / 100.0, interpolation="linear")
                threshold_value = float(threshold.item())
                
                uncertainty_thresholds[g][unc_key] = {
                    "threshold": threshold_value,
                    "mean": mean_unc,
                    "std": std_unc,
                    "median": median_unc,
                    "percentiles": percentile_values,
                    "n_samples": int(all_unc.numel()),
                }
                
                print(
                    f"  Group={g}, Metric={unc_key}: "
                    f"threshold={threshold_value:.6f} (p{threshold_percentile:.1f}), "
                    f"mean={mean_unc:.6f}, std={std_unc:.6f}, "
                    f"n_samples={all_unc.numel()}"
                )
        
        # Save thresholds to a separate file
        threshold_path = save_path.parent / f"{save_path.stem}_uncertainty_thresholds.json"
        # Convert to JSON-serializable format
        thresholds_json = {}
        for g, metrics in uncertainty_thresholds.items():
            thresholds_json[g] = {}
            for metric, stats in metrics.items():
                thresholds_json[g][metric] = {
                    k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in stats.items()
                }
        
        with open(threshold_path, "w") as f:
            json.dump(thresholds_json, f, indent=2)
        print(f"\n[train_world_model] Saved uncertainty thresholds to: {str(threshold_path)}")
        
        # Also save the primary threshold (using the specified metric) for easy access
        primary_thresholds = {}
        for g, metrics in uncertainty_thresholds.items():
            if uncertainty_metric in metrics:
                primary_thresholds[g] = metrics[uncertainty_metric]["threshold"]
        
        if primary_thresholds:
            primary_path = save_path.parent / f"{save_path.stem}_uncertainty_threshold_{uncertainty_metric}.json"
            with open(primary_path, "w") as f:
                json.dump(primary_thresholds, f, indent=2)
            print(f"[train_world_model] Saved primary thresholds ({uncertainty_metric}) to: {str(primary_path)}")
    
    algo.save_world_model(str(save_path))  # type: ignore[attr-defined]
    print(f"\n[train_world_model] Saved world model to: {str(save_path)}")


if __name__ == "__main__":
    hydra_train_world_model()


