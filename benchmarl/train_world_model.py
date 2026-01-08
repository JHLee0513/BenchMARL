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
    return (
        (group, "observation") in keys
        and (group, "action") in keys
        and ("next", group, "observation") in keys
        and ("next", group, "reward") in keys
        and ("next", group, "done") in keys
    )


def _sample_minibatch(td: TensorDictBase, batch_size: int, *, generator: torch.Generator) -> TensorDictBase:
    n = int(td.batch_size[0]) if hasattr(td, "batch_size") and len(td.batch_size) else int(td.numel())
    batch_size = min(max(1, int(batch_size)), max(1, n))
    idx = torch.randint(0, n, (batch_size,), device="cpu", generator=generator)
    return td[idx]


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

    save_path = Path(wm_cfg.save_path)
    if not save_path.is_absolute():
        save_path = (Path(HydraConfig.get().runtime.output_dir) / save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # RNG dedicated to offline world-model training for reproducibility
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(cfg.seed) + 202_601_08)

    total_updates = 0
    for epoch in range(max(1, int(wm_cfg.epochs))):
        for shard_i, shard_path in enumerate(shards):
            td: TensorDictBase = torch.load(shard_path, map_location=wm_cfg.load_device)
            td = td.detach()
            try:
                td = td.reshape(-1)
            except Exception:
                pass

            # Train per group (MBPO trains per group model ensemble).
            for group in experiment.group_map.keys():
                if not _required_keys_present(td, group):
                    continue
                for u in range(max(1, int(wm_cfg.updates_per_shard))):
                    mb = _sample_minibatch(td, batch_size, generator=rng)
                    algo._train_dynamics(group, mb)  # type: ignore[attr-defined]
                    total_updates += 1
                    if wm_cfg.log_every and (total_updates % int(wm_cfg.log_every) == 0):
                        # Best-effort: print current mean loss across ensemble if available.
                        losses = getattr(algo, "_model_losses", {}).get(group, None)
                        if losses is not None:
                            try:
                                mean_loss = float(losses.mean().detach().cpu().item())
                                print(
                                    f"[train_world_model] epoch={epoch} shard={shard_i}/{len(shards)} "
                                    f"group={group} updates={total_updates} mean_loss={mean_loss:.6f}"
                                )
                            except Exception:
                                pass

    algo.save_world_model(str(save_path))  # type: ignore[attr-defined]
    print(f"\n[train_world_model] Saved world model to: {str(save_path)}")


if __name__ == "__main__":
    hydra_train_world_model()


