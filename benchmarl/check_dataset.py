#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""
Calculate mean and std statistics for state and reward normalization from offline datasets.

This script reads datasets produced by `benchmarl/collect.py` and computes:
- State normalization stats (mean, std) for observations
- Reward normalization stats (mean, std) for rewards

The statistics are computed separately for each agent group in the dataset.

Example:
    python benchmarl/calculate_normalization_stats.py \
        --dataset-dir dataset/mappo_2agent_5seeds_merged \
        --output-file normalization_stats.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


NestedKey = Tuple[Any, ...]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_shards(dataset_dir: Path, max_shards: Optional[int]) -> List[Path]:
    """List all shard files from index.jsonl or glob pattern."""
    index = dataset_dir / "index.jsonl"
    shards: List[Path] = []
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rel = str(rec.get("path", "")).strip()
            except Exception:
                continue
            if not rel:
                continue
            p = (dataset_dir / rel).resolve()
            if p.exists():
                shards.append(p)
    else:
        shards = sorted(dataset_dir.glob("batch_*.pt"))
    if max_shards is not None:
        shards = shards[: max(0, int(max_shards))]
    return shards


def _discover_groups(keys: Sequence[NestedKey]) -> List[str]:
    """Discover agent groups from TensorDict keys."""
    groups: set[str] = set()
    for k in keys:
        if not isinstance(k, tuple) or len(k) < 2:
            continue
        if k[1] in ("observation", "action"):
            if isinstance(k[0], str):
                groups.add(k[0])
    return sorted(groups)


def _ensure_group_reward_done(td: Any, group: str) -> Any:
    """
    Ensure group-level reward/done keys exist by expanding top-level keys when needed.
    """
    try:
        keys = td.keys(True, True)
    except Exception:
        return td

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


def _as_2d_features(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to [N, D] where D is the last dimension (or 1 for scalars).
    Flattens all leading dimensions into N.
    """
    if not torch.is_tensor(x):
        raise TypeError("Expected torch.Tensor")
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    # Keep first dim as batch dim; flatten everything else.
    return x.reshape(x.shape[0], -1)

def check_dataset(
    dataset_dir: Path,
    max_shards: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Check the dataset for completeness and correctness.
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    shards = _list_shards(dataset_dir, max_shards=max_shards)
    if not shards:
        raise ValueError(f"No shard files found in: {str(dataset_dir)}")

    # Discover groups from first shard
    first = torch.load(shards[0], map_location="cpu", weights_only=False)
    try:
        keys = list(first.keys(True, True))
    except Exception:
        keys = []
    groups = _discover_groups(keys)
    if not groups:
        raise ValueError("No agent groups found in dataset")

    print(f"Found {len(groups)} group(s): {groups}")
    print(f"Processing {len(shards)} shard(s)...")

    samples: Dict[str, Dict[str, torch.Tensor]] = {
        group: {
            "state": torch.tensor([], dtype=torch.float32),
            "action": torch.tensor([], dtype=torch.float32),
            "next_state": torch.tensor([], dtype=torch.float32),
            "reward": torch.tensor([], dtype=torch.float32),
        }
        for group in groups
    }

    # Process all shards
    for shard_idx, shard_path in enumerate(shards):
        if (shard_idx + 1) % 10 == 0:
            print(f"  Processed {shard_idx + 1}/{len(shards)} shards...")

        td = torch.load(shard_path, map_location="cpu", weights_only=False)
        td = td.detach()
        try:
            td = td.reshape(-1)
        except Exception:
            continue

        for group in groups:
            tdg = _ensure_group_reward_done(td, group)
            try:
                # Get observations (current and next states)
                obs = tdg.get((group, "observation"))
                next_obs = tdg.get(("next", group, "observation"))
                # Combine current and next observations for state statistics
                # Flatten to [N, D] format
                obs_flat = _as_2d_features(obs).to(torch.float32)
                next_obs_flat = _as_2d_features(next_obs).to(torch.float32)
                action_flat = _as_2d_features(tdg.get((group, "action"))).to(torch.float32)
                # Concatenate current and next states

                # Get rewards
                rew = tdg.get(("next", group, "reward"))
                rew_flat = _as_2d_features(rew).to(torch.float32)

                # Update statistics
                samples[group]["state"] = torch.cat([samples[group]["state"], obs_flat], dim=0)
                samples[group]["action"] = torch.cat([samples[group]["action"], action_flat], dim=0)
                samples[group]["next_state"] = torch.cat([samples[group]["next_state"], next_obs_flat], dim=0)
                samples[group]["reward"] = torch.cat([samples[group]["reward"], rew_flat], dim=0)

            except Exception as e:
                # Skip this group in this shard if keys are missing
                continue

    for group in groups:
        state_mean = samples[group]["state"].mean(dim=0)
        state_std = samples[group]["state"].std(dim=0)
        action_mean = samples[group]["action"].mean(dim=0)
        action_std = samples[group]["action"].std(dim=0)
        next_state_mean = samples[group]["next_state"].mean(dim=0)
        next_state_std = samples[group]["next_state"].std(dim=0)
        reward_mean = samples[group]["reward"].mean(dim=0)
        reward_std = samples[group]["reward"].std(dim=0)

        state_normalized = (samples[group]["state"] - state_mean) / (state_std + 1e-6)
        action_normalized = (samples[group]["action"] - action_mean) / (action_std + 1e-6)
        next_state_normalized = (samples[group]["next_state"] - next_state_mean) / (next_state_std + 1e-6)
        reward_normalized = (samples[group]["reward"] - reward_mean) / (reward_std + 1e-6)

        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=5000, random_state=42)
        x = torch.cat([state_normalized, action_normalized], dim=-1)
        kmeans.fit(x.detach().cpu().numpy())
        indices = kmeans.predict(x.detach().cpu().numpy())

        # Calculate mean, std of reward and next state for each cluster
        import numpy as np

        n_clusters = kmeans.n_clusters

        # Prepare arrays for reward and next_state (already normalized, but use originals for correct stats)
        rewards = samples[group]["reward"]  # [num_samples, reward_dim]
        next_states = samples[group]["next_state"]  # [num_samples, next_state_dim]


        cnt = 0
        cluster_stats = {}
        from tqdm import tqdm
        for cid in tqdm(range(n_clusters)[::10]):
            cluster_mask = (indices == cid)
            if torch.is_tensor(cluster_mask):
                cluster_mask = cluster_mask.cpu().numpy()
            if np.sum(cluster_mask) == 0:
                # No samples in this cluster
                cluster_stats[cid] = {
                    "reward_mean": None, "reward_std": None,
                    "next_state_mean": None, "next_state_std": None,
                    "count": 0,
                }
                continue
            reward_cluster = rewards[cluster_mask]
            next_state_cluster = next_states[cluster_mask]
            reward_mean = reward_cluster.mean(dim=0).cpu().numpy().tolist()
            reward_std = reward_cluster.std(dim=0).cpu().numpy().tolist()
            next_state_mean = next_state_cluster.mean(dim=0).cpu().numpy().tolist()
            next_state_std = next_state_cluster.std(dim=0).cpu().numpy().tolist()

            ratios = []
            for i_ in range(len(reward_mean)):
                ratios.append(abs(reward_std[i_] / (reward_mean[i_] + 1e-6)))

            if max(ratios) > 100:
                continue
            cluster_stats[cnt] = {
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "next_state_mean": next_state_mean,
                "next_state_std": next_state_std,
                "count": int(cluster_mask.sum()),
            }
            cnt += 1

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 5))
        # plt.bar(range(cnt), [cluster_stats[x]['reward_std'][0] / (cluster_stats[x]['reward_mean'][0] + 1e-6) for x in range(cnt)])
        # plt.savefig(f"/home/joonho/BenchMARL/tmp/reward_std_per_cluster_2agent.png")

        # # Collect all reward_std[0] values across clusters for the distribution plot
        # reward_std_values = [cluster_stats[x]['reward_std'][0] for x in range(cnt) if cluster_stats[x]['reward_std'][0] < 0.1]

        # plt.figure(figsize=(8, 5))
        # plt.hist(reward_std_values, bins=30, color='blue', alpha=0.7, edgecolor='black')
        # plt.xlabel('reward_std[0]')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of reward_std[0] Across Clusters')
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.savefig("/home/joonho/BenchMARL/tmp/reward_std_distribution_2agent.png")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate normalization statistics for offline datasets"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory (should contain meta.json, index.jsonl, and .pt files)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Optional limit on number of shards to process",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    print(f"Calculating normalization statistics from: {dataset_dir}")
    check_dataset(
        dataset_dir=dataset_dir,
        max_shards=args.max_shards,
    )

if __name__ == "__main__":
    main()
