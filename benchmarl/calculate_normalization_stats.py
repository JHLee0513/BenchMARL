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

def calculate_normalization_stats(
    dataset_dir: Path,
    max_shards: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculate normalization statistics for states and rewards.

    Args:
        dataset_dir: Path to dataset directory containing meta.json, index.jsonl, and .pt shards
        max_shards: Optional limit on number of shards to process

    Returns:
        Dictionary containing statistics per group
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    meta = _read_json(dataset_dir / "meta.json")
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
            "reward": torch.tensor([], dtype=torch.float32),
        }
        for group in groups
    }

    # Process all shards
    total_samples = {group: 0 for group in groups}
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
                # Concatenate current and next states
                states = torch.cat([obs_flat, next_obs_flat], dim=0)

                # Get rewards
                rew = tdg.get(("next", group, "reward"))
                rew_flat = _as_2d_features(rew).to(torch.float32)

                # Update statistics
                samples[group]["state"] = torch.cat([samples[group]["state"], states], dim=0)
                samples[group]["reward"] = torch.cat([samples[group]["reward"], rew_flat], dim=0)
                total_samples[group] += int(states.shape[0])

            except Exception as e:
                # Skip this group in this shard if keys are missing
                continue

    # Convert to output format
    result: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "meta": meta,
        "num_shards_processed": len(shards),
        "groups": {},
    }

    for group in groups:
        result["groups"][group] = {
            "state": {
                "mean": samples[group]["state"].mean(dim=0).tolist(),
                "std": samples[group]["state"].std(dim=0).tolist(),
            },
            "reward": {
                "mean": samples[group]["reward"].mean(dim=0).tolist(),
                "std": samples[group]["reward"].std(dim=0).tolist(),
            },
            "total_samples": total_samples[group],
        }

    return result


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
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file path (default: <dataset_dir>/normalization_stats.json)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Optional limit on number of shards to process",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_file = (
        Path(args.output_file)
        if args.output_file
        else dataset_dir / "normalization_stats.json"
    )

    print(f"Calculating normalization statistics from: {dataset_dir}")
    stats = calculate_normalization_stats(
        dataset_dir=dataset_dir,
        max_shards=args.max_shards,
    )

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to: {output_file}")
    print("\nSummary:")
    for group, group_stats in stats["groups"].items():
        print(f"\n  Group: {group}")
        print(f"    Total samples: {group_stats['total_samples']}")
        state_stats = group_stats["state"]
        reward_stats = group_stats["reward"]
        print(f"    State - Mean: {state_stats.get('mean', [None])[0] if state_stats.get('mean') else None:.6f}, "
              f"Shape: {len(state_stats.get('mean', []))}, "
              f"Std: {state_stats.get('std', [None])[0] if state_stats.get('std') else None:.6f}")
        print(f"    Reward - Mean: {reward_stats.get('mean', [None])[0] if reward_stats.get('mean') else None:.6f}, "
              f"Shape: {len(reward_stats.get('mean', []))}, "
              f"Std: {reward_stats.get('std', [None])[0] if reward_stats.get('std') else None:.6f}")

if __name__ == "__main__":
    main()
