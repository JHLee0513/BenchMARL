#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""
Collect off-policy datasets using BenchMARL environments and (optionally) existing algorithms as behavior policies.

This script mirrors the high-level Hydra flow of `benchmarl/run.py`, but instead of training it:
- creates the task + algorithm + models (for a reference policy, if requested)
- builds a TorchRL collector
- iterates collection and saves batches to disk as a dataset

Usage (examples)
---------------

Collect with a randomly-initialized reference policy (same as if you ran `run.py` but without training):

    python benchmarl/collect.py algorithm=mbpo_mappo task=vmas/navigation ++collect.policy=reference

Collect with a fully random policy:

    python benchmarl/collect.py algorithm=mbpo_mappo task=vmas/navigation ++collect.policy=random

Epsilon-mix between reference and random:

    python benchmarl/collect.py algorithm=mbpo_mappo task=vmas/navigation ++collect.policy=epsilon_greedy ++collect.epsilon=0.2

Gaussian noise around reference (continuous actions only):

    python benchmarl/collect.py algorithm=mbpo_mappo task=vmas/navigation ++collect.policy=gaussian_noise ++collect.noise_std=0.1
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, TensorDictBase
from torchrl.collectors import SyncDataCollector
from torchrl.data import Categorical, OneHot
from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type

import torch.nn.functional as F

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_experiment_config_from_hydra,
    load_model_config_from_hydra,
    load_task_config_from_hydra,
    reload_experiment_from_file,
)
from benchmarl.utils import _add_rnn_transforms


@dataclass
class CollectConfig:
    # Behavior policy: reference | random | epsilon_greedy | gaussian_noise
    policy: str = "reference"
    # If policy==epsilon_greedy: probability of taking random action (per env-agent element).
    epsilon: float = 0.1
    # If policy==gaussian_noise (continuous): stddev of additive gaussian noise.
    noise_std: float = 0.1
    # Use deterministic actions for reference policy (mean/mode) instead of sampling.
    deterministic: bool = False

    # Optional: load trained policy weights for the *single* reference policy case.
    # If provided, the script will reload the experiment from this checkpoint and use its policy.
    reference_restore_file: Optional[str] = None

    # Optional: use multiple behavior sources (comma-separated).
    # Example: "mappo,masac,random" or "ckpt:/path/a.pt,ckpt:/path/b.pt,random"
    policies: Optional[str] = None
    # Optional: policy weights (comma-separated floats). Only used with mix="weighted_random".
    policy_weights: Optional[str] = None
    # How to mix when multiple policies are provided: round_robin | weighted_random
    mix: str = "round_robin"

    # Collection sizes
    n_envs: Optional[int] = None
    frames_per_batch: Optional[int] = None
    total_frames: int = 1_000_000
    init_random_frames: int = 0

    # Storage
    output_dir: Optional[str] = None
    flatten_batch: bool = True
    # Dataset format:
    # - "flat": save a flat batch (old behavior; equivalent to flatten_batch=True)
    # - "trajectory": save the unflattened collector batch (typically [T, B, ...])
    # - "flat_with_history": save a flat batch with `(group, "observation_history")` computed
    #   from the time dimension (useful for recurrent world models, e.g. mbpo_recurrent).
    # - "both": save both flat and trajectory shards.
    dataset_format: str = "flat"
    # When dataset_format is "flat_with_history" or "both", these control data collection:
    # max_history_length: maximum window length for history (L = max_history_length + 1)
    # max_future_length: maximum window length for future prediction (minimum 1)
    max_history_length: int = 0
    max_future_length: int = 1
    save_every_n_batches: int = 1
    # Avoid creating W&B / CSV logs for the auxiliary policy experiments.
    disable_loggers: bool = True

    # Dataset utilities
    # Merge one or more existing datasets created by this script into a single dataset directory.
    # Example:
    #   python benchmarl/collect.py task=... algorithm=... ++collect.merge_from=dataset/a,dataset/b ++collect.merge_output_dir=dataset/merged
    merge_from: Optional[str] = None
    merge_output_dir: Optional[str] = None
    # Optional: filter which shard formats to merge (comma-separated).
    # Formats are those written in index.jsonl under the "format" field, e.g.:
    # - "flat" (default flattened shards)
    # - "flat_hist" (flat shards with observation/action history + next_recurrent windows)
    # - "traj" (trajectory shards)
    # When unset/empty, all shard formats found in the sources are merged.
    merge_formats: Optional[str] = None
    # How to materialize shard files in the merged dataset:
    # - "copy": copy shard files into the merged directory (most portable)
    # - "hardlink": create hardlinks (fast, but requires same filesystem)
    merge_mode: str = "copy"


def _get_collect_cfg(cfg: DictConfig) -> CollectConfig:
    # Users can pass `++collect.*` overrides even though the base config has no collect section.
    d = {}
    if "collect" in cfg:
        try:
            d = OmegaConf.to_container(cfg.collect, resolve=True)  # type: ignore[attr-defined]
        except Exception:
            d = {}
    # Filter to dataclass fields.
    valid = set(CollectConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    d = {k: v for k, v in (d or {}).items() if k in valid}
    return CollectConfig(**d)


def _parse_csv(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_weights(s: Optional[str], n: int) -> Optional[List[float]]:
    if s is None:
        return None
    parts = _parse_csv(s)
    if not parts:
        return None
    if len(parts) != n:
        raise ValueError(f"collect.policy_weights must have {n} entries, got {len(parts)}")
    w = [float(x) for x in parts]
    if any(v < 0 for v in w) or sum(w) <= 0:
        raise ValueError("collect.policy_weights must be non-negative and sum to > 0")
    return w


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_index_records(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load index records from index.jsonl; fall back to scanning batch_*.pt."""
    idx = dataset_dir / "index.jsonl"
    recs: List[Dict[str, Any]] = []
    if idx.exists():
        for line in idx.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
        return recs

    # Fallback: scan shard files
    for p in sorted(dataset_dir.glob("batch_*.pt")):
        recs.append({"path": p.name, "numel": None})
    return recs


def merge_datasets(
    *,
    input_dirs: List[Path],
    output_dir: Path,
    mode: str = "copy",
    formats: Optional[List[str]] = None,
) -> None:
    """Merge datasets created by this script into one directory.

    Produces:
    - output_dir/meta.json: a merged meta containing a list of source metas
    - output_dir/index.jsonl: concatenated records with rewritten shard paths
    - shard files copied/hardlinked into output_dir
    """
    if not input_dirs:
        raise ValueError("merge_datasets: input_dirs is empty")
    mode = str(mode or "copy").strip().lower()
    if mode not in ("copy", "hardlink"):
        raise ValueError("collect.merge_mode must be 'copy' or 'hardlink'")

    output_dir.mkdir(parents=True, exist_ok=True)

    source_metas: List[Dict[str, Any]] = []
    merged_from: List[str] = []
    merged_records: List[Dict[str, Any]] = []
    batch_idx = 0
    allowed_formats = {str(x).strip() for x in (formats or []) if str(x).strip()}
    formats_present: set[str] = set()

    for src_i, src in enumerate(input_dirs):
        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(f"merge_datasets: dataset dir not found: {str(src)}")

        merged_from.append(str(src))
        meta = _read_json(src / "meta.json")
        if meta is not None:
            source_metas.append(meta)

        recs = _iter_index_records(src)
        for rec in recs:
            rel = str(rec.get("path", "")).strip()
            if not rel:
                continue

            # Optional filtering by shard format (e.g., keep only "flat_hist" for recurrent datasets).
            rec_fmt = rec.get("format", None)
            if allowed_formats and str(rec_fmt).strip() not in allowed_formats:
                continue
            if rec_fmt is not None:
                formats_present.add(str(rec_fmt))

            shard_src = src / rel
            if not shard_src.exists():
                # Skip missing shards rather than failing the whole merge.
                continue

            # Ensure unique shard names even if multiple datasets used the same batch_* naming.
            new_name = f"src{src_i}_{Path(rel).name}"
            shard_dst = output_dir / new_name

            if not shard_dst.exists():
                if mode == "hardlink":
                    try:
                        os.link(shard_src, shard_dst)
                    except Exception:
                        # Fall back to copy if hardlink fails (different filesystem / permissions).
                        shutil.copy2(shard_src, shard_dst)
                else:
                    shutil.copy2(shard_src, shard_dst)

            # Preserve all per-record metadata (e.g. "format") while rewriting path/batch_idx.
            new_rec = dict(rec)
            new_rec["batch_idx"] = batch_idx
            new_rec["path"] = shard_dst.name
            new_rec["source_dir"] = str(src)
            merged_records.append(new_rec)
            batch_idx += 1

    # Write merged index.jsonl
    index_path = output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for rec in merged_records:
            f.write(json.dumps(rec) + "\n")

    # Write merged meta.json
    merged_meta: Dict[str, Any] = {
        "merged_from": merged_from,
        "num_sources": len(input_dirs),
        "num_shards": len(merged_records),
        "source_metas": source_metas,
        "formats_present": sorted(formats_present),
    }
    (output_dir / "meta.json").write_text(json.dumps(merged_meta, indent=2), encoding="utf-8")


def _rand_discrete_from_mask(mask: torch.Tensor) -> torch.Tensor:
    # mask: [..., n_actions] bool/binary
    probs = mask.to(torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    a = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).reshape(
        *mask.shape[:-1], 1
    )
    return a


def _random_actions_for_group(
    *,
    td: TensorDictBase,
    group: str,
    action_spec,
) -> torch.Tensor:
    """Sample random actions consistent with the environment action spec.

    - Respects `(group, "action_mask")` when present (discrete).
    - For continuous, samples uniform in bounds if available, else standard normal.
    """
    # Infer batch shape from observation (common key always present).
    obs = td.get((group, "observation"))
    batch_shape = obs.shape[:-1]  # includes agent dim if present (e.g. [B, n_agents])

    mask = None
    if (group, "action_mask") in td.keys(True, True):
        mask = td.get((group, "action_mask"))

    spec = action_spec[group, "action"]

    # Discrete specs
    if isinstance(spec, Categorical) or isinstance(spec, OneHot) or hasattr(spec, "n"):
        # Determine number of actions.
        n = int(getattr(spec, "n", 0) or 0)
        if n <= 0 and hasattr(spec, "space") and hasattr(spec.space, "n"):
            n = int(spec.space.n)
        if n <= 0:
            # Fallback: infer from mask last dim.
            if mask is not None:
                n = int(mask.shape[-1])
        if n <= 0:
            raise ValueError(f"Could not infer discrete action count for group '{group}'")

        if mask is not None:
            a = _rand_discrete_from_mask(mask.to(obs.device))
        else:
            a = torch.randint(0, n, (*batch_shape, 1), device=obs.device, dtype=torch.long)

        # If the spec expects one-hot, convert.
        if isinstance(spec, OneHot):
            a_oh = torch.nn.functional.one_hot(a.squeeze(-1), num_classes=n).to(torch.float32)
            return a_oh
        return a

    # Continuous-ish
    # Try bounds if available, else normal.
    low = getattr(spec, "minimum", None)
    high = getattr(spec, "maximum", None)
    if low is not None and high is not None:
        low_t = torch.as_tensor(low, device=obs.device, dtype=torch.float32)
        high_t = torch.as_tensor(high, device=obs.device, dtype=torch.float32)
        # Broadcast bounds to batch.
        while low_t.dim() < len(batch_shape) + 1:
            low_t = low_t.unsqueeze(0)
            high_t = high_t.unsqueeze(0)
        u = torch.rand((*batch_shape, low_t.shape[-1]), device=obs.device)
        return (low_t + (high_t - low_t) * u).to(torch.float32)

    # Unknown bounds: standard normal with correct last dim.
    # We try to use spec.shape if present.
    last_dim = 1
    if hasattr(spec, "shape") and spec.shape is not None and len(spec.shape):
        last_dim = int(spec.shape[-1])
    return torch.randn((*batch_shape, last_dim), device=obs.device, dtype=torch.float32)


def _ensure_group_reward_done_like_next_obs(
    td: TensorDictBase, group: str
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (reward, done) tensors shaped like
    ("next", group, "observation")'s leading dims.

    Some envs/collectors store reward/done at 
    ("next","reward") / ("next","done") instead of group-level.
    This helper tries to return group-level-like tensors with
    shape [..., n_agents, 1] when possible.
    """
    keys = td.keys(True, True)
    try:
        next_obs = td.get(("next", group, "observation"))
    except Exception:
        next_obs = None

    rew = None
    done = None

    if ("next", group, "reward") in keys:
        try:
            rew = td.get(("next", group, "reward"))
        except Exception:
            rew = None
    elif ("next", "reward") in keys and next_obs is not None:
        try:
            r = td.get(("next", "reward"))
            # Expand to [..., n_agents] then add trailing singleton dim.
            lead = next_obs.shape[:-1]
            rew = r.expand(*lead).unsqueeze(-1)
        except Exception:
            rew = None

    if ("next", group, "done") in keys:
        try:
            done = td.get(("next", group, "done"))
        except Exception:
            done = None
    elif ("next", "done") in keys and next_obs is not None:
        try:
            d = td.get(("next", "done"))
            lead = next_obs.shape[:-1]
            done = d.expand(*lead).unsqueeze(-1)
        except Exception:
            done = None

    return rew, done


def _make_flat_batch_with_observation_history(
    td: TensorDictBase, *, groups: List[str], max_history_length: int = 0, max_future_length: int = 1
) -> TensorDictBase:
    """Convert a trajectory batch into a flat batch with observation/action history and future windows.

    Expected input (common TorchRL collector output): top-level batch_size = [T, B]
    with group tensors shaped like:
      (group,"observation"): [T, B, n_agents, obs_dim] (or extra obs dims)
      (group,"action"):      [T, B, n_agents, ...]
      ("next",group,"observation"): [T, B, n_agents, obs_dim]
      ("next",group,"reward"), ("next",group,"done") (or top-level next reward/done)

    Output is a flat TensorDict with batch_size [N] containing:
      - (group, "observation"): [N, n_agents, obs_dim]
      - (group, "observation_history"): [N, L, n_agents, obs_dim] where L = max_history_length + 1
      - (group, "action_history"): [N, L, n_agents, action_dim]
      - (group, "action"): [N, n_agents, action_dim]
      - ("next", group, ...): single-step (backwards compatible)
      - ("next_recurrent", group, ...): multi-step future windows [N, L', n_agents, ...] where L' = max_future_length
    """

    # If the input has no time dimension, we can't build windows
    if td.batch_size is None or len(td.batch_size) < 2:
        raise ValueError("no_time_dim")

    max_history_length = max(0, int(max_history_length))
    max_future_length = max(1, int(max_future_length))
    L = max_history_length + 1  # History window length

    # Build a per-group flat TD and then merge into a common top-level TD.
    out: Optional[TensorDict] = None
    for g in groups:
        keys = td.keys(True, True)
        if (
            (g, "observation") not in keys
            or (g, "action") not in keys
            or ("next", g, "observation") not in keys
        ):
            continue

        obs = td.get((g, "observation"))
        action = td.get((g, "action"))
        next_obs = td.get(("next", g, "observation"))
        reward, done = _ensure_group_reward_done_like_next_obs(td, g)
        if reward is None or done is None:
            continue

        # Require at least [B, T, n_agents, ...]
        if obs.dim() < 4:
            continue

        # The data is structured as [B, T, n_agents, ...], where B is the number of
        # parallel environments, and T the maximum number of timesteps.
        # The done tensor is used to segment trajectories.

        d_any = done.squeeze(-1).any(dim=-1)  # [B, T]
        try:
            # Segment trajectories for each batch (env roll)
            all_windows = []
            for b in range(d_any.shape[0]):
                done1d = d_any[b]  # [T]
                ends = (done1d == True).nonzero(as_tuple=False).flatten().tolist()  # episode ends (inclusive)
                starts = [0] + [i + 1 for i in ends]
                ends = ends + [d_any.shape[1] - 1]  # cover tail

                # Process each trajectory segment
                for s, e in zip(starts, ends):
                    if s > e or s >= obs.shape[1]:
                        continue
                    
                    traj_len = e - s + 1
                    if traj_len < 1:
                        continue

                    # Extract trajectory segment
                    traj_obs = obs[b, s : e + 1]  # [traj_len, n_agents, obs_dim]
                    traj_action = action[b, s : e + 1]  # [traj_len, n_agents, action_dim]
                    traj_next_obs = next_obs[b, s : e + 1]  # [traj_len, n_agents, obs_dim]
                    traj_reward = reward[b, s : e + 1]  # [traj_len, n_agents, ...]
                    traj_done = done[b, s : e + 1]  # [traj_len, n_agents, ...]
                    
                    # Extract action_mask if present
                    traj_action_mask = None
                    if (g, "action_mask") in keys:
                        try:
                            am = td.get((g, "action_mask"))
                            if am.dim() >= 3:  # [B, T, n_agents, ...] or similar
                                traj_action_mask = am[b, s : e + 1]
                        except Exception:
                            pass

                    # Create sliding windows
                    # Start from position where we have enough history

                    for t in range(L - 1, traj_len):
                        
                        # Check if window crosses episode boundary (done flag in history window)
                        if max_history_length > 0:
                            # Check if any done=True in the history window (excluding current step)

                            history_done = traj_done[max(0, t - L + 1) : t]
                            if history_done.numel() > 0 and history_done.any():
                                continue  # Skip windows that cross episode boundaries

                        # History window: [t-L+1, t] (inclusive)
                        hist_start = max(0, t - L + 1)
                        obs_hist = traj_obs[hist_start : t + 1]  # [actual_L, n_agents, obs_dim]
                        action_hist = traj_action[hist_start : t + 1]  # [actual_L, n_agents, action_dim]

                        # Pad history if needed (shouldn't happen if L <= traj_len, but handle edge case)
                        if obs_hist.shape[0] < L:
                            # Repeat first observation/action to pad
                            pad_len = L - obs_hist.shape[0]
                            first_obs = obs_hist[0:1].expand(pad_len, -1, -1)
                            first_action = action_hist[0:1].expand(pad_len, -1, -1)
                            obs_hist = torch.cat([first_obs, obs_hist], dim=0)
                            action_hist = torch.cat([first_action, action_hist], dim=0)

                        # Current observation and action at position t
                        curr_obs = traj_obs[t]  # [n_agents, obs_dim]
                        curr_action = traj_action[t]  # [n_agents, action_dim]
                        curr_action_mask = traj_action_mask[t] if traj_action_mask is not None else None

                        # Future window: [t+1, t+1+L'] (if available)
                        future_start = t + 1
                        future_end = min(traj_len, future_start + max_future_length)
                        future_len = future_end - future_start

                        if future_len > 0:
                            next_obs_future = traj_next_obs[future_start : future_end]  # [future_len, n_agents, obs_dim]
                            reward_future = traj_reward[future_start : future_end]  # [future_len, n_agents, ...]
                            done_future = traj_done[future_start : future_end]  # [future_len, n_agents, ...]

                            # Pad future if needed
                            if future_len < max_future_length:
                                pad_len = max_future_length - future_len
                                last_obs = next_obs_future[-1:].expand(pad_len, -1, -1)
                                last_reward = reward_future[-1:].expand(pad_len, -1, -1)
                                last_done = done_future[-1:].expand(pad_len, -1, -1)
                                next_obs_future = torch.cat([next_obs_future, last_obs], dim=0)
                                reward_future = torch.cat([reward_future, last_reward], dim=0)
                                done_future = torch.cat([done_future, last_done], dim=0)
                        else:
                            # No future available, pad with last observation
                            last_obs = traj_obs[-1:].expand(max_future_length, -1, -1)
                            last_reward = traj_reward[-1:].expand(max_future_length, -1, -1)
                            last_done = traj_done[-1:].expand(max_future_length, -1, -1)
                            next_obs_future = last_obs
                            reward_future = last_reward
                            done_future = last_done

                        # Single-step next (for backwards compatibility)
                        if future_len > 0:
                            next_obs_single = traj_next_obs[future_start]  # [n_agents, obs_dim]
                            reward_single = traj_reward[future_start]  # [n_agents, ...]
                            done_single = traj_done[future_start]  # [n_agents, ...]
                        else:
                            # Use last observation if no future
                            next_obs_single = traj_obs[-1]
                            reward_single = traj_reward[-1]
                            done_single = traj_done[-1]

                        window_data = {
                            "obs_hist": obs_hist,  # [L, n_agents, obs_dim]
                            "action_hist": action_hist,  # [L, n_agents, action_dim]
                            "obs": curr_obs,  # [n_agents, obs_dim]
                            "action": curr_action,  # [n_agents, action_dim]
                            "next_obs_single": next_obs_single,  # [n_agents, obs_dim]
                            "reward_single": reward_single,  # [n_agents, ...]
                            "done_single": done_single,  # [n_agents, ...]
                            "next_obs_future": next_obs_future,  # [max_future_length, n_agents, obs_dim]
                            "reward_future": reward_future,  # [max_future_length, n_agents, ...]
                            "done_future": done_future,  # [max_future_length, n_agents, ...]
                        }

                        if curr_action_mask is not None:
                            window_data["action_mask"] = curr_action_mask
                        all_windows.append(window_data)

            if not all_windows:
                continue


            # Stack all windows
            N = len(all_windows)
            device = all_windows[0]["obs"].device
            n_agents = int(all_windows[0]["obs"].shape[-2])
            obs_dim = int(all_windows[0]["obs"].shape[-1])

            obs_hist_stack = torch.stack([w["obs_hist"] for w in all_windows], dim=0)  # [N, L, n_agents, obs_dim]
            action_hist_stack = torch.stack([w["action_hist"] for w in all_windows], dim=0)  # [N, L, n_agents, action_dim]
            obs_stack = torch.stack([w["obs"] for w in all_windows], dim=0)  # [N, n_agents, obs_dim]
            action_stack = torch.stack([w["action"] for w in all_windows], dim=0)  # [N, n_agents, action_dim]
            next_obs_single_stack = torch.stack([w["next_obs_single"] for w in all_windows], dim=0)  # [N, n_agents, obs_dim]
            reward_single_stack = torch.stack([w["reward_single"] for w in all_windows], dim=0)  # [N, n_agents, ...]
            done_single_stack = torch.stack([w["done_single"] for w in all_windows], dim=0)  # [N, n_agents, ...]
            next_obs_future_stack = torch.stack([w["next_obs_future"] for w in all_windows], dim=0)  # [N, max_future_length, n_agents, obs_dim]
            reward_future_stack = torch.stack([w["reward_future"] for w in all_windows], dim=0)  # [N, max_future_length, n_agents, ...]
            done_future_stack = torch.stack([w["done_future"] for w in all_windows], dim=0)  # [N, max_future_length, n_agents, ...]

            # Build group TensorDict
            group_payload: Dict[str, torch.Tensor] = {
                "observation": obs_stack.contiguous(),
                "observation_history": obs_hist_stack.contiguous(),
                "action_history": action_hist_stack.contiguous(),
                "action": action_stack.contiguous(),
            }

            # Handle action_mask if present
            if any("action_mask" in w for w in all_windows):
                try:
                    action_mask_list = [w.get("action_mask") for w in all_windows if "action_mask" in w]
                    if action_mask_list and len(action_mask_list) == N:
                        action_mask_stack = torch.stack(action_mask_list, dim=0)
                        group_payload["action_mask"] = action_mask_stack.contiguous()
                except Exception:
                    pass

            # Batch-size should match the leading dimension(s) of tensors. Here all tensors
            # are shaped with leading dim [N, ...] (agent is a tensor dim, not a TD batch dim).
            group_td = TensorDict(group_payload, batch_size=[N], device=device)

            # Build next TensorDict (backwards compatible single-step)
            next_group_td = TensorDict(
                {
                    "observation": next_obs_single_stack.contiguous(),
                    "reward": reward_single_stack.contiguous(),
                    "done": done_single_stack.contiguous(),
                },
                batch_size=[N],
                device=device,
            )

            # Build next_recurrent TensorDict (multi-step future)
            next_recurrent_group_td = TensorDict(
                {
                    "observation": next_obs_future_stack.contiguous(),
                    "reward": reward_future_stack.contiguous(),
                    "done": done_future_stack.contiguous(),
                },
                batch_size=[N],
                device=device,
            )

            if out is None:
                out = TensorDict({}, batch_size=[N], device=device)
                out.set("next", TensorDict({}, batch_size=[N], device=device))
                out.set("next_recurrent", TensorDict({}, batch_size=[N], device=device))

            out.set(g, group_td)
            out.get("next").set(g, next_group_td)
            out.get("next_recurrent").set(g, next_recurrent_group_td)

        except Exception as e:
            print(f"Error in _make_flat_batch_with_observation_history for group {g}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return out


def _make_behavior_policy(
    *,
    experiment,
    collect_cfg: CollectConfig,
):
    policy_name = str(collect_cfg.policy or "reference").strip().lower()
    if policy_name not in ("reference", "random", "epsilon_greedy", "gaussian_noise"):
        raise ValueError(
            f"Unknown collect.policy={collect_cfg.policy}. "
            "Valid: reference|random|epsilon_greedy|gaussian_noise"
        )

    reference_policy = experiment.policy  # TensorDictSequential

    def random_policy(td: TensorDictBase) -> TensorDictBase:
        for group in experiment.group_map.keys():
            a = _random_actions_for_group(td=td, group=group, action_spec=experiment.action_spec)
            td.set((group, "action"), a)
        return td

    def epsilon_greedy_policy(td: TensorDictBase) -> TensorDictBase:
        eps = float(collect_cfg.epsilon)
        eps = min(max(eps, 0.0), 1.0)
        td_ref = reference_policy(td.clone())
        td_rand = random_policy(td.clone())
        for group in experiment.group_map.keys():
            a_ref = td_ref.get((group, "action"))
            a_rand = td_rand.get((group, "action"))
            # Bernoulli per env-agent element (no action-dim).
            shape = a_ref.shape[:-1] if a_ref.dim() else a_ref.shape
            take_rand = (torch.rand(shape, device=a_ref.device) < eps).unsqueeze(-1)
            a = torch.where(take_rand, a_rand, a_ref)
            td.set((group, "action"), a)
        return td

    def gaussian_noise_policy(td: TensorDictBase) -> TensorDictBase:
        td_ref = reference_policy(td)
        std = float(collect_cfg.noise_std)
        for group in experiment.group_map.keys():
            a = td_ref.get((group, "action")).to(torch.float32)
            noise = torch.randn_like(a) * std
            a2 = a + noise
            # Clamp if bounds exist.
            spec = experiment.action_spec[group, "action"]
            low = getattr(spec, "minimum", None)
            high = getattr(spec, "maximum", None)
            if low is not None and high is not None:
                low_t = torch.as_tensor(low, device=a2.device, dtype=a2.dtype)
                high_t = torch.as_tensor(high, device=a2.device, dtype=a2.dtype)
                while low_t.dim() < a2.dim():
                    low_t = low_t.unsqueeze(0)
                    high_t = high_t.unsqueeze(0)
                a2 = torch.max(torch.min(a2, high_t), low_t)
            td.set((group, "action"), a2)
        return td

    if policy_name == "reference":
        return reference_policy
    if policy_name == "random":
        return random_policy
    if policy_name == "epsilon_greedy":
        return epsilon_greedy_policy
    return gaussian_noise_policy


def _instantiate_experiment_for_algorithm(
    *,
    cfg: DictConfig,
    algorithm_name: str,
    task_name: str,
    disable_loggers: bool,
) -> Experiment:
    """Create an Experiment for a specific algorithm, using defaults + cfg.algorithm overrides."""
    algorithm_name = str(algorithm_name).strip()
    if algorithm_name not in algorithm_config_registry:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. Available: {sorted(list(algorithm_config_registry.keys()))}"
        )

    # Task / models / experiment config from the current Hydra cfg.
    task_cfg = load_task_config_from_hydra(cfg.task, task_name)
    exp_cfg = load_experiment_config_from_hydra(cfg.experiment)
    if disable_loggers:
        exp_cfg.loggers = []
        exp_cfg.create_json = False
        # Avoid rendering/eval overhead for pure collection unless user explicitly wants it.
        exp_cfg.evaluation = False
        exp_cfg.render = False

    model_cfg = load_model_config_from_hydra(cfg.model)
    critic_model_cfg = load_model_config_from_hydra(cfg.critic_model)

    # Algorithm config: start from yaml defaults and overlay cfg.algorithm keys that match dataclass fields.
    config_cls = algorithm_config_registry[algorithm_name]
    base_cfg = config_cls.get_from_yaml()
    base_dict = dict(base_cfg.__dict__)
    cfg_algo = OmegaConf.to_container(cfg.algorithm, resolve=True)  # type: ignore[arg-type]
    valid_fields = {f.name for f in fields(config_cls)}
    for k, v in (cfg_algo or {}).items():
        if k in valid_fields:
            base_dict[k] = v
    algo_cfg = config_cls(**base_dict)

    return Experiment(
        task=task_cfg,
        algorithm_config=algo_cfg,
        model_config=model_cfg,
        critic_model_config=critic_model_cfg,
        seed=int(cfg.seed),
        config=exp_cfg,
        callbacks=[],
    )


def _load_policy_sources(
    *,
    cfg: DictConfig,
    task_name: str,
    collect_cfg: CollectConfig,
    algorithm_name_default: str,
) -> Tuple[List[Any], List[str]]:
    """Return (policies, labels). Policies are callables(TensorDict)->TensorDict."""
    # Multi-policy mode
    tokens = _parse_csv(collect_cfg.policies)
    if not tokens:
        # Single-policy mode
        if collect_cfg.reference_restore_file:
            exp = reload_experiment_from_file(str(collect_cfg.reference_restore_file))
            return [exp.policy], [f"ckpt:{collect_cfg.reference_restore_file}"]
        # Fall back to "collect.policy" with the current experiment's policy created later.
        return [], []

    policies = []
    labels = []
    for tok in tokens:
        t = tok.strip()
        if t.lower() == "random":
            policies.append("RANDOM_SENTINEL")
            labels.append("random")
            continue
        if t.lower().startswith("ckpt:"):
            ckpt = t.split("ckpt:", 1)[1]
            exp = reload_experiment_from_file(ckpt)
            policies.append(exp.policy)
            labels.append(f"ckpt:{ckpt}")
            continue
        # Otherwise interpret as algorithm name (build an experiment from defaults + overrides).
        algo_name = t
        exp = _instantiate_experiment_for_algorithm(
            cfg=cfg,
            algorithm_name=algo_name,
            task_name=task_name,
            disable_loggers=collect_cfg.disable_loggers,
        )
        policies.append(exp.policy)
        labels.append(f"algo:{algo_name}")

    return policies, labels


def _build_env_func_for_collection(experiment, *, n_envs: int):
    """Build an env factory for collection, independent of algorithm on/off-policy settings.

    We reuse the task transforms and the algorithm env wrapper (process_env_fun) so the policy sees the same interface.
    """
    n_envs = max(1, int(n_envs))
    device = experiment.config.sampling_device

    # Build a "base env" instance just to derive transforms.
    base_env = experiment.task.get_env_fun(
        num_envs=n_envs,
        continuous_actions=experiment.continuous_actions,
        seed=experiment.seed,
        device=device,
    )()

    transforms_env_list = experiment.task.get_env_transforms(base_env)
    transforms_training = Compose(
        *(transforms_env_list + [experiment.task.get_reward_sum_transform(base_env)])
    )

    env_fun = experiment.task.get_env_fun(
        num_envs=n_envs,
        continuous_actions=experiment.continuous_actions,
        seed=experiment.seed,
        device=device,
    )

    # Add RNN transforms if needed (mirrors Experiment._setup_task).
    if experiment.model_config.is_rnn:
        env_fun = _add_rnn_transforms(env_fun, experiment.group_map, experiment.model_config)

    # If env is not vectorized, simulate vectorization.
    if base_env.batch_size == ():
        env_class = SerialEnv if not experiment.config.parallel_collection else ParallelEnv

        def _env_func():
            return TransformedEnv(env_class(n_envs, env_fun), transforms_training.clone())

        env_func = _env_func
    else:
        env_func = lambda: TransformedEnv(env_fun(), transforms_training.clone())

    # Let algorithm wrap env if needed.
    return experiment.algorithm.process_env_fun(env_func)


def _default_output_dir() -> str:
    try:
        return str(Path(HydraConfig.get().runtime.output_dir) / "dataset")
    except Exception:
        return str(Path.cwd() / "dataset")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_collect(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\n[collect] Algorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    collect_cfg = _get_collect_cfg(cfg)

    # Utility mode: merge existing datasets and exit.
    merge_tokens = _parse_csv(collect_cfg.merge_from)
    if merge_tokens:
        out_dir = Path(collect_cfg.merge_output_dir or _default_output_dir())
        print("\n[collect] Merge mode enabled")
        print(f"[collect] Inputs: {merge_tokens}")
        print(f"[collect] Output: {str(out_dir)}")
        merge_datasets(
            input_dirs=[Path(p) for p in merge_tokens],
            output_dir=out_dir,
            mode=str(collect_cfg.merge_mode),
            formats=_parse_csv(getattr(collect_cfg, "merge_formats", None)),
        )
        print(f"\n[collect] Merged dataset written to: {str(out_dir)}")
        return

    # Base experiment (used for env specs + (optional) reference policy when collect.policy=reference).
    # We disable loggers by default for collection to avoid creating extra W&B/CSV artifacts.
    base_experiment = _instantiate_experiment_for_algorithm(
        cfg=cfg,
        algorithm_name=algorithm_name,
        task_name=task_name,
        disable_loggers=collect_cfg.disable_loggers,
    )

    # Override collection sizes if requested; default to experiment off-policy knobs when available.
    if collect_cfg.n_envs is None:
        # Prefer off-policy setting for dataset collection (even if algorithm is on-policy).
        try:
            collect_cfg.n_envs = int(base_experiment.config.off_policy_n_envs_per_worker)
        except Exception:
            collect_cfg.n_envs = int(
                base_experiment.config.n_envs_per_worker(base_experiment.on_policy)
            )
    if collect_cfg.frames_per_batch is None:
        try:
            collect_cfg.frames_per_batch = int(
                base_experiment.config.off_policy_collected_frames_per_batch
            )
        except Exception:
            collect_cfg.frames_per_batch = int(
                base_experiment.config.collected_frames_per_batch(base_experiment.on_policy)
            )

    out_dir = Path(collect_cfg.output_dir or _default_output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load behavior policies (optionally multiple).
    loaded_policies, policy_labels = _load_policy_sources(
        cfg=cfg,
        task_name=task_name,
        collect_cfg=collect_cfg,
        algorithm_name_default=algorithm_name,
    )

    def random_policy(td: TensorDictBase) -> TensorDictBase:
        for group in base_experiment.group_map.keys():
            a = _random_actions_for_group(
                td=td, group=group, action_spec=base_experiment.action_spec
            )
            td.set((group, "action"), a)
        return td

    if loaded_policies:
        # Turn sentinels into callables.
        policy_fns: List[Any] = []
        for p in loaded_policies:
            if p == "RANDOM_SENTINEL":
                policy_fns.append(random_policy)
            else:
                # Ensure policy modules are on the sampling device.
                try:
                    p = p.to(base_experiment.config.sampling_device)
                except Exception:
                    pass
                policy_fns.append(p)

        mix = str(collect_cfg.mix or "round_robin").strip().lower()
        if mix not in ("round_robin", "weighted_random"):
            raise ValueError("collect.mix must be round_robin or weighted_random")
        weights = _parse_weights(collect_cfg.policy_weights, len(policy_fns))
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(cfg.seed) + 12_345_67)
        rr_idx = {"i": 0}

        def behavior_policy(td: TensorDictBase) -> TensorDictBase:
            if mix == "round_robin":
                i = rr_idx["i"] % len(policy_fns)
                rr_idx["i"] += 1
            else:
                w = torch.tensor(weights, dtype=torch.float32)
                i = int(torch.multinomial(w, num_samples=1, generator=rng).item())
            return policy_fns[i](td)

    else:
        # Single-policy mode: use collect.policy on top of the base experiment.
        behavior_policy = _make_behavior_policy(experiment=base_experiment, collect_cfg=collect_cfg)

    env_func = _build_env_func_for_collection(base_experiment, n_envs=int(collect_cfg.n_envs))

    exploration = ExplorationType.MEAN if collect_cfg.deterministic else ExplorationType.RANDOM

    collector = SyncDataCollector(
        env_func,
        behavior_policy,
        device=base_experiment.config.sampling_device,
        storing_device=base_experiment.config.sampling_device,
        frames_per_batch=int(collect_cfg.frames_per_batch),
        total_frames=int(collect_cfg.total_frames),
        init_random_frames=int(collect_cfg.init_random_frames),
    )

    meta = {
        "task": task_name,
        "algorithm": algorithm_name,
        "seed": int(cfg.seed),
        "collect": OmegaConf.to_container(OmegaConf.structured(collect_cfg), resolve=True),
        "behavior_sources": policy_labels if policy_labels else [collect_cfg.policy],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    index_path = out_dir / "index.jsonl"
    n_batches = 0
    total_frames = 0

    fmt = str(getattr(collect_cfg, "dataset_format", "flat") or "flat").strip().lower()
    if fmt not in ("flat", "trajectory", "flat_with_history", "both"):
        raise ValueError(
            "collect.dataset_format must be one of: flat|trajectory|flat_with_history|both"
        )

    with set_exploration_type(exploration):
        for batch in collector:
            if batch is None:
                continue
            batch_raw = batch.detach()

            groups = list(base_experiment.group_map.keys())

            # Build one or more payloads depending on requested format.
            payloads: List[Tuple[str, TensorDictBase]] = []
            if fmt in ("trajectory", "both"):
                payloads.append(("traj", batch_raw))
            if fmt in ("flat_with_history", "both"):
                pl = _make_flat_batch_with_observation_history(
                    batch_raw,
                    groups=groups,
                    max_history_length=int(getattr(collect_cfg, "max_history_length", 0)),
                    max_future_length=int(getattr(collect_cfg, "max_future_length", 1)),
                )
                if pl is not None:
                    payloads.append(
                        ("flat_hist", pl)
                    )
            if fmt == "flat":
                flat = batch_raw
                if bool(getattr(collect_cfg, "flatten_batch", True)):
                    try:
                        flat = batch_raw.reshape(-1)
                    except Exception:
                        pass
                payloads.append(("flat", flat))

            # Save shard(s)
            if n_batches % int(collect_cfg.save_every_n_batches) == 0:
                for suffix, td_to_save in payloads:
                    # Keep the original shard naming for the default "flat" mode for backward compatibility.
                    if fmt == "flat":
                        shard_path = out_dir / f"batch_{n_batches:06d}.pt"
                    else:
                        shard_path = out_dir / f"batch_{n_batches:06d}_{suffix}.pt"

                    # Save on CPU to make dataset portable.
                    try:
                        to_save = td_to_save.to("cpu")
                    except Exception:
                        to_save = td_to_save
                    torch.save(to_save, shard_path)

                    rec = {
                        "batch_idx": n_batches,
                        "path": shard_path.name,
                        "numel": int(td_to_save.numel()),
                        "format": suffix,
                    }
                    with index_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")

            # Best-effort frame count
            try:
                bs0 = (
                    int(batch_raw.batch_size[0])
                    if hasattr(batch_raw, "batch_size") and len(batch_raw.batch_size)
                    else int(batch_raw.numel())
                )
            except Exception:
                bs0 = int(batch_raw.numel())
            total_frames += bs0
            n_batches += 1
            if total_frames >= int(collect_cfg.total_frames):
                break

    try:
        collector.shutdown()
    except Exception:
        pass

    print(f"\n[collect] Wrote dataset to: {str(out_dir)}")
    print(f"[collect] Batches: {n_batches} | approx frames: {total_frames}")


if __name__ == "__main__":
    hydra_collect()


