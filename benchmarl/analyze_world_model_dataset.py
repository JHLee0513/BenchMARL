#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""
Analyze statistics of offline datasets used for world-model training.

This script is designed to work with datasets produced by `benchmarl/collect.py`:
- a directory containing `meta.json`, `index.jsonl` (optional), and `batch_*.pt` shards
- each shard is a TensorDict containing transitions with keys like:
  (group, "observation"), (group, "action"), ("next", group, "observation"),
  ("next", group, "reward"), ("next", group, "done")

It computes:
- per-group shapes, counts, dtypes
- state/action/reward distributions (min/max/mean/std, quantiles)
- approximate "coverage" via 1D bin occupancy per feature dimension
- optional plots (reward histogram, coverage plots, PCA 2D occupancy)

Example:

  python benchmarl/analyze_world_model_dataset.py \
    --dataset-dir /path/to/dataset \
    --out-dir /path/to/dataset_analysis \
    --bins 50 --max-shards 200 --max-samples 300000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:
    plt = None  # type: ignore
    _HAS_MPL = False


NestedKey = Tuple[Any, ...]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_shards(dataset_dir: Path, max_shards: Optional[int]) -> List[Path]:
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

    Mirrors `benchmarl/train_world_model.py` behavior for maximum compatibility.
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


def _select_transitions(td: Any, n: int, *, generator: torch.Generator) -> Any:
    """Randomly sample up to n transitions from a (flattened) TensorDict-like object."""
    try:
        total = int(td.batch_size[0]) if hasattr(td, "batch_size") and len(td.batch_size) else int(td.numel())
    except Exception:
        total = int(td.numel())
    n = int(n)
    if n <= 0 or n >= total:
        return td
    idx = torch.randint(0, total, (n,), device="cpu", generator=generator)
    return td[idx]


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
    # Keep last dim as feature dim; flatten everything else.
    return x.reshape(-1, x.shape[-1])


def _safe_float(x: torch.Tensor) -> float:
    try:
        return float(x.detach().cpu().item())
    except Exception:
        return float("nan")


@dataclass
class RunningMoments:
    n: int = 0
    mean: Optional[torch.Tensor] = None
    m2: Optional[torch.Tensor] = None
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None
    nan_count: int = 0
    inf_count: int = 0

    def update(self, x: torch.Tensor) -> None:
        """Update running mean/variance/min/max for x shaped [N, D]."""
        x = _as_2d_features(x)
        x = x.detach()
        if x.numel() == 0:
            return

        # Track non-finite counts.
        finite = torch.isfinite(x)
        self.nan_count += int(torch.isnan(x).sum().item())
        self.inf_count += int((~finite & ~torch.isnan(x)).sum().item())

        # For moment updates, drop non-finite values per-dim (replace with 0 and mask via counts).
        # We keep it simple: if there are non-finites, we remove entire rows that contain any.
        row_finite = finite.all(dim=1)
        x = x[row_finite]
        if x.numel() == 0:
            return

        n_b = int(x.shape[0])
        batch_mean = x.mean(dim=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)
        batch_min = x.min(dim=0).values
        batch_max = x.max(dim=0).values

        if self.n == 0 or self.mean is None or self.m2 is None:
            self.n = n_b
            self.mean = batch_mean
            self.m2 = batch_m2
            self.min = batch_min
            self.max = batch_max
            return

        assert self.mean is not None and self.m2 is not None and self.min is not None and self.max is not None
        n_a = self.n
        n = n_a + n_b
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n_b / n)
        self.m2 = self.m2 + batch_m2 + (delta**2) * (n_a * n_b / n)
        self.n = n
        self.min = torch.minimum(self.min, batch_min)
        self.max = torch.maximum(self.max, batch_max)

    def finalize(self) -> Dict[str, Any]:
        if self.n <= 0 or self.mean is None or self.m2 is None:
            return {
                "n": int(self.n),
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "nan_count": int(self.nan_count),
                "inf_count": int(self.inf_count),
            }
        var = self.m2 / max(1, self.n - 1)
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "n": int(self.n),
            "mean": self.mean.detach().cpu().tolist(),
            "std": std.detach().cpu().tolist(),
            "min": (self.min.detach().cpu().tolist() if self.min is not None else None),
            "max": (self.max.detach().cpu().tolist() if self.max is not None else None),
            "nan_count": int(self.nan_count),
            "inf_count": int(self.inf_count),
        }


def _quantiles(x: torch.Tensor, qs: Sequence[float]) -> List[float]:
    x = x.detach().flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return [float("nan") for _ in qs]
    q = torch.tensor(list(qs), dtype=torch.float32, device=x.device)
    out = torch.quantile(x, q, interpolation="linear")
    return [float(v.item()) for v in out.detach().cpu()]


def _range_per_dim_from_sample(x: torch.Tensor, q_lo: float, q_hi: float, *, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate a robust [lo, hi] range per feature dim from a sample x shaped [N, D].
    """
    x = _as_2d_features(x)
    x = x.detach()
    x = x[torch.isfinite(x).all(dim=1)]
    if x.numel() == 0:
        d = int(x.shape[-1]) if x.ndim else 1
        lo = torch.full((d,), float("nan"))
        hi = torch.full((d,), float("nan"))
        return lo, hi
    q = torch.tensor([float(q_lo), float(q_hi)], dtype=torch.float32, device=x.device)
    qq = torch.quantile(x, q, dim=0, interpolation="linear")  # [2, D]
    lo = qq[0]
    hi = qq[1]
    # Avoid degenerate ranges.
    span = torch.clamp(hi - lo, min=0.0)
    hi = torch.where(span < eps, lo + eps, hi)
    return lo, hi


def _hist_coverage_1d(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    bins: int,
    max_dims: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """
    Compute per-dim 1D histogram coverage.

    Returns:
    - coverage_per_dim: fraction of non-empty bins per dim
    - oob_rate_per_dim: out-of-bounds rate per dim (outside [lo, hi])
    """
    x = _as_2d_features(x)
    x = x.detach()
    x = x[torch.isfinite(x).all(dim=1)]
    if x.numel() == 0:
        return [], []

    d = int(x.shape[1])
    if max_dims is not None:
        d = min(d, int(max_dims))
        x = x[:, :d]
        lo = lo[:d]
        hi = hi[:d]

    cov: List[float] = []
    oob: List[float] = []
    bins = max(1, int(bins))
    for j in range(d):
        xj = x[:, j]
        loj = float(lo[j].item())
        hij = float(hi[j].item())
        if not math.isfinite(loj) or not math.isfinite(hij) or hij <= loj:
            cov.append(float("nan"))
            oob.append(float("nan"))
            continue
        oob_mask = (xj < loj) | (xj > hij)
        oob.append(float(oob_mask.float().mean().item()))
        x_in = xj[~oob_mask]
        if x_in.numel() == 0:
            cov.append(0.0)
            continue
        h = torch.histc(x_in, bins=bins, min=loj, max=hij)
        cov.append(float((h > 0).float().mean().item()))
    return cov, oob


def _maybe_plot_reward_hist(reward: torch.Tensor, out_path: Path, *, title: str) -> None:
    if not _HAS_MPL:
        return
    r = reward.detach().flatten().cpu()
    r = r[torch.isfinite(r)]
    if r.numel() == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(r.numpy(), bins=80)
    plt.title(title)
    plt.xlabel("reward")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _maybe_plot_coverage(cov: Sequence[float], out_path: Path, *, title: str) -> None:
    if not _HAS_MPL:
        return
    if not cov:
        return
    y = torch.tensor([v for v in cov if math.isfinite(float(v))], dtype=torch.float32)
    if y.numel() == 0:
        return
    plt.figure(figsize=(8, 3.2))
    plt.plot(y.numpy(), linewidth=1.0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("feature dim")
    plt.ylabel("bin occupancy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _maybe_plot_pca_occupancy(x: torch.Tensor, out_path: Path, *, title: str) -> None:
    if not _HAS_MPL:
        return
    x = _as_2d_features(x).detach()
    x = x[torch.isfinite(x).all(dim=1)]
    if x.numel() == 0:
        return
    # Subsample for PCA for speed.
    n = min(int(x.shape[0]), 5000)
    if int(x.shape[0]) > n:
        idx = torch.randperm(int(x.shape[0]))[:n]
        x = x[idx]
    x = x - x.mean(dim=0, keepdim=True)
    try:
        # pca_lowrank returns V with shape [D, q]
        _, _, v = torch.pca_lowrank(x, q=2, center=False)
        z = x @ v[:, :2]
    except Exception:
        return
    z = z.detach().cpu()
    plt.figure(figsize=(5.5, 5))
    plt.hist2d(z[:, 0].numpy(), z[:, 1].numpy(), bins=60)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze(
    *,
    dataset_dir: Path,
    out_dir: Path,
    groups: Optional[List[str]],
    bins: int,
    q_lo: float,
    q_hi: float,
    max_shards: Optional[int],
    range_samples: int,
    max_samples: int,
    sample_per_shard: int,
    coverage_max_dims: int,
    seed: int,
) -> Dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _read_json(dataset_dir / "meta.json")
    shards = _list_shards(dataset_dir, max_shards=max_shards)
    if not shards:
        raise ValueError(f"No shard files found in: {str(dataset_dir)}")

    # RNGs
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 202_601_12)

    # First pass: gather small samples to estimate robust ranges for coverage bins.
    first = torch.load(shards[0], map_location="cpu", weights_only=False)
    try:
        keys = list(first.keys(True, True))
    except Exception:
        keys = []
    discovered_groups = _discover_groups(keys)
    if groups is None or len(groups) == 0:
        groups = discovered_groups
    else:
        groups = [str(x) for x in groups]

    range_sample: Dict[str, Dict[str, torch.Tensor]] = {gname: {} for gname in groups}
    per_group_found = {gname: 0 for gname in groups}

    for shard_path in shards:
        td = torch.load(shard_path, map_location="cpu", weights_only=False)
        td = td.detach()
        try:
            td = td.reshape(-1)
        except Exception:
            continue

        for gname in groups:
            tdg = _ensure_group_reward_done(td, gname)
            # Observation
            try:
                obs = tdg.get((gname, "observation"))
                rew = tdg.get(("next", gname, "reward"))
            except Exception:
                continue

            per_group_found[gname] += 1
            # Sample transitions within shard for range estimation.
            tds = _select_transitions(tdg, min(range_samples, sample_per_shard), generator=g)
            try:
                obs_s = _as_2d_features(tds.get((gname, "observation"))).to(torch.float32)
                rew_s = _as_2d_features(tds.get(("next", gname, "reward"))).to(torch.float32)
            except Exception:
                continue

            # Accumulate a capped sample buffer for range estimation.
            if "observation" not in range_sample[gname]:
                range_sample[gname]["observation"] = obs_s
            else:
                cur = range_sample[gname]["observation"]
                if int(cur.shape[0]) < int(range_samples):
                    range_sample[gname]["observation"] = torch.cat(
                        [cur, obs_s], dim=0
                    )[: int(range_samples)]

            if "reward" not in range_sample[gname]:
                range_sample[gname]["reward"] = rew_s
            else:
                cur = range_sample[gname]["reward"]
                if int(cur.shape[0]) < int(range_samples):
                    range_sample[gname]["reward"] = torch.cat([cur, rew_s], dim=0)[
                        : int(range_samples)
                    ]

        # Stop early if we have enough range samples for all groups.
        if all(
            ("observation" in range_sample[gname] and int(range_sample[gname]["observation"].shape[0]) >= int(range_samples))
            for gname in groups
        ):
            break

    ranges: Dict[str, Dict[str, Any]] = {}
    for gname in groups:
        ranges[gname] = {}
        if "observation" in range_sample[gname]:
            lo, hi = _range_per_dim_from_sample(range_sample[gname]["observation"], q_lo, q_hi)
            ranges[gname]["observation"] = {"q_lo": float(q_lo), "q_hi": float(q_hi), "lo": lo, "hi": hi}
        if "reward" in range_sample[gname]:
            r = range_sample[gname]["reward"]
            # Reward is typically scalar; use global lo/hi.
            rlo, rhi = _range_per_dim_from_sample(r, q_lo, q_hi)
            ranges[gname]["reward"] = {"q_lo": float(q_lo), "q_hi": float(q_hi), "lo": rlo, "hi": rhi}

    # Second pass: streaming stats + coverage using fixed ranges.
    stats: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "meta": meta,
        "num_shards": int(len(shards)),
        "groups": {},
        "notes": {
            "coverage": "Coverage is approximate 1D bin occupancy computed over a sample of transitions; "
            "it is not a true measure of continuous state space coverage.",
        },
    }

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for gname in groups:
        rm_obs = RunningMoments()
        rm_act = RunningMoments()
        rm_next_obs = RunningMoments()
        rm_rew = RunningMoments()
        done_rate_num = 0.0
        done_rate_den = 0.0

        obs_for_cov: List[torch.Tensor] = []
        rew_for_hist: List[torch.Tensor] = []
        obs_for_pca: List[torch.Tensor] = []
        act_for_misc: List[torch.Tensor] = []
        done_for_misc: List[torch.Tensor] = []

        seen = 0
        shard_used = 0
        for shard_path in shards:
            if seen >= int(max_samples):
                break
            td = torch.load(shard_path, map_location="cpu", weights_only=False)
            td = td.detach()
            try:
                td = td.reshape(-1)
            except Exception:
                continue

            td = _ensure_group_reward_done(td, gname)
            try:
                obs = td.get((gname, "observation"))
                act = td.get((gname, "action"))
                next_obs = td.get(("next", gname, "observation"))
                rew = td.get(("next", gname, "reward"))
                done = td.get(("next", gname, "done"))
            except Exception:
                continue

            shard_used += 1
            # Sample within shard to respect max_samples budget.
            remaining = int(max_samples) - seen
            n_take = min(int(sample_per_shard), remaining)
            td = _select_transitions(td, n_take, generator=g)
            try:
                obs = _as_2d_features(td.get((gname, "observation"))).to(torch.float32)
                act = _as_2d_features(td.get((gname, "action"))).to(torch.float32)
                next_obs = _as_2d_features(td.get(("next", gname, "observation"))).to(torch.float32)
                rew = _as_2d_features(td.get(("next", gname, "reward"))).to(torch.float32)
                done = _as_2d_features(td.get(("next", gname, "done"))).to(torch.float32)
            except Exception:
                continue

            seen += int(obs.shape[0])
            rm_obs.update(obs)
            rm_act.update(act)
            rm_next_obs.update(next_obs)
            rm_rew.update(rew)

            # Done rate (treat non-finite as missing).
            dmask = torch.isfinite(done).all(dim=1)
            if int(dmask.sum().item()) > 0:
                done_rate_num += float(done[dmask].mean().item()) * float(dmask.sum().item())
                done_rate_den += float(dmask.sum().item())

            # Keep small buffers for plotting/coverage.
            if int(len(obs_for_cov)) == 0:
                obs_for_cov.append(obs)
            else:
                cur = torch.cat(obs_for_cov, dim=0)
                if int(cur.shape[0]) < int(max_samples):
                    obs_for_cov = [torch.cat([cur, obs], dim=0)[: int(max_samples)]]

            if int(len(rew_for_hist)) == 0:
                rew_for_hist.append(rew)
            else:
                cur = torch.cat(rew_for_hist, dim=0)
                if int(cur.shape[0]) < int(max_samples):
                    rew_for_hist = [torch.cat([cur, rew], dim=0)[: int(max_samples)]]

            if int(len(obs_for_pca)) == 0:
                obs_for_pca.append(obs)
            else:
                cur = torch.cat(obs_for_pca, dim=0)
                if int(cur.shape[0]) < 8000:
                    obs_for_pca = [torch.cat([cur, obs], dim=0)[:8000]]

            # Keep small buffers for extra diagnostics.
            if int(len(act_for_misc)) == 0:
                act_for_misc.append(act)
            else:
                cur = torch.cat(act_for_misc, dim=0)
                if int(cur.shape[0]) < int(max_samples):
                    act_for_misc = [torch.cat([cur, act], dim=0)[: int(max_samples)]]
            if int(len(done_for_misc)) == 0:
                done_for_misc.append(done)
            else:
                cur = torch.cat(done_for_misc, dim=0)
                if int(cur.shape[0]) < int(max_samples):
                    done_for_misc = [torch.cat([cur, done], dim=0)[: int(max_samples)]]

        group_stats: Dict[str, Any] = {
            "shards_with_group_data": int(shard_used),
            "sampled_transitions": int(seen),
            "observation": rm_obs.finalize(),
            "action": rm_act.finalize(),
            "next_observation": rm_next_obs.finalize(),
            "reward": rm_rew.finalize(),
            "done_rate": (done_rate_num / max(1e-12, done_rate_den)) if done_rate_den > 0 else float("nan"),
        }
        # If done is a pure timeout at max_steps, then done_rate ~ 1 / max_steps.
        try:
            dr = float(group_stats["done_rate"])
            group_stats["estimated_episode_len_from_done_rate"] = (
                (1.0 / dr) if (dr > 0 and math.isfinite(dr)) else float("nan")
            )
        except Exception:
            pass

        # Quantiles (reward scalar + a few summary quantiles for obs norms).
        if rew_for_hist:
            r = torch.cat(rew_for_hist, dim=0).flatten()
            group_stats["reward_quantiles"] = {
                "q": [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
                "v": _quantiles(r, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]),
            }
            # Concentration / tail mass (helps interpret sparse-ish rewards).
            r_f = r[torch.isfinite(r)]
            if r_f.numel() > 0:
                group_stats["reward_mass"] = {
                    "frac_abs_le_1e-4": float((r_f.abs() <= 1e-4).float().mean().item()),
                    "frac_abs_le_1e-3": float((r_f.abs() <= 1e-3).float().mean().item()),
                    "frac_abs_le_1e-2": float((r_f.abs() <= 1e-2).float().mean().item()),
                    "frac_le_-0.02": float((r_f <= -0.02).float().mean().item()),
                    "frac_le_-0.05": float((r_f <= -0.05).float().mean().item()),
                    "frac_le_-0.1": float((r_f <= -0.1).float().mean().item()),
                    "frac_le_-0.5": float((r_f <= -0.5).float().mean().item()),
                    "frac_le_-1.0": float((r_f <= -1.0).float().mean().item()),
                    "frac_le_-2.0": float((r_f <= -2.0).float().mean().item()),
                }
        if obs_for_cov:
            o = torch.cat(obs_for_cov, dim=0)
            on = torch.linalg.norm(o, ord=2, dim=1)
            group_stats["observation_l2_norm_quantiles"] = {
                "q": [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
                "v": _quantiles(on, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]),
            }
            # Sparsity / mostly-zero diagnostics (common for lidar-like features in VMAS navigation).
            eps0 = 1e-8
            o_abs = o.abs()
            group_stats["observation_sparsity"] = {
                "eps0": eps0,
                "frac_abs_le_1e-3_per_dim": (o_abs <= 1e-3).float().mean(dim=0).detach().cpu().tolist(),
                "frac_eq0_per_dim": (o_abs <= eps0).float().mean(dim=0).detach().cpu().tolist(),
            }

        if act_for_misc:
            a = torch.cat(act_for_misc, dim=0)
            a = a[torch.isfinite(a).all(dim=1)]
            if a.numel() > 0:
                group_stats["action_saturation"] = {
                    "frac_abs_ge_0.99_per_dim": (a.abs() >= 0.99).float().mean(dim=0).detach().cpu().tolist(),
                    "frac_abs_ge_0.999_per_dim": (a.abs() >= 0.999).float().mean(dim=0).detach().cpu().tolist(),
                }

        if done_for_misc:
            d = torch.cat(done_for_misc, dim=0).flatten()
            d = d[torch.isfinite(d)]
            if d.numel() > 0:
                group_stats["done_stats"] = {
                    "count": int(d.numel()),
                    "count_done_1": int((d > 0.5).sum().item()),
                    "frac_done_1": float((d > 0.5).float().mean().item()),
                }

        # Coverage (1D bin occupancy per dim, using robust lo/hi from first-pass sample).
        cov: Dict[str, Any] = {}
        if obs_for_cov and "observation" in ranges.get(gname, {}):
            o = torch.cat(obs_for_cov, dim=0)
            lo = ranges[gname]["observation"]["lo"]
            hi = ranges[gname]["observation"]["hi"]
            cov_dim, oob_dim = _hist_coverage_1d(
                o, lo, hi, bins=bins, max_dims=int(coverage_max_dims)
            )
            cov["observation"] = {
                "bins": int(bins),
                "dims_covered": int(len(cov_dim)),
                "coverage_per_dim": cov_dim,
                "coverage_mean": float(torch.tensor(cov_dim).nanmean().item()) if cov_dim else float("nan"),
                "oob_rate_mean": float(torch.tensor(oob_dim).nanmean().item()) if oob_dim else float("nan"),
                "range_quantiles": {"q_lo": float(q_lo), "q_hi": float(q_hi)},
            }

        if rew_for_hist and "reward" in ranges.get(gname, {}):
            r = torch.cat(rew_for_hist, dim=0)
            lo = ranges[gname]["reward"]["lo"]
            hi = ranges[gname]["reward"]["hi"]
            cov_dim, oob_dim = _hist_coverage_1d(
                r, lo, hi, bins=bins, max_dims=1
            )
            cov["reward"] = {
                "bins": int(bins),
                "coverage": (float(cov_dim[0]) if cov_dim else float("nan")),
                "oob_rate": (float(oob_dim[0]) if oob_dim else float("nan")),
                "range_quantiles": {"q_lo": float(q_lo), "q_hi": float(q_hi)},
            }

        group_stats["coverage"] = cov

        # Plots
        if rew_for_hist:
            _maybe_plot_reward_hist(
                torch.cat(rew_for_hist, dim=0),
                plots_dir / f"{gname}_reward_hist.png",
                title=f"Reward histogram | group={gname}",
            )
        if "observation" in cov:
            _maybe_plot_coverage(
                cov["observation"]["coverage_per_dim"],
                plots_dir / f"{gname}_obs_coverage.png",
                title=f"Observation coverage (1D bin occupancy) | group={gname}",
            )
        if obs_for_pca:
            _maybe_plot_pca_occupancy(
                torch.cat(obs_for_pca, dim=0),
                plots_dir / f"{gname}_obs_pca_hist2d.png",
                title=f"Observation PCA occupancy | group={gname}",
            )

        stats["groups"][gname] = group_stats

    # Top-level writeout
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "BenchMARL dataset analysis output",
                f"- dataset_dir: {str(dataset_dir)}",
                f"- stats: {str((out_dir / 'stats.json').resolve())}",
                f"- plots_dir: {str(plots_dir.resolve())} (requires matplotlib: {'yes' if _HAS_MPL else 'no'})",
                "",
                "Notes:",
                "- Coverage metrics are approximate and computed on sampled transitions.",
                "- For continuous observations, 1D bin-occupancy is only a rough proxy for coverage.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=str, required=True, help="Dataset directory produced by benchmarl/collect.py")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <dataset-dir>/analysis)")
    p.add_argument(
        "--groups",
        type=str,
        default="",
        help="Comma-separated group names to analyze (default: auto-discover from shards)",
    )
    p.add_argument("--max-shards", type=int, default=None, help="Max shard files to read (default: all)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")

    p.add_argument("--bins", type=int, default=50, help="Bins for coverage histograms")
    p.add_argument("--q-lo", type=float, default=0.01, help="Lower quantile for robust coverage range")
    p.add_argument("--q-hi", type=float, default=0.99, help="Upper quantile for robust coverage range")
    p.add_argument("--range-samples", type=int, default=10_000, help="Samples used to estimate coverage ranges")

    p.add_argument(
        "--max-samples",
        type=int,
        default=200_000,
        help="Max transitions sampled per group (for stats/coverage/plots)",
    )
    p.add_argument(
        "--sample-per-shard",
        type=int,
        default=10_000,
        help="Max transitions sampled per shard per group",
    )
    p.add_argument(
        "--coverage-max-dims",
        type=int,
        default=64,
        help="Max feature dims to compute coverage for (still computes moments for all dims)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (dataset_dir / "analysis")
    groups = [g.strip() for g in str(args.groups).split(",") if g.strip()] or None

    stats = analyze(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        groups=groups,
        bins=int(args.bins),
        q_lo=float(args.q_lo),
        q_hi=float(args.q_hi),
        max_shards=args.max_shards,
        range_samples=int(args.range_samples),
        max_samples=int(args.max_samples),
        sample_per_shard=int(args.sample_per_shard),
        coverage_max_dims=int(args.coverage_max_dims),
        seed=int(args.seed),
    )

    print(f"\n[analyze_world_model_dataset] dataset_dir={str(dataset_dir)}")
    print(f"[analyze_world_model_dataset] out_dir={str(out_dir)}")
    print(f"[analyze_world_model_dataset] groups={list(stats.get('groups', {}).keys())}")
    if not _HAS_MPL:
        print("[analyze_world_model_dataset] matplotlib not available: plots disabled")
    print(f"[analyze_world_model_dataset] wrote: {str((out_dir / 'stats.json').resolve())}\n")


if __name__ == "__main__":
    main()

