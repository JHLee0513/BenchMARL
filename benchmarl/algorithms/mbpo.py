import json
import math
import pathlib
import warnings
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
    Unbounded,
)
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs import Compose, Transform
from torchrl.objectives import LossModule

from benchmarl.algorithms.mappo import Mappo, MappoConfig
from benchmarl.algorithms.masac import Masac, MasacConfig
from benchmarl.models.common import ModelConfig

try:
    # Optional: used to detect W&B logger for video logging (same pattern as Logger.log_evaluation).
    from torchrl.record.loggers.wandb import WandbLogger  # type: ignore
except Exception:  # pragma: no cover
    WandbLogger = None  # type: ignore


class _MixedReplayBuffer:
    """A thin wrapper that mixes samples from a real and a synthetic replay buffer.

    This lets us keep BenchMARL's core Experiment logic unchanged:
    - Experiment always calls `replay_buffer.extend(real_batch)` once per iteration.
    - Experiment always calls `replay_buffer.sample()` to obtain training minibatches.

    For MBPO we want synthetic rollouts to *not* pollute the on-policy (real) buffer,
    but we still want them to participate in training. This wrapper routes:
    - `extend()` -> real buffer
    - `sample()` -> returns a concatenation of samples from real + synthetic buffers
    """

    def __init__(
        self,
        *,
        real: ReplayBuffer,
        synthetic: ReplayBuffer,
        get_real_ratio: Callable[[], float],
        get_syn_ratio: Callable[[], Optional[float]],
        on_policy: bool,
    ):
        self._real = real
        self._synthetic = synthetic
        self._get_real_ratio = get_real_ratio
        self._get_syn_ratio = get_syn_ratio
        self._on_policy = on_policy
        self._last_sample_n_real: Optional[int] = None
        # When syn_ratio is used, we keep per-sample batch sizes bounded by the baseline
        # replay buffer batch size, and instead increase the number of optimizer minibatches.
        # This avoids OOM from returning huge concatenated batches.
        self._ratio_plan: List[Tuple[float, float]] = [(1.0, 0.0)]  # [(real_units, synth_units)]
        self._ratio_plan_idx: int = 0

    def _refresh_ratio_plan(self) -> None:
        """Build a per-call sampling plan from (real_ratio, syn_ratio) in 'unit' semantics.

        Each plan element is (real_units, synth_units) with real_units+synth_units <= 1.0.
        Over one full cycle of the plan, we will draw approximately:
          real_ratio * baseline_batch_size real samples, and
          syn_ratio  * baseline_batch_size synthetic samples.
        """
        sr = self._get_syn_ratio()
        if sr is None:
            self._ratio_plan = [(1.0, 0.0)]
            self._ratio_plan_idx = 0
            return

        r_rem = max(0.0, float(self._get_real_ratio() or 0.0))
        # On-policy algorithms (e.g., MBPO-MAPPO) must keep full coverage of the real batch
        # per PPO epoch. Synthetic samples are added *on top*; they must not dilute real data.
        if self._on_policy:
            if r_rem < 1.0 - 1e-12:
                # NOTE: This is a common source of confusion when users try real_ratio=0.0.
                # For on-policy PPO, we cannot "replace" real samples with synthetic ones without
                # turning the update off-policy (log_probs/advantages would no longer match the
                # collected trajectory distribution). Therefore, `real_ratio` is clamped to 1.0
                # under syn_ratio semantics; only `syn_ratio` changes the amount of synthetic data.
                if not getattr(self, "_warned_real_ratio_on_policy_syn_ratio", False):
                    warnings.warn(
                        "MBPO-MAPPO: `syn_ratio` is set, and this is an on-policy algorithm. "
                        "`real_ratio < 1.0` cannot be honored (it would dilute/replace on-policy data). "
                        "Clamping `real_ratio` to 1.0, so only `syn_ratio` affects sampling. "
                        "If you want the old semantics where `real_ratio` controls how many synthetic "
                        "samples are added, set `syn_ratio=null` and use `real_ratio` instead."
                    )
                    setattr(self, "_warned_real_ratio_on_policy_syn_ratio", True)
            r_rem = max(1.0, r_rem)
        s_rem = max(0.0, float(sr or 0.0))
        plan: List[Tuple[float, float]] = []
        # Greedy packing into capacity-1.0 "minibatch slots": fill real first, then synthetic.
        while r_rem > 1e-12 or s_rem > 1e-12:
            r_take = min(1.0, r_rem)
            r_rem -= r_take
            cap = 1.0 - r_take
            s_take = min(cap, s_rem)
            s_rem -= s_take
            plan.append((float(r_take), float(s_take)))
        if not plan:
            plan = [(1.0, 0.0)]
        self._ratio_plan = plan
        self._ratio_plan_idx = 0

    @property
    def storage(self):
        # Experiment uses `group_buffer.storage.device` before extend.
        return self._real.storage

    @property
    def batch_size(self):
        return getattr(self._real, "batch_size", None)

    def __len__(self):
        return len(self._real)

    def state_dict(self):
        return {
            "real": self._real.state_dict(),
            "synthetic": self._synthetic.state_dict(),
        }

    def load_state_dict(self, state_dict):
        # Backward compatible: if older checkpoints only contain a single buffer state,
        # load it into the real buffer.
        if isinstance(state_dict, dict) and "real" in state_dict and "synthetic" in state_dict:
            self._real.load_state_dict(state_dict["real"])
            self._synthetic.load_state_dict(state_dict["synthetic"])
        else:
            self._real.load_state_dict(state_dict)

    def empty(self):
        if hasattr(self._real, "empty"):
            self._real.empty()
        if hasattr(self._synthetic, "empty"):
            self._synthetic.empty()

    def extend(self, data: TensorDictBase):
        # Treat each extend() as a new "iteration" boundary and reset the plan pointer.
        # Experiment calls extend() exactly once per iteration.
        try:
            self._refresh_ratio_plan()
        except Exception:
            # Keep previous plan on failure.
            pass
        return self._real.extend(data)

    def update_tensordict_priority(self, data: TensorDictBase):
        # Priorities apply to the real buffer only.
        if hasattr(self._real, "update_tensordict_priority"):
            n_real = self._last_sample_n_real
            if n_real is not None:
                return self._real.update_tensordict_priority(data[:n_real])
            return self._real.update_tensordict_priority(data)

    def _safe_sample(self, buffer: ReplayBuffer, n: int) -> Optional[TensorDictBase]:
        if n <= 0:
            return None
        # Clamp to buffer length when available (avoids errors for SamplerWithoutReplacement).
        try:
            buf_len = int(len(buffer))
            if buf_len <= 0:
                return None
            n = min(int(n), buf_len)
        except Exception:
            pass
        try:
            return buffer.sample(batch_size=n)
        except TypeError:
            # Some torchrl versions don't accept batch_size kwarg.
            try:
                return buffer.sample(n)
            except Exception:
                td = buffer.sample()
                return td[:n]

    def sample(self, *args, **kwargs) -> TensorDictBase:
        # Experiment calls `sample()` without args, so we treat any passed args as optional.
        if args:
            total = int(args[0])
        else:
            total = int(kwargs.get("batch_size", self.batch_size))

        # Guard against unexpected None.
        total = max(1, int(total))

        # New semantics (optional): if syn_ratio is set, interpret ratios as *multipliers*
        # of the baseline minibatch size (the replay buffer's configured batch_size).
        # - n_real  = ceil(total * real_ratio)
        # - n_synth = ceil(total * syn_ratio)
        #
        # Backward compatible: if syn_ratio is None, keep the old "real_ratio is a fraction"
        # behavior, where synthetic count is derived as (1-real_ratio).
        sr_raw = self._get_syn_ratio()
        use_new = sr_raw is not None

        if use_new:
            # Bounded minibatch: choose (real_units, synth_units) from the plan and sample at most `total`.
            if not self._ratio_plan:
                self._refresh_ratio_plan()
            ru, su = self._ratio_plan[self._ratio_plan_idx % len(self._ratio_plan)]
            self._ratio_plan_idx += 1

            # Convert units to counts, ensuring we never exceed `total`.
            ru = max(0.0, float(ru))
            su = max(0.0, float(su))
            if ru + su >= 1.0 - 1e-12:
                # Fill the whole minibatch.
                n_real = int(math.floor(total * ru))
                n_real = min(total, max(0, n_real))
                n_synth = total - n_real
            else:
                n_real = int(math.floor(total * ru))
                n_real = min(total, max(0, n_real))
                remaining = total - n_real
                n_synth = min(remaining, int(math.ceil(total * su)))

            self._last_sample_n_real = n_real
            real_td = self._safe_sample(self._real, n_real) if n_real > 0 else None
            synth_td = self._safe_sample(self._synthetic, n_synth) if n_synth > 0 else None

            if real_td is None and synth_td is None:
                return self._safe_sample(self._real, total)
            if real_td is None:
                return synth_td
            if synth_td is None:
                # CRITICAL: When synthetic buffer is empty, we must sample exactly `total` samples
                # from the real buffer to match baseline RNG consumption. The ratio plan might
                # have calculated n_real < total, which would cause RNG differences.
                # Return a fresh sample with the exact requested batch size.
                return self._safe_sample(self._real, total)
            return torch.cat([real_td, synth_td], dim=0)

        rr = float(self._get_real_ratio() or 1.0)
        rr = min(max(rr, 0.0), 1.0)
        if not self._on_policy:
            # Off-policy: keep previous semantics (fixed total size with rr fraction real).
            n_real = max(1, min(total, int(math.ceil(total * rr))))
            n_synth = max(0, total - n_real)
            self._last_sample_n_real = n_real
            if n_synth <= 0:
                return self._safe_sample(self._real, total)
            try:
                synth_len = len(self._synthetic)
            except Exception:
                synth_len = 0
            if synth_len <= 0:
                return self._safe_sample(self._real, total)
            n_synth = min(n_synth, synth_len)
            real_td = self._safe_sample(self._real, total - n_synth)
            synth_td = self._safe_sample(self._synthetic, n_synth)
            if synth_td is None:
                return real_td
            return torch.cat([real_td, synth_td], dim=0)

        # On-policy: keep the requested number of *real* samples fixed, and add synthetic on top.
        n_real = total
        n_synth_target = int(math.ceil(float(n_real) * (1.0 - rr)))

        # If no synthetic is requested (rr=1), behave exactly like the real buffer.
        if n_synth_target <= 0:
            self._last_sample_n_real = n_real
            return self._safe_sample(self._real, n_real)

        try:
            synth_len = len(self._synthetic)
        except Exception:
            synth_len = 0
        if synth_len <= 0:
            # No synthetic available yet (e.g., warmup): keep MAPPO-equivalent sampling.
            self._last_sample_n_real = n_real
            return self._safe_sample(self._real, n_real)

        n_synth = min(n_synth_target, synth_len)
        real_td = self._safe_sample(self._real, n_real)
        synth_td = self._safe_sample(self._synthetic, n_synth) if n_synth > 0 else None

        self._last_sample_n_real = n_real
        if synth_td is None:
            return real_td
        return torch.cat([real_td, synth_td], dim=0)


class _DynamicsTrainBuffer:
    """Simple ring buffer for TensorDicts used to train the dynamics model.

    Key design goal: sampling must NOT consume the global torch RNG (to keep on-policy
    training identical to model-free baselines until synthetic rollouts are used).
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = max(1, int(capacity))
        self.device = device
        self._storage: Optional[TensorDictBase] = None
        self._cursor: int = 0
        self._len: int = 0

    def __len__(self) -> int:
        return self._len

    def extend(self, td: TensorDictBase) -> None:
        if td is None or td.numel() == 0:
            return
        td = td.detach()

        # Ensure leading dimension exists.
        if td.batch_size is None or len(td.batch_size) == 0:
            td = td.reshape(-1)

        n = int(td.batch_size[0])
        if n <= 0:
            return

        if self._storage is None:
            # Allocate storage with fixed capacity.
            self._storage = td[: min(n, self.capacity)].to(self.device)
            # If n < capacity, expand storage by cloning schema and resizing batch dim.
            if self._storage.batch_size[0] < self.capacity:
                # Create empty storage by repeating first element then overwriting.
                first = self._storage[:1]
                self._storage = torch.cat(
                    [first.expand(self.capacity), self._storage], dim=0
                )[: self.capacity]
            self._cursor = 0
            self._len = 0

        # Write in chunks to handle wrap-around.
        start = 0
        while start < n:
            remaining = n - start
            space_to_end = self.capacity - self._cursor
            k = min(remaining, space_to_end)
            chunk = td[start : start + k].to(self.device)
            self._storage[self._cursor : self._cursor + k] = chunk
            self._cursor = (self._cursor + k) % self.capacity
            self._len = min(self.capacity, self._len + k)
            start += k

    def sample(self, batch_size: int, *, generator: Optional[torch.Generator]) -> TensorDictBase:
        if self._storage is None or self._len <= 0:
            raise RuntimeError("DynamicsTrainBuffer is empty")
        batch_size = max(1, int(batch_size))
        # Sample indices using the provided generator on CPU to avoid global RNG.
        idx = torch.randint(
            0,
            self._len,
            (batch_size,),
            device="cpu",
            generator=generator,
        ).to(self.device)
        return self._storage[idx]


class _DynamicsModel(nn.Module):
    """
    Simple dynamics model that predicts next observations, rewards and done logits
    for a whole agent group.

    It supports two modes:
    - per-agent: predicts per-agent next observation, reward and done.
    - centralized: predicts joint next observation and per-agent reward/done from
      joint observation and joint action.
    """

    def __init__(
        self,
        input_dim: int,
        next_obs_dim: int,
        reward_dim: int,
        done_dim: int,
        hidden_size: int,
        num_layers: int,
        stochastic: bool,
        min_log_var: float,
        max_log_var: float,
        separate_reward_net: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for _ in range(num_layers):
            # layers += [nn.Linear(last_dim, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU()]
            layers += [nn.Linear(last_dim, hidden_size), nn.LeakyReLU()]
            last_dim = hidden_size
        self.net = nn.Sequential(*layers)

        self._stochastic = stochastic
        self._separate_reward_net = bool(separate_reward_net)
        # Bounds used to keep log-variances well-behaved (softplus clamp like PETS/MAMBPO)
        self.register_buffer("_min_log_var", torch.tensor(float(min_log_var)))
        self.register_buffer("_max_log_var", torch.tensor(float(max_log_var)))

        # Predict mean(delta_next_obs), mean(reward), and optionally log-variances; plus done logits.
        # We keep separate heads for obs vs reward; optionally also separate the trunk for reward.
        self.delta_mu_head = nn.Linear(last_dim, next_obs_dim)
        self.delta_log_var_head = nn.Linear(last_dim, next_obs_dim)

        if self._separate_reward_net:
            r_layers: List[nn.Module] = []
            r_last = input_dim
            for _ in range(num_layers):
                r_layers += [nn.Linear(r_last, hidden_size), nn.LeakyReLU()]
                r_last = hidden_size
            self.reward_net = nn.Sequential(*r_layers)
            reward_last_dim = r_last
        else:
            self.reward_net = None
            reward_last_dim = last_dim

        self.rew_mu_head = nn.Linear(reward_last_dim, reward_dim)
        self.rew_log_var_head = nn.Linear(reward_last_dim, reward_dim)
        self.done_head = nn.Linear(last_dim, done_dim)

        self._next_obs_dim = next_obs_dim
        self._reward_dim = reward_dim
        self._done_dim = done_dim

    def forward(
        self, obs_flat: torch.Tensor, action_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_flat, action_flat], dim=-1)
        feat = self.net(x)
        delta_mu = self.delta_mu_head(feat)
        delta_log_var = self.delta_log_var_head(feat)

        if self.reward_net is None:
            r_feat = feat
        else:
            r_feat = self.reward_net(x)
        rew_mu = self.rew_mu_head(r_feat)
        rew_log_var = self.rew_log_var_head(r_feat)

        done_logit = self.done_head(feat)

        # Softplus-based clamp to [min_log_var, max_log_var]
        max_lv = self._max_log_var
        min_lv = self._min_log_var
        delta_log_var = max_lv - F.softplus(max_lv - delta_log_var)
        delta_log_var = min_lv + F.softplus(delta_log_var - min_lv)
        rew_log_var = max_lv - F.softplus(max_lv - rew_log_var)
        rew_log_var = min_lv + F.softplus(rew_log_var - min_lv)

        if not self._stochastic:
            delta_log_var = delta_log_var * 0.0
            rew_log_var = rew_log_var * 0.0

        mu_next_obs = obs_flat + delta_mu
        # return mu_next_obs, delta_log_var, rew_mu, rew_log_var, done_logit
        return delta_mu, delta_log_var, rew_mu, rew_log_var, done_logit


class _MbpoWorldModelMixin:
    """MBPO world-model implementation shared across different base algorithms."""

    #############################
    # World-model debug video
    #############################

    def _wm_vis_enabled(self) -> bool:
        return bool(getattr(self, "world_model_debug_video", False))

    def _wm_vis_interval(self) -> int:
        return max(
            1, int(getattr(self, "world_model_debug_video_interval", 0) or 1)
        )

    def _wm_vis_horizon(self) -> int:
        return max(1, int(getattr(self, "world_model_debug_video_horizon", 0) or 1))

    def _wm_vis_fps(self) -> int:
        return max(1, int(getattr(self, "world_model_debug_video_fps", 0) or 20))

    def _wm_vis_env_index(self) -> int:
        return max(0, int(getattr(self, "world_model_debug_video_env_index", 0) or 0))

    def _wm_vis_mode(self) -> str:
        """Either 'open_loop' (default) or 'closed_loop'."""
        mode = str(getattr(self, "world_model_debug_video_mode", "open_loop") or "open_loop")
        mode = mode.strip().lower()
        if mode not in ("open_loop", "closed_loop"):
            mode = "open_loop"
        return mode

    def _wm_vis_xy_indices(self) -> Optional[Tuple[int, int]]:
        idx = getattr(self, "world_model_debug_video_obs_xy_indices", None)
        if idx is None:
            return None
        try:
            if isinstance(idx, (list, tuple)) and len(idx) == 2:
                return (int(idx[0]), int(idx[1]))
        except Exception:
            return None
        return None

    def _wm_vis_should_log(self, group: str) -> bool:
        if not self._wm_vis_enabled():
            return False
        if not hasattr(self, "experiment") or self.experiment is None:
            return False
        if not hasattr(self.experiment, "logger") or self.experiment.logger is None:
            return False
        step = int(getattr(self, "_train_steps", {}).get(group, 0))
        return step > 0 and (step % self._wm_vis_interval() == 0)

    def _wm_vis_extract_xy(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract 2D coordinates from an observation tensor for visualization.

        This is task-dependent: by default we use the first two obs dims, or the
        indices in `world_model_debug_video_obs_xy_indices` if provided.
        """
        xy_idx = self._wm_vis_xy_indices()
        if xy_idx is not None:
            i, j = xy_idx
            if obs.shape[-1] > max(i, j):
                return torch.stack([obs[..., i], obs[..., j]], dim=-1)
        if obs.shape[-1] >= 2:
            return obs[..., :2]
        z = torch.zeros((*obs.shape[:-1], 2), device=obs.device, dtype=obs.dtype)
        if obs.shape[-1] == 1:
            z[..., 0] = obs[..., 0]
        return z

    def _wm_vis_render_frames(
        self, *, group: str, traj: TensorDictBase, horizon: int
    ) -> Optional[List]:
        """Create RGB frames comparing simulator transitions vs world-model predictions.

        Each frame contains:
        - left: ground-truth obs_t -> obs_{t+1} (as 2D points/arrows)
        - right: predicted obs_t -> predicted obs_{t+1} (closed-loop rollout)
        - bottom: reward curve (true vs predicted) up to current t
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        except Exception:
            return None

        keys = traj.keys(True, True)
        if (group, "observation") not in keys or ("next", group, "observation") not in keys:
            return None
        if (group, "action") not in keys:
            return None

        if not hasattr(traj, "batch_size") or len(traj.batch_size) == 0:
            return None
        T = int(traj.batch_size[0])
        if T <= 0:
            return None

        horizon = max(1, min(int(horizon), T))

        # Closed-loop state (only used when mode == 'closed_loop').
        obs_pred: Optional[torch.Tensor] = None
        if self._wm_vis_mode() == "closed_loop":
            obs0 = traj[0].get((group, "observation")).to(self.device)
            obs_pred = obs0.unsqueeze(0) if obs0.dim() == 2 else obs0

        reward_true_hist: List[float] = []
        reward_pred_hist: List[float] = []
        frames: List = []

        # Stable limits from ground truth, if possible.
        xlim = None
        ylim = None
        try:
            all_xy = []
            for t in range(horizon):
                o = traj[t].get((group, "observation")).to(self.device)
                no = traj[t].get(("next", group, "observation")).to(self.device)
                o = o.unsqueeze(0) if o.dim() == 2 else o
                no = no.unsqueeze(0) if no.dim() == 2 else no
                all_xy.append(self._wm_vis_extract_xy(o[0]))
                all_xy.append(self._wm_vis_extract_xy(no[0]))
            all_xy = torch.cat(all_xy, dim=0)
            xmin = float(all_xy[:, 0].min().detach().cpu().item())
            xmax = float(all_xy[:, 0].max().detach().cpu().item())
            ymin = float(all_xy[:, 1].min().detach().cpu().item())
            ymax = float(all_xy[:, 1].max().detach().cpu().item())
            pad_x = 0.1 * max(1e-6, xmax - xmin)
            pad_y = 0.1 * max(1e-6, ymax - ymin)
            xlim = (xmin - pad_x, xmax + pad_x)
            ylim = (ymin - pad_y, ymax + pad_y)
        except Exception:
            pass

        for t in range(horizon):
            td_t = traj[t]
            obs_true = td_t.get((group, "observation")).to(self.device)
            act_true = td_t.get((group, "action")).to(self.device)
            next_obs_true = td_t.get(("next", group, "observation")).to(self.device)
            rew_true = td_t.get(("next", group, "reward"), None)
            if rew_true is None:
                rew_true = td_t.get(("next", "reward"), None)

            obs_true_b = obs_true.unsqueeze(0) if obs_true.dim() == 2 else obs_true
            act_true_b = act_true.unsqueeze(0) if act_true.dim() == 2 else act_true
            next_obs_true_b = (
                next_obs_true.unsqueeze(0) if next_obs_true.dim() == 2 else next_obs_true
            )

            # IMPORTANT: to compare fairly, use the SAME starting state as the simulator:
            # - open_loop: predict from simulator obs_true_b every step
            # - closed_loop: predict from previous model prediction obs_pred (initialized from t=0)
            if self._wm_vis_mode() == "closed_loop":
                if obs_pred is None:
                    obs_pred = obs_true_b
                obs_in = obs_pred
            else:
                obs_in = obs_true_b

            oracle_vis = bool(getattr(self, "use_oracle", False))
            oracle_used = False
            oracle_failed = False
            with torch.no_grad():
                if oracle_vis:
                    # IMPORTANT:
                    # Most TorchRL envs cannot be "stepped from an arbitrary observation" without
                    # also restoring the full simulator state. Attempting to call env.step() with
                    # a TensorDict that merely contains `observation` will often:
                    # - silently step from the env's *internal* state (unrelated to obs_in), or
                    # - error out.
                    #
                    # That makes oracle-visualization misleading. For `use_oracle=True`, what we
                    # actually want to show is the *oracle transition* (i.e., the simulator ground truth).
                    #
                    # Therefore, for visualization we set the "predicted" next state to the true
                    # next state from the trajectory, and likewise use the true reward/done when present.
                    oracle_used = True
                    next_obs_pred_b = next_obs_true_b

                    # Reward: best-effort extraction + shaping to [B, ..., 1]
                    if rew_true is not None:
                        rew_pred_b = rew_true.to(self.device)
                        if rew_pred_b.dim() == 2:
                            rew_pred_b = rew_pred_b.unsqueeze(0)
                        elif rew_pred_b.dim() == 1:
                            rew_pred_b = rew_pred_b.view(1, -1, 1)
                        if rew_pred_b.shape[-1] != 1:
                            rew_pred_b = rew_pred_b.unsqueeze(-1)
                    else:
                        rew_pred_b = torch.zeros(
                            (*next_obs_true_b.shape[:-1], 1),
                            device=self.device,
                            dtype=torch.float32,
                        )

                    # Done: best-effort extraction from traj; otherwise assume not-done.
                    done_true = td_t.get(("next", group, "done"), None)
                    if done_true is None:
                        done_true = td_t.get(("next", "done"), None)
                    if done_true is not None:
                        done_true_b = done_true.to(self.device).to(torch.bool)
                        if done_true_b.dim() == 2:
                            done_true_b = done_true_b.unsqueeze(0)
                        if done_true_b.shape[-1] != 1:
                            done_true_b = done_true_b.unsqueeze(-1)
                    else:
                        done_true_b = torch.zeros(
                            (*next_obs_true_b.shape[:-1], 1),
                            device=self.device,
                            dtype=torch.bool,
                        )
                    done_logit_b = done_true_b.float() * 10.0 - 5.0
                else:
                    # For visualization, use mean predictions (not stochastic samples) to show model accuracy
                    # rather than sampling noise. This makes it easier to see if the model is learning correctly.
                    next_obs_pred_b, rew_pred_b, done_logit_b = self._predict_next_for_vis(
                        group, obs_in, act_true_b
                    )

            # Try multiple ways to extract ground truth reward
            r_true = 0.0
            if rew_true is not None:
                try:
                    r_true = float(rew_true.to(torch.float32).mean().detach().cpu().item())
                except Exception:
                    pass
            # Also try getting reward from the group's episode_reward or other keys
            if r_true == 0.0:
                try:
                    ep_rew = td_t.get(("next", group, "episode_reward"), None)
                    if ep_rew is not None:
                        r_true = float(ep_rew.to(torch.float32).mean().detach().cpu().item())
                except Exception:
                    pass
            if r_true == 0.0:
                try:
                    ep_rew = td_t.get(("next", "episode_reward"), None)
                    if ep_rew is not None:
                        # Expand to match group shape if needed
                        if hasattr(ep_rew, 'expand') and hasattr(td_t, 'get'):
                            group_td = td_t.get(group, None)
                            if group_td is not None and hasattr(group_td, 'shape'):
                                ep_rew = ep_rew.expand(group_td.shape).unsqueeze(-1)
                        r_true = float(ep_rew.to(torch.float32).mean().detach().cpu().item())
                except Exception:
                    pass
            r_pred = float(rew_pred_b.to(torch.float32).mean().detach().cpu().item())
            reward_true_hist.append(r_true)
            reward_pred_hist.append(r_pred)

            obs_rmse = float(
                torch.sqrt(torch.mean((next_obs_pred_b - next_obs_true_b) ** 2))
                .detach()
                .cpu()
                .item()
            )
            done_prob = float(torch.sigmoid(done_logit_b).mean().detach().cpu().item())

            xy_true = self._wm_vis_extract_xy(obs_true_b[0])
            xy_true_next = self._wm_vis_extract_xy(next_obs_true_b[0])
            # Pred panel uses the simulator start state too (so arrows start at same position),
            # unless closed-loop mode is enabled.
            xy_pred = self._wm_vis_extract_xy(obs_in[0])
            xy_pred_next = self._wm_vis_extract_xy(next_obs_pred_b[0])

            fig = plt.Figure(figsize=(9.5, 6.0), dpi=120)
            canvas = FigureCanvas(fig)
            gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0])
            ax_true = fig.add_subplot(gs[0, 0])
            ax_pred = fig.add_subplot(gs[0, 1])
            ax_rew = fig.add_subplot(gs[1, :])

            ax_true.set_title("Simulator (ground truth): state change")
            ax_true.scatter(
                xy_true[:, 0].detach().cpu(),
                xy_true[:, 1].detach().cpu(),
                s=40,
                label="obs_t",
            )
            ax_true.scatter(
                xy_true_next[:, 0].detach().cpu(),
                xy_true_next[:, 1].detach().cpu(),
                s=40,
                marker="x",
                label="obs_{t+1}",
            )
            for i in range(xy_true.shape[0]):
                ax_true.plot(
                    [
                        float(xy_true[i, 0].detach().cpu()),
                        float(xy_true_next[i, 0].detach().cpu()),
                    ],
                    [
                        float(xy_true[i, 1].detach().cpu()),
                        float(xy_true_next[i, 1].detach().cpu()),
                    ],
                    linewidth=1.0,
                    alpha=0.8,
                )
            if xlim is not None:
                ax_true.set_xlim(*xlim)
            if ylim is not None:
                ax_true.set_ylim(*ylim)
            ax_true.set_aspect("equal", adjustable="box")
            ax_true.grid(True, alpha=0.25)
            ax_true.legend(loc="upper right", fontsize=8)

            if oracle_used:
                ax_pred.set_title("Oracle (simulator): state change")
            else:
                ax_pred.set_title("World model: predicted state change")
            ax_pred.scatter(
                xy_pred[:, 0].detach().cpu(),
                xy_pred[:, 1].detach().cpu(),
                s=40,
                label="obs_t",
            )
            ax_pred.scatter(
                xy_pred_next[:, 0].detach().cpu(),
                xy_pred_next[:, 1].detach().cpu(),
                s=40,
                marker="x",
                label="obs_{t+1} (pred)",
            )
            for i in range(xy_pred.shape[0]):
                ax_pred.plot(
                    [
                        float(xy_pred[i, 0].detach().cpu()),
                        float(xy_pred_next[i, 0].detach().cpu()),
                    ],
                    [
                        float(xy_pred[i, 1].detach().cpu()),
                        float(xy_pred_next[i, 1].detach().cpu()),
                    ],
                    linewidth=1.0,
                    alpha=0.8,
                )
            if xlim is not None:
                ax_pred.set_xlim(*xlim)
            if ylim is not None:
                ax_pred.set_ylim(*ylim)
            ax_pred.set_aspect("equal", adjustable="box")
            ax_pred.grid(True, alpha=0.25)
            ax_pred.legend(loc="upper right", fontsize=8)

            ax_rew.set_title("Reward (mean over agents): predicted vs ground truth")
            xs = np.arange(len(reward_true_hist))
            ax_rew.plot(xs, reward_true_hist, label="true", linewidth=2.0)
            ax_rew.plot(xs, reward_pred_hist, label="pred", linewidth=2.0)
            ax_rew.axvline(len(reward_true_hist) - 1, color="k", alpha=0.2)
            ax_rew.grid(True, alpha=0.25)
            ax_rew.legend(loc="upper right", fontsize=9)
            ax_rew.set_xlabel("t")

            title_line1 = (
                f"MBPO world-model debug ({self._wm_vis_mode()}) | group={group} | t={t}"
            )
            title_line2 = (
                f"obs_rmse={obs_rmse:.4f} | r_true={r_true:.4f} r_pred={r_pred:.4f} | done_prob~{done_prob:.2f}"
                + (
                    f" | oracle_vis=1 used={int(oracle_used)} failed={int(oracle_failed)}"
                    if oracle_vis
                    else ""
                )
            )
            fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=10)

            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            frames.append(rgba[..., :3].copy().astype(np.uint8))

            # Advance closed-loop state only if enabled.
            if self._wm_vis_mode() == "closed_loop":
                obs_pred = next_obs_pred_b.detach()

        return frames

    def _wm_vis_log_video(self, *, group: str, frames: List) -> None:
        if not frames:
            return
        try:
            import numpy as np
        except Exception:
            return

        video_frames = np.stack(frames, axis=0)  # [T,H,W,3]
        vid = torch.tensor(
            np.transpose(video_frames, (0, 3, 1, 2)), dtype=torch.uint8
        ).unsqueeze(0)

        step = getattr(self.experiment, "total_frames", None)
        if step is None:
            step = int(getattr(self, "_train_steps", {}).get(group, 0))

        for logger in getattr(self.experiment.logger, "loggers", []):
            if WandbLogger is not None and isinstance(logger, WandbLogger):
                logger.log_video(
                    f"world_model/{group}/debug_video",
                    vid,
                    fps=self._wm_vis_fps(),
                    commit=False,
                )
            else:
                # Other loggers cannot deal with odd video sizes; mirror Logger.log_evaluation behavior.
                for index in (-1, -2):
                    if vid.shape[index] % 2 != 0:
                        vid = vid.index_select(index, torch.arange(1, vid.shape[index]))
                logger.log_video(f"world_model_{group}_debug_video", vid, step=step)

    def _maybe_log_world_model_debug_video(
        self, *, group: str, batch: TensorDictBase
    ) -> None:
        if not self._wm_vis_should_log(group):
            return
        if batch is None or batch.numel() == 0:
            return

        traj = None
        try:
            if not hasattr(batch, "batch_size") or len(batch.batch_size) == 0:
                return

            # We want a *time-contiguous* trajectory from a *single* env index.
            # Collector output is typically 2D, but the order can be [T,B] or [B,T].
            if len(batch.batch_size) >= 2:
                env_i = self._wm_vis_env_index()

                # Use collector/traj_ids (if available) to infer which axis is time:
                # the time axis should have *few* switches in traj_id for a fixed env.
                time_axis = 0  # default assume [T,B]
                try:
                    traj_ids = batch.get(("collector", "traj_ids"), None)
                except Exception:
                    traj_ids = None

                if traj_ids is not None and hasattr(traj_ids, "ndim") and int(traj_ids.ndim) == 2:
                    # Candidate 1: axis0 is time => seq = traj_ids[:, env]
                    env0 = min(env_i, int(traj_ids.shape[1]) - 1)
                    seq0 = traj_ids[:, env0]
                    # Candidate 2: axis1 is time => seq = traj_ids[env, :]
                    env1 = min(env_i, int(traj_ids.shape[0]) - 1)
                    seq1 = traj_ids[env1, :]
                    try:
                        sw0 = float((seq0[1:] != seq0[:-1]).float().mean().detach().cpu().item())
                    except Exception:
                        sw0 = 1.0
                    try:
                        sw1 = float((seq1[1:] != seq1[:-1]).float().mean().detach().cpu().item())
                    except Exception:
                        sw1 = 1.0
                    time_axis = 0 if sw0 <= sw1 else 1

                # Extract env trajectory along inferred time axis.
                if time_axis == 0:
                    env_i = min(env_i, int(batch.batch_size[1]) - 1)
                    traj = batch[:, env_i]
                else:
                    env_i = min(env_i, int(batch.batch_size[0]) - 1)
                    traj = batch[env_i, :]

                # Cut to a single contiguous "episode segment" using traj_ids if possible.
                # This avoids videos that look like random frames due to env resets within the batch.
                try:
                    ids = traj.get(("collector", "traj_ids"), None)
                except Exception:
                    ids = None
                if ids is not None:
                    ids = ids.detach()
                    if hasattr(ids, "ndim") and int(ids.ndim) >= 1:
                        ids_1d = ids.reshape(-1)
                        if ids_1d.numel() >= 2:
                            change = (ids_1d[1:] != ids_1d[:-1]).nonzero(as_tuple=True)[0] + 1
                            starts = torch.cat(
                                [torch.zeros(1, device=change.device, dtype=change.dtype), change],
                                dim=0,
                            )
                            ends = torch.cat(
                                [change, torch.tensor([ids_1d.numel()], device=change.device, dtype=change.dtype)],
                                dim=0,
                            )
                            # Pick the longest segment (more likely to be a coherent episode chunk).
                            lens = (ends - starts).to(torch.long)
                            seg_idx = int(torch.argmax(lens).detach().cpu().item())
                            seg_start = int(starts[seg_idx].detach().cpu().item())
                            seg_end = int(ends[seg_idx].detach().cpu().item())
                            if seg_end - seg_start >= 2:
                                traj = traj[seg_start:seg_end]

                # Also cut at the first done within the selected segment to mimic a single episode.
                try:
                    d = traj.get(("next", "done"), None)
                except Exception:
                    d = None
                if d is None:
                    try:
                        d = traj.get(("next", group, "done"), None)
                        if d is not None and d.dim() >= 2:
                            # Reduce agent dim.
                            d = d.any(-2)
                    except Exception:
                        d = None
                if d is not None:
                    d1 = d.reshape(-1).to(torch.bool)
                    done_idx = d1.nonzero(as_tuple=True)[0]
                    if done_idx.numel() > 0:
                        stop = int(done_idx[0].detach().cpu().item()) + 1
                        if stop >= 2:
                            traj = traj[:stop]

            # Fallback: treat single-dim batch as time-indexable.
            elif len(batch.batch_size) == 1:
                if int(batch.batch_size[0]) > 1:
                    traj = batch
        except Exception:
            traj = None

        if traj is None:
            return
        # Need at least 2 steps to visualize a state change.
        if not hasattr(traj, "batch_size") or int(traj.batch_size[0]) < 2:
            return

        # IMPORTANT: video generation must not change training results.
        # Rendering calls into the world model (e.g., ensemble member selection via torch.randint),
        # which consumes torch RNG. If we let that touch the global RNG stream, it will change
        # synthetic rollout randomness / sampling order and produce different policies.
        #
        # We therefore fork RNG state (CPU + CUDA) so any RNG consumption here is rolled back.
        devices: List[int] = []
        try:
            dev = torch.device(self.device)
            if dev.type == "cuda":
                devices = [int(dev.index or 0)]
        except Exception:
            devices = []

        with torch.random.fork_rng(devices=devices, enabled=True):
            # Optional: seed the visualization RNG for stable videos across runs.
            try:
                base_seed = int(getattr(self.experiment, "seed", 0))
                step = int(getattr(self, "_train_steps", {}).get(group, 0))
                torch.manual_seed(base_seed + 99_999 + step)
                if devices:
                    torch.cuda.manual_seed_all(base_seed + 99_999 + step)
            except Exception:
                pass

            frames = self._wm_vis_render_frames(
                group=group, traj=traj, horizon=self._wm_vis_horizon()
            )
        if frames is None:
            return
        self._wm_vis_log_video(group=group, frames=frames)

    def get_replay_buffer(
        self, group: str, transforms: List = None
    ) -> ReplayBuffer:
        # Create the real buffer using the default BenchMARL logic.
        real_buffer = super().get_replay_buffer(group=group, transforms=transforms)

        # Check if we should skip world model operations entirely (e.g., threshold=0.0)
        # If so, return the real buffer directly without the MixedReplayBuffer wrapper
        # to ensure exact equivalence with the baseline algorithm.
        # CRITICAL: When threshold=0.0, we must bypass _MixedReplayBuffer even if syn_ratio > 0,
        # because _MixedReplayBuffer.sample() with syn_ratio > 0 uses ratio planning that can
        # affect RNG consumption even when the synthetic buffer is empty.
        skip_world_model = False
        unc_threshold = self._get_rollout_uncertainty_threshold(group)
        if unc_threshold is not None and unc_threshold == 0.0:
            if not getattr(self, "use_oracle", False):
                skip_world_model = True
        if skip_world_model:
            # Return the real buffer directly, bypassing MixedReplayBuffer entirely.
            # This ensures exact equivalence with the baseline algorithm:
            # - No _MixedReplayBuffer wrapper overhead
            # - No ratio planning logic that could affect RNG
            # - Direct pass-through to underlying real buffer
            # - Same sampling behavior as baseline MAPPO
            return real_buffer

        # Create a dedicated synthetic buffer with enough capacity for model rollouts.
        # We size it relative to the on-policy real buffer size for this run.
        base_memory_size = self.experiment_config.replay_buffer_memory_size(self.on_policy)
        sr = getattr(self, "syn_ratio", None)
        if sr is not None:
            # New semantics: `syn_ratio` is a multiplier of the baseline batch.
            synthetic_memory_size = int(math.ceil(base_memory_size * max(0.0, float(sr))))
        else:
            # Backward-compatible semantics: synthetic count per iteration ~= (1-real_ratio) * N_real.
            rr = float(getattr(self, "real_ratio", 1.0))
            rr = min(max(rr, 0.0), 1.0)
            synthetic_memory_size = int(math.ceil(base_memory_size * max(0.0, (1.0 - rr))))
        synthetic_memory_size = max(base_memory_size, int(synthetic_memory_size))

        synthetic_buffer = self._get_synthetic_rollout_replay_buffer(
            group=group, memory_size=synthetic_memory_size, transforms=transforms
        )

        # Keep a handle to the synthetic buffer so `process_batch()` can fill it.
        if not hasattr(self, "_synthetic_replay_buffers"):
            self._synthetic_replay_buffers = {}
        self._synthetic_replay_buffers[group] = synthetic_buffer

        return _MixedReplayBuffer(
            real=real_buffer,
            synthetic=synthetic_buffer,
            get_real_ratio=lambda: getattr(self, "real_ratio", 1.0),
            get_syn_ratio=lambda: getattr(self, "syn_ratio", None),
            on_policy=getattr(self, "on_policy", False),
        )

    def replay_buffer_memory_size_multiplier(self) -> float:
        # Do NOT inflate the main replay buffer: synthetic rollouts are stored separately.
        return 1.0

    def train_minibatch_multiplier(self, group: str) -> int:
        """How many optimizer minibatches to run per *base* minibatch.

        This is used to avoid returning over-sized concatenated minibatches when `syn_ratio` is large.
        Instead, we run more optimizer minibatches of bounded size.
        
        CRITICAL: When world model is skipped (threshold=0.0), return 1 to match baseline behavior.
        """
        # Check if world model is skipped - if so, return baseline value to avoid RNG differences
        unc_threshold = self._get_rollout_uncertainty_threshold(group)
        if unc_threshold is not None and unc_threshold == 0.0:
            if not getattr(self, "use_oracle", False):
                return 1
        
        sr = getattr(self, "syn_ratio", None)
        if sr is None:
            return 1
        rr = float(getattr(self, "real_ratio", 1.0) or 0.0)
        sr = float(sr or 0.0)
        rr = max(0.0, rr)
        sr = max(0.0, sr)
        # For on-policy algorithms (MBPO-MAPPO), ensure we always run enough minibatches
        # to fully cover the real batch once per PPO epoch, and then add synthetic minibatches.
        if getattr(self, "on_policy", False):
            rr = max(1.0, rr)
        return max(1, int(math.ceil(rr + sr)))

    def n_optimizer_steps_multiplier(self, group: str) -> float:
        """Multiplier for the number of optimizer steps to maintain constant sample coverage.

        When syn_ratio > 0, the total number of available samples per iteration increases
        (real + synthetic). To ensure each sample is seen the same number of times as in
        the base case (syn_ratio=0), we scale the number of optimizer steps proportionally.

        Returns:
            float: Multiplier to apply to off_policy_n_optimizer_steps. Returns 1.0 if
                   syn_ratio is None (old semantics) or 0.0, since in those cases the
                   total sample count per iteration doesn't increase.
        
        CRITICAL: When world model is skipped (threshold=0.0), return 1.0 to match baseline behavior.
        """
        # On-policy PPO-style training already defines "how many times each sample is used"
        # via `on_policy_n_minibatch_iters` (epochs over the batch). When we add synthetic
        # samples, we increase the number of minibatches per epoch (see `train_minibatch_multiplier`)
        # rather than increasing the number of PPO epochs, so each sample's reuse remains unchanged.
        if getattr(self, "on_policy", False):
            return 1.0

        # Check if world model is skipped - if so, return baseline value to avoid RNG differences
        unc_threshold = self._get_rollout_uncertainty_threshold(group)
        if unc_threshold is not None and unc_threshold == 0.0:
            if not getattr(self, "use_oracle", False):
                return 1.0
        
        sr = getattr(self, "syn_ratio", None)
        if sr is None:
            # Old semantics: syn_ratio is None, samples are mixed (not added)
            # Total effective samples per iteration doesn't increase, so no scaling needed
            return 1.0
        sr = float(sr or 0.0)
        sr = max(0.0, sr)
        if sr <= 0.0:
            return 1.0
        # New semantics: syn_ratio is a multiplier
        # We generate approximately syn_ratio * N_real synthetic samples per iteration
        # Total samples = N_real + syn_ratio * N_real = (1 + syn_ratio) * N_real
        # To keep the same number of times each sample is seen, scale optimizer steps by (1 + syn_ratio)
        return 1.0 + sr

    def _get_dynamics_training_buffer(self, memory_size: int) -> _DynamicsTrainBuffer:
        return _DynamicsTrainBuffer(capacity=memory_size, device=torch.device(self.device))

    def _get_synthetic_rollout_replay_buffer(
        self, group: str, memory_size: int, transforms: Optional[List[Transform]] = None
    ) -> ReplayBuffer:
        """Create a dedicated replay buffer for synthetic world-model rollouts."""
        memory_size = max(1, int(memory_size))

        sampling_size = self.experiment_config.train_minibatch_size(self.on_policy)
        if self.has_rnn:
            # RNN synthetic mixing isn't supported yet (matches existing code behaviour).
            sampling_size = 1

        sampler = SamplerWithoutReplacement() if self.on_policy else RandomSampler()

        if self.buffer_device == "disk" and not self.on_policy:
            storage = LazyMemmapStorage(
                memory_size,
                device=self.device,
                scratch_dir=self.experiment.folder_name / f"synthetic_buffer_{group}",
            )
        else:
            storage = LazyTensorStorage(
                memory_size,
                device=self.device if self.on_policy else self.buffer_device,
            )

        return TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=sampling_size,
            priority_key=(group, "td_error"),
            transform=Compose(*transforms) if transforms is not None else None,
        )

    def _ensure_synthetic_replay_buffers(self) -> None:
        # Kept for backward compatibility with older in-progress branches, but buffer
        # creation is now handled by `get_replay_buffer()` to avoid touching Experiment.
        if not hasattr(self, "_synthetic_replay_buffers"):
            self._synthetic_replay_buffers: Dict[str, ReplayBuffer] = {}


    def _load_normalization_stats(self) -> None:
        """Load normalization statistics from JSON file for state and reward normalization."""
        if self.normalization_stats_json is None:
            return
        
        path = pathlib.Path(self.normalization_stats_json)
        if not path.exists():
            raise FileNotFoundError(
                f"Normalization stats file not found: {self.normalization_stats_json}"
            )
        
        with open(path, "r") as f:
            stats_data = json.load(f)
        
        groups_data = stats_data.get("groups", {})
        if not groups_data:
            raise ValueError(
                f"No groups found in normalization stats file: {self.normalization_stats_json}"
            )
        
        for group in self.group_map.keys():
            if group not in groups_data:
                raise KeyError(
                    f"Group '{group}' not found in normalization stats file. "
                    f"Available groups: {list(groups_data.keys())}"
                )
            
            group_stats = groups_data[group]
            
            # Load state normalization stats
            state_stats = group_stats.get("state", {})
            if "mean" not in state_stats or "std" not in state_stats:
                raise ValueError(
                    f"State normalization stats (mean/std) not found for group '{group}'"
                )
            state_mean = torch.tensor(
                state_stats["mean"], dtype=torch.float32, device=self.device
            )
            state_std = torch.tensor(
                state_stats["std"], dtype=torch.float32, device=self.device
            )
            # Ensure std is positive
            state_std = torch.clamp(state_std, min=1e-6)
            self._state_norm_mean[group] = state_mean
            self._state_norm_std[group] = state_std
            
            # Load reward normalization stats
            reward_stats = group_stats.get("reward", {})
            if "mean" not in reward_stats or "std" not in reward_stats:
                raise ValueError(
                    f"Reward normalization stats (mean/std) not found for group '{group}'"
                )
            reward_mean = torch.tensor(
                reward_stats["mean"], dtype=torch.float32, device=self.device
            )
            reward_std = torch.tensor(
                reward_stats["std"], dtype=torch.float32, device=self.device
            )
            # Ensure std is positive
            reward_std = torch.clamp(reward_std, min=1e-6)
            self._reward_norm_mean[group] = reward_mean
            self._reward_norm_std[group] = reward_std


    def _mbpo_init(
        self,
        *,
        rollout_horizon: int,
        # Optional schedule: linearly increase rollout horizon over training.
        # If any of these are None/0, we fall back to constant `rollout_horizon`.
        rollout_horizon_min: Optional[int] = None,
        rollout_horizon_max: Optional[int] = None,
        rollout_horizon_ramp_steps: int = 0,
        model_train_freq: int,
        ensemble_size: int,
        model_batch_size: int,
        real_ratio: float,
        temperature: float,
        model_lr: float,
        model_hidden_size: int,
        model_num_layers: int,
        # New semantics: optional multiplier for synthetic samples per minibatch.
        # If set, it overrides the old "real_ratio fraction" mixing behavior in sampling.
        syn_ratio: Optional[float] = None,
        reward_loss_coef: float = 1.0,
        logvar_loss_coef: float = 1.0,  # scaling coefficient for log-variance term in NLL loss
        state_normalize: bool = False,
        reward_normalize: bool = False,
        reward_norm_eps: float = 1e-6,
        normalization_stats_json: Optional[str] = None,
        separate_reward_net: bool = False,
        training_iterations: int = 0,
        centralized_dynamics: bool = False,
        stochastic_dynamics: bool = True,
        n_elites: Optional[int] = None,
        min_log_var: float = -10.0,
        max_log_var: float = -2.0,
        warmup_steps: int = 0,
        # Rollout uncertainty gating: if a threshold is provided, synthetic transitions with
        # uncertainty above the threshold will be marked invalid and excluded from the
        # synthetic replay buffer (and the rollout for that env will be terminated).
        rollout_uncertainty_threshold: Optional[float] = None,
        # Optionally provide per-group thresholds (overrides the scalar threshold when present).
        # Example Hydra override: ++algorithm.rollout_uncertainty_thresholds.'agents'=0.25
        rollout_uncertainty_thresholds: Optional[Dict[str, float]] = None,
        # Which uncertainty metric to threshold on. Must be one of:
        # 'epistemic_obs_var', 'epistemic_rew_std', 'aleatoric_obs_logvar', 'aleatoric_rew_logvar',
        # 'total_obs_unc', 'total_rew_unc'
        rollout_uncertainty_metric: str = "total_rew_unc",
        # Use oracle (simulator) instead of world model for synthetic rollouts.
        # When enabled, the environment is stepped directly to get ground truth next states and rewards.
        # This is useful for sanity checking the MBPO algorithm without model errors.
        use_oracle: bool = False,
        load_world_model_path: Optional[str] = None,
        load_world_model_strict: bool = True,
        save_world_model_path: Optional[str] = None,
        save_world_model_interval: Optional[int] = None,
        # Oracle reward from predicted next-state (used for synthetic rollouts).
        # If enabled, synthetic rewards come from a deterministic function of the *predicted* state
        # rather than the learned reward head.
        oracle_reward: bool = False,
        oracle_reward_mode: str = "goal_distance_from_obs",
        # For mode=goal_distance_from_obs: indices (len=2) in the per-agent observation vector
        # that represent the goal-relative position (dx, dy). Reward is -||[dx,dy]||.
        oracle_reward_goal_rel_indices: Optional[List[int]] = None,
        oracle_reward_scale: float = 1.0,
        # If True, do not train the reward head (loss term set to 0) since reward is computed by oracle.
        oracle_reward_disable_reward_head_loss: bool = True,
        world_model_debug_video: bool = False,
        world_model_debug_video_interval: int = 200,
        world_model_debug_video_horizon: int = 50,
        world_model_debug_video_fps: int = 20,
        world_model_debug_video_env_index: int = 0,
        world_model_debug_video_obs_xy_indices: Optional[List[int]] = None,
        world_model_debug_video_mode: str = "open_loop",
    ) -> None:
        self.rollout_horizon = int(rollout_horizon)
        self.rollout_horizon_min = (
            None if rollout_horizon_min is None else int(rollout_horizon_min)
        )
        self.rollout_horizon_max = (
            None if rollout_horizon_max is None else int(rollout_horizon_max)
        )
        self.rollout_horizon_ramp_steps = int(rollout_horizon_ramp_steps or 0)
        self.model_train_freq = model_train_freq
        self.ensemble_size = ensemble_size
        self.model_batch_size = model_batch_size
        self.real_ratio = real_ratio
        self.syn_ratio = syn_ratio
        self.temperature = max(temperature, 1e-3)
        self.model_lr = model_lr
        self.model_hidden_size = model_hidden_size
        self.model_num_layers = model_num_layers
        self.reward_loss_coef = float(reward_loss_coef)
        self.logvar_loss_coef = float(logvar_loss_coef)
        self.state_normalize = bool(state_normalize)
        self.reward_normalize = bool(reward_normalize)
        self.reward_norm_eps = float(reward_norm_eps)
        # Normalize normalization_stats_json: handle string "null" from Hydra CLI
        if normalization_stats_json is not None:
            if isinstance(normalization_stats_json, str):
                normalization_stats_json_normalized = normalization_stats_json.strip().lower()
                if normalization_stats_json_normalized in ("null", "none", ""):
                    normalization_stats_json = None
        self.normalization_stats_json = normalization_stats_json
        # Require normalization_stats_json if normalization is enabled
        if (self.state_normalize or self.reward_normalize) and self.normalization_stats_json is None:
            raise ValueError(
                "normalization_stats_json must be provided when state_normalize or reward_normalize is True"
            )
        self.separate_reward_net = bool(separate_reward_net)
        self.training_iterations = max(0, int(training_iterations))
        self.centralized_dynamics = centralized_dynamics
        self.stochastic_dynamics = stochastic_dynamics
        self.n_elites = n_elites
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.warmup_steps = warmup_steps
        self.rollout_uncertainty_threshold = (
            None
            if rollout_uncertainty_threshold is None
            else float(rollout_uncertainty_threshold)
        )
        self.rollout_uncertainty_thresholds = rollout_uncertainty_thresholds
        self.rollout_uncertainty_metric = str(rollout_uncertainty_metric)
        self.use_oracle = bool(use_oracle)
        self.save_world_model_path = save_world_model_path
        self.save_world_model_interval = save_world_model_interval
        self.oracle_reward = bool(oracle_reward)
        self.oracle_reward_mode = str(oracle_reward_mode)
        self.oracle_reward_goal_rel_indices = oracle_reward_goal_rel_indices
        self.oracle_reward_scale = float(oracle_reward_scale)
        self.oracle_reward_disable_reward_head_loss = bool(
            oracle_reward_disable_reward_head_loss
        )
        self.world_model_debug_video = bool(world_model_debug_video)
        self.world_model_debug_video_interval = int(world_model_debug_video_interval)
        self.world_model_debug_video_horizon = int(world_model_debug_video_horizon)
        self.world_model_debug_video_fps = int(world_model_debug_video_fps)
        self.world_model_debug_video_env_index = int(world_model_debug_video_env_index)
        self.world_model_debug_video_obs_xy_indices = world_model_debug_video_obs_xy_indices
        self.world_model_debug_video_mode = str(world_model_debug_video_mode)
        # Normalize load_world_model_path: handle string "null" from Hydra CLI

        if load_world_model_path is not None:
            if isinstance(load_world_model_path, str):
                load_world_model_path_normalized = load_world_model_path.strip().lower()
                if load_world_model_path_normalized in ("null", "none", ""):
                    load_world_model_path = None
        self._dynamics: Dict[str, List[_DynamicsModel]] = {}
        self._dyn_optimizers: Dict[str, List[torch.optim.Optimizer]] = {}
        self._model_losses: Dict[str, torch.Tensor] = {}
        self._train_steps: Dict[str, int] = {}
        # Check if we should skip world model initialization entirely (e.g., threshold=0.0)
        # This avoids creating world model infrastructure when it won't be used
        skip_world_model_init = False
        # Check global threshold first (per-group thresholds checked later in get_replay_buffer)
        if self.rollout_uncertainty_threshold is not None and self.rollout_uncertainty_threshold == 0.0:
            if not getattr(self, "use_oracle", False):
                skip_world_model_init = True

        # Initialize train_steps for all groups (needed even if world model is skipped)
        for group in self.group_map.keys():
            self._train_steps[group] = 0

        self._elite_indices: Dict[str, torch.Tensor] = {}
        self._dynamics_train_buffers: Dict[str, _DynamicsTrainBuffer] = {}
        # Pre-computed normalization stats from JSON file (per group). Stored on device.
        self._state_norm_mean: Dict[str, torch.Tensor] = {}
        self._state_norm_std: Dict[str, torch.Tensor] = {}
        self._reward_norm_mean: Dict[str, torch.Tensor] = {}
        self._reward_norm_std: Dict[str, torch.Tensor] = {}
        
        # Load normalization stats from JSON file if normalization is enabled
        if self.normalization_stats_json is not None and (self.state_normalize or self.reward_normalize):
            self._load_normalization_stats()
        # Only initialize world model infrastructure if we'll actually use it
        if not skip_world_model_init:
            for group in self.group_map.keys():
                obs_shape = self.observation_spec[group, "observation"].shape
                n_agents = obs_shape[0]
                obs_dim = int(math.prod(obs_shape[1:]))

                action_space = self.action_spec[group, "action"]
                if hasattr(action_space, "space"):
                    if hasattr(action_space.space, "n"):
                        action_dim = action_space.space.n
                    else:
                        action_dim = int(math.prod(action_space.shape[1:]))
                else:
                    action_dim = int(math.prod(action_space.shape[1:]))

                # Per-agent vs centralized world model sizes
                if self.centralized_dynamics:
                    dyn_input_dim = n_agents * (obs_dim + action_dim)
                    next_obs_dim_out = n_agents * obs_dim
                    reward_dim_out = n_agents  # per-agent reward (flattened)
                    done_dim_out = n_agents  # per-agent done logits (flattened)
                else:
                    dyn_input_dim = obs_dim + action_dim
                    next_obs_dim_out = obs_dim
                    reward_dim_out = 1
                    done_dim_out = 1

                models: List[_DynamicsModel] = []
                opts: List[torch.optim.Optimizer] = []
                for _ in range(self.ensemble_size):
                    model = _DynamicsModel(
                        input_dim=dyn_input_dim,
                        next_obs_dim=next_obs_dim_out,
                        reward_dim=reward_dim_out,
                        done_dim=done_dim_out,
                        hidden_size=self.model_hidden_size,
                        num_layers=self.model_num_layers,
                        stochastic=self.stochastic_dynamics,
                        min_log_var=self.min_log_var,
                        max_log_var=self.max_log_var,
                        separate_reward_net=getattr(self, "separate_reward_net", False),
                    ).to(self.device)
                    models.append(model)
                    opts.append(torch.optim.Adam(model.parameters(), lr=self.model_lr))

                self._dynamics[group] = models
                self._dyn_optimizers[group] = opts
                self._model_losses[group] = torch.zeros(
                    self.ensemble_size, device=self.device
                )
                self._train_steps[group] = 0
                self._elite_indices[group] = torch.arange(
                    min(self.ensemble_size, self.n_elites or self.ensemble_size),
                    device=self.device,
                )

                # Dedicated dynamics training buffer (keeps history across iterations).
                # Size: a few iterations worth of on-policy data.
                # We avoid exposing a new hyperparam for now; adjust multiplier if needed.
                base_mem = int(self.experiment_config.replay_buffer_memory_size(self.on_policy))
                dyn_mem = max(base_mem * 10, int(self.model_batch_size) * 10)
                self._dynamics_train_buffers[group] = self._get_dynamics_training_buffer(
                    memory_size=dyn_mem
                )

        
        # Load pretrained world model if path is provided (only if we'll use it)

        if load_world_model_path is not None and not skip_world_model_init:
            self.load_world_model(load_world_model_path, strict=load_world_model_strict)
        
        # Track last save step for interval-based saving
        self._last_save_step: Dict[str, int] = {g: 0 for g in self.group_map.keys()}
        # Use a dedicated RNG for world-model training so we don't perturb the policy/replay sampling RNG.
        # This is critical for on-policy algorithms: consuming global RNG here can change minibatch order.
        # Only create RNG if we'll use the world model
        if not skip_world_model_init:
            seed = int(getattr(self.experiment, "seed", 0))
            self._world_model_rng = torch.Generator(device="cpu")
            self._world_model_rng.manual_seed(seed + 10_000_019)

    def _current_rollout_horizon(self, group: str) -> int:
        """Current synthetic rollout horizon.

        If scheduling is enabled, linearly ramps horizon from `rollout_horizon_min` to
        `rollout_horizon_max` over `rollout_horizon_ramp_steps` world-model training steps.
        """
        base = max(1, int(getattr(self, "rollout_horizon", 1)))
        h_min = getattr(self, "rollout_horizon_min", None)
        h_max = getattr(self, "rollout_horizon_max", None)
        ramp = int(getattr(self, "rollout_horizon_ramp_steps", 0) or 0)

        # Backward-compatible: no schedule.
        if h_min is None and h_max is None:
            return base
        if ramp <= 0:
            # No ramp => use max if provided, else min/base.
            if h_max is not None:
                return max(1, int(h_max))
            if h_min is not None:
                return max(1, int(h_min))
            return base

        h0 = max(1, int(h_min if h_min is not None else base))
        h1 = max(1, int(h_max if h_max is not None else base))

        step = int(getattr(self, "_train_steps", {}).get(group, 0))
        # Warmup: before any training steps, keep horizon at h0.
        if step <= 0:
            return h0
        frac = min(1.0, max(0.0, float(step) / float(ramp)))
        h = int(round(h0 + (h1 - h0) * frac))
        return max(1, h)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        # Check if we should skip world model operations entirely (e.g., threshold=0.0)
        # Do this first to avoid any world model overhead
        skip_world_model = False
        unc_threshold = self._get_rollout_uncertainty_threshold(group)
        if unc_threshold is not None and unc_threshold == 0.0:
            # With threshold=0.0, only samples with exactly 0 uncertainty will pass
            # This is extremely rare, so we skip world model operations to match baseline
            if not getattr(self, "use_oracle", False):
                skip_world_model = True
        
        # Ensure the synthetic buffer exists (created via get_replay_buffer()).
        # We don't create buffers here to avoid touching Experiment setup logic.
        # Skip if world model is disabled to avoid any overhead
        if not skip_world_model:
            synth_buf = getattr(self, "_synthetic_replay_buffers", {}).get(group, None)
            if synth_buf is not None and hasattr(synth_buf, "empty"):
                # Synthetic rollouts are per-iteration; clear at the start of the step.
                synth_buf.empty()
        batch = super().process_batch(group, batch)
        
        # Skip all world model operations if disabled
        if skip_world_model:
            return batch
        
        flat_batch = batch.reshape(-1) if not self.has_rnn else batch

        # Store transitions for dynamics training (do not clear between iterations).
        # This reduces overfitting to a single on-policy batch.
        dyn_buf = getattr(self, "_dynamics_train_buffers", {}).get(group, None)
        if dyn_buf is not None and flat_batch.numel() > 0:
            try:
                dyn_buf.extend(flat_batch)
            except Exception:
                pass

        self._train_steps[group] += 1
        
        if self._train_steps[group] % self.model_train_freq == 0 and not skip_world_model:
            # Train the dynamics model for multiple iterations, sampling a fresh minibatch each time.
            dyn_buf = getattr(self, "_dynamics_train_buffers", {}).get(group, None)
            for _ in range(int(getattr(self, "training_iterations", 0))):
                if dyn_buf is not None and len(dyn_buf) > 0:
                    try:
                        dyn_batch = dyn_buf.sample(
                            int(getattr(self, "model_batch_size", 256)),
                            generator=getattr(self, "_world_model_rng", None),
                        )
                    except Exception:
                        dyn_batch = flat_batch
                else:
                    dyn_batch = flat_batch
                self._train_dynamics(group, dyn_batch)

            # Optional debug video: compare predicted vs simulator transitions (and rewards) on a real rollout.
            # This logs a video where each frame includes both the state-change comparison and reward plot.
            try:
                self._maybe_log_world_model_debug_video(group=group, batch=batch)
            except Exception:
                pass

            # Auto-save world model if path is specified and interval condition is met
            # Save once per training step (check all groups, use minimum step)
            if self.save_world_model_path is not None:
                min_train_step = min(self._train_steps.values())
                should_save = False
                if self.save_world_model_interval is None:
                    # Save after each training step if no interval specified
                    should_save = True
                else:
                    # Save at specified intervals (check if any group has reached the interval)
                    min_last_save = min(self._last_save_step.values())
                    steps_since_last_save = min_train_step - min_last_save
                    if steps_since_last_save >= self.save_world_model_interval:
                        should_save = True
                
                if should_save:
                    self.save_world_model(self.save_world_model_path)
                    # Update all groups' last save step
                    for g in self._last_save_step.keys():
                        self._last_save_step[g] = self._train_steps[g]

        if self._train_steps[group] > self.warmup_steps and not skip_world_model:
            synthetic = self._generate_model_rollouts(group, flat_batch)
            if synthetic is not None and synthetic.numel() > 0:
                # Optional lightweight debug: log how many synthetic transitions were generated
                # (helps confirm synthetic generation is actually happening).
                try:
                    if hasattr(self.experiment, "logger") and self.experiment.logger is not None:
                        step = getattr(self.experiment, "total_frames", self._train_steps[group])
                        self.experiment.logger.log(
                            {
                                f"world_model/{group}/synthetic/num_transitions": float(
                                    int(synthetic.numel())
                                )
                            },
                            step=step,
                        )
                except Exception:
                    pass
                synthetic = super().process_batch(group, synthetic)
                use_synth = False
                if not self.has_rnn:
                    synthetic = synthetic.reshape(-1)
                    # Drop invalid synthetic transitions (e.g., high-uncertainty rollouts).
                    try:
                        if "synthetic_valid" in synthetic.keys(True, True):
                            v = synthetic.get("synthetic_valid").to(torch.bool).reshape(-1)
                            use_synth = v.any()
                            n_before = int(v.numel())
                            n_keep = int(v.sum().item())
                            n_drop = int(n_before - n_keep)
                            if n_drop > 0:
                                synthetic = synthetic[v]
                                try:
                                    if hasattr(self.experiment, "logger") and self.experiment.logger is not None:
                                        step = getattr(self.experiment, "total_frames", self._train_steps[group])
                                        self.experiment.logger.log(
                                            {
                                                f"world_model/{group}/synthetic/dropped_transitions": float(n_drop),
                                                f"world_model/{group}/synthetic/kept_transitions": float(n_keep),
                                            },
                                            step=step,
                                        )
                                except Exception:
                                    pass
                        else:
                            use_synth = True
                    except Exception:
                        pass
                    # Store synthetic rollouts in the dedicated buffer.
                    synth_buf = getattr(self, "_synthetic_replay_buffers", {}).get(
                        group, None
                    )
                    if synth_buf is not None and use_synth:
                        synth_buf.extend(synthetic.to(synth_buf.storage.device))
                else:
                    # RNN batches have sequence structure; mixing is not supported yet.
                    pass

        return batch

    #####################
    # Dynamics utilities
    #####################

    def _encode_action(self, group: str, action: torch.Tensor) -> torch.Tensor:
        if action.dtype in (torch.int, torch.long):
            n_actions = self.action_spec[group, "action"].space.n
            # torchrl often stores discrete actions with a trailing singleton dim (..., 1)
            if action.shape[-1] == 1:
                action = action.squeeze(-1)
            action = F.one_hot(action.to(torch.long), num_classes=n_actions).to(
                torch.float32
            )
        return action

    def _oracle_reward_enabled(self) -> bool:
        return bool(getattr(self, "oracle_reward", False))

    def _oracle_reward_from_predicted_next_obs(
        self, group: str, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute oracle reward from predicted next observation.

        Currently implemented mode:
        - goal_distance_from_obs: reward = -||goal_rel_xy|| (per agent), where goal_rel_xy
          comes from `oracle_reward_goal_rel_indices` in the observation vector.
        """
        mode = str(getattr(self, "oracle_reward_mode", "goal_distance_from_obs") or "").strip().lower()
        if mode != "goal_distance_from_obs":
            raise ValueError(
                f"Unknown oracle_reward_mode={mode}. Currently supported: goal_distance_from_obs"
            )
        idx = getattr(self, "oracle_reward_goal_rel_indices", None)
        if idx is None or not isinstance(idx, (list, tuple)) or len(idx) != 2:
            raise ValueError(
                "oracle_reward_goal_rel_indices must be a list/tuple of two ints when using "
                "oracle_reward_mode=goal_distance_from_obs"
            )
        i0, i1 = int(idx[0]), int(idx[1])
        if next_obs.shape[-1] <= max(i0, i1):
            raise ValueError(
                f"oracle_reward_goal_rel_indices out of bounds for group '{group}': "
                f"max(idx)={max(i0,i1)} but obs_dim={next_obs.shape[-1]}"
            )
        rel = torch.stack([next_obs[..., i0], next_obs[..., i1]], dim=-1).to(torch.float32)
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)  # [..., 1]
        scale = float(getattr(self, "oracle_reward_scale", 1.0))
        return (-dist * scale).to(next_obs.device)

    @torch.no_grad()
    def _world_model_eval_losses(
        self, group: str, flat_batch: TensorDictBase
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute per-model evaluation losses on a batch WITHOUT bootstrap sampling.

        Returns:
            - losses_per_model: tensor of shape [ensemble_size] (total loss)
            - metrics: dict with averaged components (obs_nll, rew_nll, done_bce, obs_mse, rew_mse)

        Notes:
            - This does not update model parameters.
            - This does not update running reward normalization stats.
        """
        keys = flat_batch.keys(True, True)
        if (group, "observation") not in keys or ("next", group, "observation") not in keys:
            return torch.full((self.ensemble_size,), float("nan"), device=self.device), {}

        obs = flat_batch.get((group, "observation")).to(self.device)
        next_obs = flat_batch.get(("next", group, "observation")).to(self.device)
        action = flat_batch.get((group, "action")).to(self.device)
        reward = flat_batch.get(("next", group, "reward")).to(self.device)
        done = flat_batch.get(("next", group, "done")).to(self.device)

        obs = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
        next_obs = next_obs.reshape(-1, next_obs.shape[-2], next_obs.shape[-1])
        action = action.reshape(-1, action.shape[-2], *action.shape[-1:])
        reward = reward.reshape(-1, reward.shape[-2], reward.shape[-1])
        done = done.reshape(-1, done.shape[-2], done.shape[-1])

        action = self._encode_action(group, action)

        if self.centralized_dynamics:
            obs_flat = obs.reshape(obs.shape[0], -1)
            next_obs_flat = next_obs.reshape(next_obs.shape[0], -1)
            action_flat = action.reshape(action.shape[0], -1)
            reward_flat = reward.squeeze(-1)
            done_flat = done.squeeze(-1)
        else:
            obs_flat = obs.flatten(0, 1)
            next_obs_flat = next_obs.flatten(0, 1)
            action_flat = action.flatten(0, 1)
            reward_flat = reward.flatten(0, 1)
            done_flat = done.flatten(0, 1)

        losses = []
        total_loss_obs = 0.0
        total_loss_rew = 0.0
        total_loss_done = 0.0
        total_mse_obs = 0.0
        total_mse_rew = 0.0
        total_smape_obs = 0.0
        total_smape_rew = 0.0
        total_r2_rew = 0.0
        total_r2_obs = 0.0

        # Normalize observations/states if enabled
        if getattr(self, "state_normalize", False) and group in self._state_norm_mean:
            obs_mean = self._state_norm_mean[group]
            obs_std = self._state_norm_std[group]
            eps = float(getattr(self, "reward_norm_eps", 1e-6))
            # Normalize observations and next observations
            # obs_mean and obs_std should have shape [obs_dim] for per-agent or [n_agents*obs_dim] for centralized
            # obs_flat has shape [B*n_agents, obs_dim] for per-agent or [B, n_agents*obs_dim] for centralized
            # Ensure the last dimension matches exactly
            if obs_flat.shape[-1] != obs_mean.shape[-1]:
                raise ValueError(
                    f"Observation dimension mismatch for group '{group}': "
                    f"obs_flat.shape[-1]={obs_flat.shape[-1]}, "
                    f"norm_mean.shape[-1]={obs_mean.shape[-1]}"
                )
            obs_flat = (obs_flat - obs_mean) / (obs_std + eps)
            next_obs_flat = (next_obs_flat - obs_mean) / (obs_std + eps)
        
        # Reward normalization uses pre-loaded stats if enabled
        use_rn = bool(getattr(self, "reward_normalize", False))
        if use_rn and group in self._reward_norm_mean:
            eps = float(getattr(self, "reward_norm_eps", 1e-6))
            r_mean = self._reward_norm_mean[group]
            r_std = self._reward_norm_std[group] + eps

        logvar_coef = float(getattr(self, "logvar_loss_coef", 1.0))
        for model in self._dynamics[group]:
            mu_next, lv_delta, mu_rew, lv_rew, done_logit = model(obs_flat, action_flat)
            inv_var_obs = torch.exp(-lv_delta)
            inv_var_rew = torch.exp(-lv_rew)
            loss_obs = torch.mean((mu_next - next_obs_flat) ** 2 * inv_var_obs + logvar_coef * lv_delta)

            # R^2 diagnostics (scale-free): compare to predicting the batch mean.
            # These are computed on raw (non-normalized) targets.
            rew_target = reward_flat.to(torch.float32)
            rew_pred = mu_rew.to(torch.float32)
            mse_rew = torch.mean((rew_pred - rew_target) ** 2)
            var_rew = torch.mean((rew_target - rew_target.mean()) ** 2)
            r2_rew = 1.0 - (mse_rew / (var_rew + 1e-12))

            obs_target = next_obs_flat.to(torch.float32)
            obs_pred = mu_next.to(torch.float32)
            mse_obs = torch.mean((obs_pred - obs_target) ** 2)
            var_obs = torch.mean((obs_target - obs_target.mean()) ** 2)
            r2_obs = 1.0 - (mse_obs / (var_obs + 1e-12))

            # Symmetric MAPE (sMAPE): 200 * |pred-actual| / (|pred| + |actual| + eps), bounded in [0, 200].
            smape_eps = 1e-6
            smape_rew = (
                200.0
                * (rew_pred - rew_target).abs()
                / (rew_pred.abs() + rew_target.abs() + smape_eps)
            )
            smape_obs = (
                200.0
                * (obs_pred - obs_target).abs()
                / (obs_pred.abs() + obs_target.abs() + smape_eps)
            )

            if use_rn:
                # Handle shape: reward stats may be per-agent or scalar
                if r_mean.dim() == 0:
                    # Scalar: broadcast to match reward shape
                    r_n = (reward_flat - r_mean) / r_std
                    mu_rew_n = (mu_rew - r_mean) / r_std
                else:
                    # Per-agent: ensure shapes match
                    if reward_flat.shape[-1] != r_mean.shape[-1]:
                        raise ValueError(
                            f"Reward dimension mismatch for group '{group}': "
                            f"reward_flat.shape[-1]={reward_flat.shape[-1]}, "
                            f"norm_mean.shape[-1]={r_mean.shape[-1]}"
                        )
                    r_n = (reward_flat - r_mean) / r_std
                    mu_rew_n = (mu_rew - r_mean) / r_std
                lv_rew_n = lv_rew - 2.0 * torch.log(r_std)
                inv_var_rew_n = torch.exp(-lv_rew_n)
                loss_rew = torch.mean((mu_rew_n - r_n) ** 2 * inv_var_rew_n + logvar_coef * lv_rew_n)
            else:
                loss_rew = torch.mean((mu_rew - reward_flat) ** 2 * inv_var_rew + logvar_coef * lv_rew)
            loss_rew = loss_rew * float(getattr(self, "reward_loss_coef", 1.0))
            loss_rew = loss_rew * float(getattr(self, "reward_loss_coef", 1.0))
            loss_done = F.binary_cross_entropy_with_logits(done_logit, done_flat.float())
            loss = loss_obs + loss_rew + loss_done
            losses.append(loss)

            total_loss_obs += float(loss_obs.detach().cpu().item())
            total_loss_rew += float(loss_rew.detach().cpu().item())
            total_loss_done += float(loss_done.detach().cpu().item())
            total_mse_obs += float(mse_obs.detach().cpu().item())
            total_mse_rew += float(mse_rew.detach().cpu().item())
            total_smape_obs += float(smape_obs.mean().detach().cpu().item())
            total_smape_rew += float(smape_rew.mean().detach().cpu().item())
            total_r2_obs += float(r2_obs.detach().cpu().item())
            total_r2_rew += float(r2_rew.detach().cpu().item())

        losses_t = torch.stack(losses, dim=0).to(self.device)
        denom = max(1, len(self._dynamics[group]))
        metrics = {
            "eval/loss_observation": total_loss_obs / denom,
            "eval/loss_reward": total_loss_rew / denom,
            "eval/loss_done": total_loss_done / denom,
            "eval/mse_observation": total_mse_obs / denom,
            "eval/mse_reward": total_mse_rew / denom,
            "eval/smape_observation": total_smape_obs / denom,
            "eval/smape_reward": total_smape_rew / denom,
            "eval/r2_observation": total_r2_obs / denom,
            "eval/r2_reward": total_r2_rew / denom,
            "eval/loss_total": float(losses_t.mean().detach().cpu().item()),
        }
        return losses_t, metrics

    def _set_elites_from_losses(self, group: str, losses_per_model: torch.Tensor) -> None:
        """Update elite indices given per-model losses (lower is better)."""
        n_elites = self.n_elites or self.ensemble_size
        n_elites = max(1, min(int(n_elites), self.ensemble_size))
        # Ensure on device
        l = losses_per_model.to(self.device)
        self._elite_indices[group] = torch.argsort(l)[:n_elites]

    def _compute_per_sample_uncertainty(
        self, group: str, flat_batch: TensorDictBase
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample uncertainty values for validation threshold computation.
        
        Returns:
            Dictionary with per-sample uncertainty tensors:
            - 'epistemic_obs_var': [n_samples] - variance of ensemble means for observations
            - 'epistemic_rew_std': [n_samples] - std of ensemble means for rewards
            - 'aleatoric_obs_logvar': [n_samples] - mean predicted log-variance for observations
            - 'aleatoric_rew_logvar': [n_samples] - mean predicted log-variance for rewards
            - 'total_obs_unc': [n_samples] - combined uncertainty (epistemic + aleatoric) for obs
            - 'total_rew_unc': [n_samples] - combined uncertainty (epistemic + aleatoric) for rewards
        """
        keys = flat_batch.keys(True, True)
        if (group, "observation") not in keys:
            return {}
        
        obs = flat_batch.get((group, "observation")).to(self.device)
        action = flat_batch.get((group, "action")).to(self.device)
        
        obs = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
        action = action.reshape(-1, action.shape[-2], *action.shape[-1:])
        action = self._encode_action(group, action)
        
        if self.centralized_dynamics:
            obs_flat = obs.reshape(obs.shape[0], -1)
            action_flat = action.reshape(action.shape[0], -1)
        else:
            obs_flat = obs.flatten(0, 1)
            action_flat = action.flatten(0, 1)
        
        with torch.no_grad():
            mu_next_all = []
            mu_rew_all = []
            lv_delta_all = []
            lv_rew_all = []
            for model in self._dynamics[group]:
                mu_next, lv_delta, mu_rew, lv_rew, _ = model(obs_flat, action_flat)
                mu_next_all.append(mu_next)
                mu_rew_all.append(mu_rew)
                lv_delta_all.append(lv_delta)
                lv_rew_all.append(lv_rew)
            
            mu_next_all = torch.stack(mu_next_all, dim=0)  # [ensemble_size, n_samples, obs_dim]
            mu_rew_all = torch.stack(mu_rew_all, dim=0)  # [ensemble_size, n_samples, rew_dim]
            lv_delta_all = torch.stack(lv_delta_all, dim=0)  # [ensemble_size, n_samples, obs_dim]
            lv_rew_all = torch.stack(lv_rew_all, dim=0)  # [ensemble_size, n_samples, rew_dim]
            
            # Epistemic uncertainty: variance/std of ensemble means per sample
            epistemic_obs_var = torch.var(mu_next_all, dim=0)  # [n_samples, obs_dim]
            epistemic_obs_var = torch.mean(epistemic_obs_var, dim=-1)  # [n_samples] - mean across obs dims
            
            epistemic_rew_std = torch.std(mu_rew_all, dim=0)  # [n_samples, rew_dim]
            epistemic_rew_std = torch.mean(epistemic_rew_std, dim=-1)  # [n_samples] - mean across rew dims
            
            # Aleatoric uncertainty: mean predicted log-variance per sample
            aleatoric_obs_logvar = torch.mean(lv_delta_all, dim=0)  # [n_samples, obs_dim]
            aleatoric_obs_logvar = torch.mean(aleatoric_obs_logvar, dim=-1)  # [n_samples]
            
            aleatoric_rew_logvar = torch.mean(lv_rew_all, dim=0)  # [n_samples, rew_dim]
            aleatoric_rew_logvar = torch.mean(aleatoric_rew_logvar, dim=-1)  # [n_samples]
            
            # Combined uncertainty: epistemic + aleatoric
            # For aleatoric, convert log-var to std: std = exp(0.5 * log_var)
            aleatoric_obs_std = torch.exp(0.5 * aleatoric_obs_logvar)
            aleatoric_rew_std = torch.exp(0.5 * aleatoric_rew_logvar)
            
            # Total uncertainty: sqrt(epistemic_var + aleatoric_var) = sqrt(epistemic_var) + aleatoric_std (approximation)
            # Or more accurately: total_var = epistemic_var + aleatoric_var, total_std = sqrt(total_var)
            epistemic_obs_std = torch.sqrt(epistemic_obs_var.clamp_min(0.0))
            total_obs_unc = torch.sqrt(epistemic_obs_var + aleatoric_obs_std.pow(2).clamp_min(0.0))
            
            total_rew_unc = torch.sqrt(epistemic_rew_std.pow(2) + aleatoric_rew_std.pow(2).clamp_min(0.0))
            
            return {
                "epistemic_obs_var": epistemic_obs_var.detach().cpu(),
                "epistemic_rew_std": epistemic_rew_std.detach().cpu(),
                "aleatoric_obs_logvar": aleatoric_obs_logvar.detach().cpu(),
                "aleatoric_rew_logvar": aleatoric_rew_logvar.detach().cpu(),
                "total_obs_unc": total_obs_unc.detach().cpu(),
                "total_rew_unc": total_rew_unc.detach().cpu(),
            }

    def _train_dynamics(self, group: str, flat_batch: TensorDictBase) -> None:
        keys = flat_batch.keys(True, True)
        if (group, "observation") not in keys or ("next", group, "observation") not in keys:
            return

        obs = flat_batch.get((group, "observation")).to(self.device)
        next_obs = flat_batch.get(("next", group, "observation")).to(self.device)
        action = flat_batch.get((group, "action")).to(self.device)
        reward = flat_batch.get(("next", group, "reward")).to(self.device)
        done = flat_batch.get(("next", group, "done")).to(self.device)

        obs = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
        next_obs = next_obs.reshape(-1, next_obs.shape[-2], next_obs.shape[-1])
        action = action.reshape(-1, action.shape[-2], *action.shape[-1:])
        reward = reward.reshape(-1, reward.shape[-2], reward.shape[-1])
        done = done.reshape(-1, done.shape[-2], done.shape[-1])

        action = self._encode_action(group, action)

        if self.centralized_dynamics:
            # Flatten joint (all agents) into a single vector per timestep.
            # obs: [B, n_agents, obs_dim]
            # action: [B, n_agents, action_dim] (or one-hot)
            obs_flat = obs.reshape(obs.shape[0], -1)
            next_obs_flat = next_obs.reshape(next_obs.shape[0], -1)
            action_flat = action.reshape(action.shape[0], -1)
            # reward/done are per-agent scalars => flatten agents, keep per-step batch
            reward_flat = reward.squeeze(-1)
            done_flat = done.squeeze(-1)
        else:
            # Per-agent independent model: each agent is a training sample.
            obs_flat = obs.flatten(0, 1)
            next_obs_flat = next_obs.flatten(0, 1)
            action_flat = action.flatten(0, 1)
            reward_flat = reward.flatten(0, 1)
            done_flat = done.flatten(0, 1)

        # Normalize observations/states if enabled
        if getattr(self, "state_normalize", False) and group in self._state_norm_mean:
            obs_mean = self._state_norm_mean[group]
            obs_std = self._state_norm_std[group]
            eps = float(getattr(self, "reward_norm_eps", 1e-6))
            # Normalize observations and next observations
            # obs_mean and obs_std should have shape [obs_dim] for per-agent or [n_agents*obs_dim] for centralized
            # obs_flat has shape [B*n_agents, obs_dim] for per-agent or [B, n_agents*obs_dim] for centralized
            # Ensure the last dimension matches exactly
            if obs_flat.shape[-1] != obs_mean.shape[-1]:
                raise ValueError(
                    f"Observation dimension mismatch for group '{group}': "
                    f"obs_flat.shape[-1]={obs_flat.shape[-1]}, "
                    f"norm_mean.shape[-1]={obs_mean.shape[-1]}"
                )
            obs_flat = (obs_flat - obs_mean) / (obs_std + eps)
            next_obs_flat = (next_obs_flat - obs_mean) / (obs_std + eps)

        # Bootstrap sampling per model encourages ensemble diversity.
        n_samples = obs_flat.shape[0]

        # Track metrics for logging (computed on full batch without bootstrap for accuracy)
        total_loss_obs = 0.0
        total_loss_rew = 0.0
        total_loss_done = 0.0
        total_mse_obs = 0.0
        total_mse_rew = 0.0
        total_done_accuracy = 0.0

        for i, model in enumerate(self._dynamics[group]):
            optimizer = self._dyn_optimizers[group][i]
            optimizer.zero_grad()

            # Bootstrap indices (with replacement)
            # IMPORTANT: use a dedicated RNG so world-model training does not perturb
            # the global RNG used by on-policy sampling / policy stochasticity.
            idx = torch.randint(
                0,
                n_samples,
                (n_samples,),
                device="cpu",
                generator=getattr(self, "_world_model_rng", None),
            ).to(obs_flat.device)
            o = obs_flat[idx]
            a = action_flat[idx]
            no = next_obs_flat[idx]
            r = reward_flat[idx]
            d = done_flat[idx]

            mu_next, lv_delta, mu_rew, lv_rew, done_logit = model(o, a)

            # Gaussian NLL-style losses (per-dim): (x-mu)^2 * exp(-log_var) + logvar_loss_coef * log_var
            # Scaling the log_var term prevents the model from exploiting high uncertainty to make loss negative
            inv_var_obs = torch.exp(-lv_delta)
            inv_var_rew = torch.exp(-lv_rew)
            logvar_coef = float(getattr(self, "logvar_loss_coef", 1.0))
            loss_obs = torch.mean((mu_next - no) ** 2 * inv_var_obs + logvar_coef * lv_delta)

            # Reward loss scaling / normalization:
            # When rewards have small magnitude, unweighted MSE/NLL often collapses toward ~0.
            # If enabled, we compute the reward loss in *normalized* space using pre-loaded mean/std.
            # This uses both mean and variance, making gradients invariant to reward offset/scale.
            if getattr(self, "reward_normalize", False) and group in self._reward_norm_mean:
                eps = float(getattr(self, "reward_norm_eps", 1e-6))
                r_mean = self._reward_norm_mean[group]
                r_std = self._reward_norm_std[group] + eps
                # Normalize targets and predictions.
                # Handle shape: reward stats may be per-agent or scalar
                if r_mean.dim() == 0:
                    # Scalar: broadcast to match reward shape
                    r_n = (r - r_mean) / r_std
                    mu_rew_n = (mu_rew - r_mean) / r_std
                else:
                    # Per-agent: ensure shapes match
                    if r.shape[-1] != r_mean.shape[-1]:
                        raise ValueError(
                            f"Reward dimension mismatch for group '{group}': "
                            f"r.shape[-1]={r.shape[-1]}, norm_mean.shape[-1]={r_mean.shape[-1]}"
                        )
                    r_n = (r - r_mean) / r_std
                    mu_rew_n = (mu_rew - r_mean) / r_std
                # If stochastic (log-variance provided), convert to normalized-space log-variance:
                # var_n = var / std^2  => log_var_n = log_var - 2*log(std)
                lv_rew_n = lv_rew - 2.0 * torch.log(r_std)
                inv_var_rew_n = torch.exp(-lv_rew_n)
                loss_rew = torch.mean((mu_rew_n - r_n) ** 2 * inv_var_rew_n + logvar_coef * lv_rew_n)
            else:
                loss_rew = torch.mean((mu_rew - r) ** 2 * inv_var_rew + logvar_coef * lv_rew)
                # loss_rew = torch.nn.functional.huber_loss(mu_rew, r)
            if self._oracle_reward_enabled() and bool(
                getattr(self, "oracle_reward_disable_reward_head_loss", True)
            ):
                loss_rew = loss_rew * 0.0
            else:
                loss_rew = loss_rew * float(getattr(self, "reward_loss_coef", 1.0))
            loss_done = F.binary_cross_entropy_with_logits(done_logit, d.float())
            loss = loss_obs + loss_rew + loss_done
            loss.backward()
            optimizer.step()
            self._model_losses[group][i] = loss.detach()

            # Accumulate metrics for logging (using bootstrap-sampled data)
            total_loss_obs += loss_obs.detach().item()
            total_loss_rew += loss_rew.detach().item()
            total_loss_done += loss_done.detach().item()
            total_mse_obs += torch.mean((mu_next - no) ** 2).detach().item()
            total_mse_rew += torch.mean((mu_rew - r) ** 2).detach().item()
            # Done accuracy: percentage of correct predictions
            done_pred = (torch.sigmoid(done_logit) > 0.5).float()
            done_accuracy = torch.mean((done_pred == d.float()).float()).detach().item()
            total_done_accuracy += done_accuracy

        # Uncertainty metrics on the full (non-bootstrapped) batch
        with torch.no_grad():
            def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> float:
                """Compute Pearson correlation between 1D tensors; returns 0.0 if ill-defined."""
                x = x.flatten().to(torch.float32)
                y = y.flatten().to(torch.float32)
                n = min(x.numel(), y.numel())
                if n <= 1:
                    return 0.0
                x = x[:n]
                y = y[:n]
                x = x - x.mean()
                y = y - y.mean()
                denom = (x.pow(2).mean().sqrt() * y.pow(2).mean().sqrt()).clamp_min(1e-12)
                return float((x.mul(y).mean() / denom).detach().cpu().item())

            mu_next_all = []
            mu_rew_all = []
            lv_delta_all = []
            lv_rew_all = []
            done_logit_all = []
            for model in self._dynamics[group]:
                mu_next, lv_delta, mu_rew, lv_rew, done_logit = model(
                    obs_flat, action_flat
                )
                mu_next_all.append(mu_next)
                mu_rew_all.append(mu_rew)
                lv_delta_all.append(lv_delta)
                lv_rew_all.append(lv_rew)
                done_logit_all.append(done_logit)

            mu_next_all = torch.stack(mu_next_all, dim=0)
            mu_rew_all = torch.stack(mu_rew_all, dim=0)
            lv_delta_all = torch.stack(lv_delta_all, dim=0)
            lv_rew_all = torch.stack(lv_rew_all, dim=0)
            done_logit_all = torch.stack(done_logit_all, dim=0)

            # Aleatoric uncertainty: mean predicted log-variance
            aleatoric_obs_logvar = lv_delta_all.mean().detach().item()
            aleatoric_rew_logvar = lv_rew_all.mean().detach().item()

            # Epistemic uncertainty: variance of ensemble means
            epistemic_obs_var = torch.var(mu_next_all, dim=0).mean().detach().item()
            epistemic_rew_var = torch.var(mu_rew_all, dim=0).mean().detach().item()
            epistemic_done_var = torch.var(done_logit_all, dim=0).mean().detach().item()

            # Model bias assessment: compare ensemble predictions with real rewards
            # mu_rew_all shape: [ensemble_size, n_samples, reward_dim]
            # Compute mean predicted reward across ensemble (per sample)
            mean_predicted_reward = mu_rew_all.mean(dim=0)  # [n_samples, reward_dim]
            # Compute std of predicted rewards across ensemble (epistemic uncertainty)
            std_predicted_reward = torch.std(mu_rew_all, dim=0)  # [n_samples, reward_dim]
            
            # Flatten to compare with actual rewards
            # For centralized: reward_flat is [B, n_agents], mu_rew_all is [ensemble_size, B, n_agents]
            # For per-agent: reward_flat is [B*n_agents, 1], mu_rew_all is [ensemble_size, B*n_agents, 1]
            mean_pred_flat = mean_predicted_reward.flatten()
            std_pred_flat = std_predicted_reward.flatten()
            reward_flat_for_bias = reward_flat.flatten()
            
            # Ensure same length (should already match, but be safe)
            min_len = min(len(mean_pred_flat), len(reward_flat_for_bias))
            mean_pred_flat = mean_pred_flat[:min_len]
            std_pred_flat = std_pred_flat[:min_len]
            reward_flat_for_bias = reward_flat_for_bias[:min_len]
            
            # Compute statistics
            mean_pred_reward = mean_pred_flat.mean().detach().item()
            std_pred_reward = std_pred_flat.mean().detach().item()  # Average std across samples
            mean_actual_reward = reward_flat_for_bias.mean().detach().item()
            std_actual_reward = reward_flat_for_bias.std().detach().item()
            
            # Compute bias: difference between predicted and actual mean
            reward_bias = mean_pred_reward - mean_actual_reward
            
            # Compute absolute error per sample, then mean
            abs_error = torch.abs(mean_pred_flat - reward_flat_for_bias)
            mean_abs_error = abs_error.mean().detach().item()
            
            # Percentage error (MAPE) per sample: |pred - actual| / |actual| * 100
            # Note: MAPE can exceed 100% (e.g., when |error| > |actual|) and is unstable near 0.
            denom = reward_flat_for_bias.abs()
            nonzero_mask = denom > 1e-6
            mape_mask_rate = 1.0 - float(nonzero_mask.float().mean().detach().cpu().item())
            if nonzero_mask.any():
                percentage_error = (abs_error[nonzero_mask] / denom[nonzero_mask]) * 100.0
                mean_percentage_error = percentage_error.mean().detach().item()
            else:
                mean_percentage_error = 0.0

            # Epsilon-stabilized "relative error %" (includes all samples, but still can be large).
            # Useful when rewards can be ~0 frequently.
            rel_eps = 1e-3
            mean_percentage_error_eps = (
                (abs_error / (denom + rel_eps)) * 100.0
            ).mean().detach().item()

            # Symmetric MAPE (sMAPE): 200 * |pred-actual| / (|pred| + |actual| + eps), bounded in [0, 200].
            smape_eps = 1e-6
            smape = (
                200.0
                * abs_error
                / (mean_pred_flat.abs() + reward_flat_for_bias.abs() + smape_eps)
            )
            mean_smape = smape.mean().detach().item()

            # Normalized MAE (%): mean(|error|) / (mean(|actual|) + eps) * 100 (much more stable than per-sample MAPE).
            nmae_eps = 1e-6
            mean_nmae_percent = (
                abs_error.mean() / (reward_flat_for_bias.abs().mean() + nmae_eps) * 100.0
            ).detach().item()

            # Expose current reward normalization stats when enabled (helps debugging).
            if getattr(self, "reward_normalize", False) and group in self._reward_norm_mean:
                reward_norm_mean = self._reward_norm_mean[group].detach().cpu()
                reward_norm_std = self._reward_norm_std[group].detach().cpu()
            else:
                reward_norm_mean = 0.0
                reward_norm_std = 0.0
            
            # Compute range metrics: mean ± std (averaged across samples)
            pred_reward_min = (mean_pred_flat - std_pred_flat).mean().detach().item()
            pred_reward_max = (mean_pred_flat + std_pred_flat).mean().detach().item()

            # --- Optimism diagnostics for reward predictions ---
            # Signed error per sample
            signed_error = mean_pred_flat - reward_flat_for_bias
            over_mask = signed_error > 0
            under_mask = signed_error < 0

            over_rate = over_mask.float().mean().detach().item()
            under_rate = under_mask.float().mean().detach().item()

            mean_overpred = torch.clamp(signed_error, min=0.0).mean().detach().item()
            mean_underpred = torch.clamp(-signed_error, min=0.0).mean().detach().item()

            # Lower-confidence-bound optimism: even (pred - std) is above actual.
            lcb = mean_pred_flat - std_pred_flat
            lcb_over_rate = (lcb > reward_flat_for_bias).float().mean().detach().item()
            lcb_over_mean = torch.clamp(lcb - reward_flat_for_bias, min=0.0).mean().detach().item()

            # Tail metrics: focus on high predicted reward and high uncertainty regions.
            n = mean_pred_flat.numel()
            k = max(1, int(0.1 * n))  # top 10%

            top_pred_idx = torch.topk(mean_pred_flat, k=k, largest=True).indices
            top_pred_bias = signed_error[top_pred_idx].mean().detach().item()
            top_pred_over_rate = over_mask[top_pred_idx].float().mean().detach().item()

            top_unc_idx = torch.topk(std_pred_flat, k=k, largest=True).indices
            top_unc_bias = signed_error[top_unc_idx].mean().detach().item()
            top_unc_over_rate = over_mask[top_unc_idx].float().mean().detach().item()

            # --- Calibration / usefulness of uncertainty ---
            # Reward: does epistemic uncertainty (std across ensemble) correlate with abs error?
            reward_unc = std_pred_flat
            reward_err = abs_error
            reward_unc_err_pearson = _pearsonr(reward_unc, reward_err)

            # Observation: uncertainty (var across ensemble) vs prediction error (MSE) per sample.
            obs_mean = mu_next_all.mean(dim=0)  # [n_samples, obs_dim]
            obs_err = torch.mean((obs_mean - next_obs_flat) ** 2, dim=-1)  # [n_samples]
            obs_unc = torch.mean(torch.var(mu_next_all, dim=0), dim=-1)  # [n_samples]
            obs_unc_err_pearson = _pearsonr(obs_unc, obs_err)

            # Done: uncertainty (var of logits) vs BCE error per sample.
            done_logit_mean = done_logit_all.mean(dim=0)  # [n_samples, done_dim]
            done_prob_mean = torch.sigmoid(done_logit_mean)
            done_target = done_flat.float()
            # Reduce to per-sample scalar.
            done_err = F.binary_cross_entropy(done_prob_mean, done_target, reduction="none")
            done_err = done_err.mean(dim=-1)
            done_unc = torch.var(done_logit_all, dim=0).mean(dim=-1)
            done_unc_err_pearson = _pearsonr(done_unc, done_err)

            # Log means for interpretability
            reward_unc_mean = float(reward_unc.mean().detach().cpu().item()) if reward_unc.numel() else 0.0
            reward_err_mean = float(reward_err.mean().detach().cpu().item()) if reward_err.numel() else 0.0
            obs_unc_mean = float(obs_unc.mean().detach().cpu().item()) if obs_unc.numel() else 0.0
            obs_err_mean = float(obs_err.mean().detach().cpu().item()) if obs_err.numel() else 0.0
            done_unc_mean = float(done_unc.mean().detach().cpu().item()) if done_unc.numel() else 0.0
            done_err_mean = float(done_err.mean().detach().cpu().item()) if done_err.numel() else 0.0

        # Update elite indices (lowest loss are best)
        n_elites = self.n_elites or self.ensemble_size
        n_elites = max(1, min(int(n_elites), self.ensemble_size))
        self._elite_indices[group] = torch.argsort(self._model_losses[group])[:n_elites]

        # Log world model metrics to wandb
        if hasattr(self.experiment, 'logger') and self.experiment.logger is not None:
            metrics_to_log = {
                f"world_model/{group}/loss/observation": total_loss_obs / self.ensemble_size,
                f"world_model/{group}/loss/reward": total_loss_rew / self.ensemble_size,
                f"world_model/{group}/loss/done": total_loss_done / self.ensemble_size,
                f"world_model/{group}/loss/total": (
                    total_loss_obs + total_loss_rew + total_loss_done
                ) / self.ensemble_size,
                f"world_model/{group}/mse/observation": total_mse_obs / self.ensemble_size,
                f"world_model/{group}/mse/reward": total_mse_rew / self.ensemble_size,
                f"world_model/{group}/accuracy/done": total_done_accuracy / self.ensemble_size,
                f"world_model/{group}/uncertainty/aleatoric/obs_logvar": aleatoric_obs_logvar,
                f"world_model/{group}/uncertainty/aleatoric/reward_logvar": aleatoric_rew_logvar,
                f"world_model/{group}/uncertainty/epistemic/obs_var": epistemic_obs_var,
                f"world_model/{group}/uncertainty/epistemic/reward_var": epistemic_rew_var,
                f"world_model/{group}/uncertainty/epistemic/done_var": epistemic_done_var,
                # Model bias metrics
                f"world_model/{group}/bias/reward_bias": reward_bias,
                f"world_model/{group}/bias/reward_mean_predicted": mean_pred_reward,
                f"world_model/{group}/bias/reward_mean_actual": mean_actual_reward,
                f"world_model/{group}/bias/reward_std_predicted": std_pred_reward,
                f"world_model/{group}/bias/reward_std_actual": std_actual_reward,
                f"world_model/{group}/bias/reward_mean_abs_error": mean_abs_error,
                f"world_model/{group}/percentage_error/reward": mean_percentage_error,
                f"world_model/{group}/percentage_error/reward_mask_rate": mape_mask_rate,
                f"world_model/{group}/percentage_error/reward_eps": mean_percentage_error_eps,
                f"world_model/{group}/percentage_error/reward_smape": mean_smape,
                f"world_model/{group}/percentage_error/reward_nmae": mean_nmae_percent,
                f"world_model/{group}/reward_normalize/enabled": float(getattr(self, "reward_normalize", False)),
                f"world_model/{group}/state_normalize/enabled": float(getattr(self, "state_normalize", False)),
                f"world_model/{group}/reward_normalize/norm_mean": reward_norm_mean,
                f"world_model/{group}/reward_normalize/norm_std": reward_norm_std,
                f"world_model/{group}/reward_normalize/reward_loss_coef": float(getattr(self, "reward_loss_coef", 1.0)),
                f"world_model/{group}/bias/reward_range_min": pred_reward_min,
                f"world_model/{group}/bias/reward_range_max": pred_reward_max,
                # Reward optimism diagnostics
                f"world_model/{group}/optimism/reward_over_rate": over_rate,
                f"world_model/{group}/optimism/reward_under_rate": under_rate,
                f"world_model/{group}/optimism/reward_mean_overpred": mean_overpred,
                f"world_model/{group}/optimism/reward_mean_underpred": mean_underpred,
                f"world_model/{group}/optimism/reward_lcb_over_rate": lcb_over_rate,
                f"world_model/{group}/optimism/reward_lcb_over_mean": lcb_over_mean,
                f"world_model/{group}/optimism/top10p_pred_bias": top_pred_bias,
                f"world_model/{group}/optimism/top10p_pred_over_rate": top_pred_over_rate,
                f"world_model/{group}/optimism/top10p_unc_bias": top_unc_bias,
                f"world_model/{group}/optimism/top10p_unc_over_rate": top_unc_over_rate,
                # Uncertainty calibration: correlation between epistemic uncertainty and error
                f"world_model/{group}/calibration/reward_unc_abs_err_pearson": reward_unc_err_pearson,
                f"world_model/{group}/calibration/obs_unc_mse_pearson": obs_unc_err_pearson,
                f"world_model/{group}/calibration/done_unc_bce_pearson": done_unc_err_pearson,
                f"world_model/{group}/calibration/reward_unc_mean": reward_unc_mean,
                f"world_model/{group}/calibration/reward_abs_err_mean": reward_err_mean,
                f"world_model/{group}/calibration/obs_unc_mean": obs_unc_mean,
                f"world_model/{group}/calibration/obs_mse_mean": obs_err_mean,
                f"world_model/{group}/calibration/done_unc_mean": done_unc_mean,
                f"world_model/{group}/calibration/done_bce_mean": done_err_mean,
                f"world_model/{group}/ensemble_size": self.ensemble_size,
                f"world_model/{group}/n_elites": n_elites,
            }
            # Use total_frames as step if available, otherwise use train_steps
            step = getattr(self.experiment, 'total_frames', self._train_steps[group])
            self.experiment.logger.log(metrics_to_log, step=step)

    def _predict_next(
        self, group: str, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_action = self._encode_action(group, action)
        if self.centralized_dynamics:
            obs_flat = obs.reshape(obs.shape[0], -1)
            act_flat = encoded_action.reshape(encoded_action.shape[0], -1)
        else:
            obs_flat = obs.flatten(0, 1)
            act_flat = encoded_action.flatten(0, 1)

        elites = self._elite_indices.get(group, None)
        if elites is None or elites.numel() == 0:
            elites = torch.arange(self.ensemble_size, device=self.device)

        # Inference: sample a random ensemble member *per transition* (per batch element),
        # similar to MBPO/PETS-style rollouts. The ensemble mean is only used conceptually
        # for training/targets; rollouts come from individual models.
        mu_next_list = []
        lv_delta_list = []
        mu_rew_list = []
        lv_rew_list = []
        done_logit_list = []
        for i in elites.tolist():
            mu_next, lv_delta, mu_rew, lv_rew, done_logit = self._dynamics[group][i](
                obs_flat, act_flat
            )
            mu_next_list.append(mu_next)
            lv_delta_list.append(lv_delta)
            mu_rew_list.append(mu_rew)
            lv_rew_list.append(lv_rew)
            done_logit_list.append(done_logit)

        mu_next_s = torch.stack(mu_next_list, dim=0)
        lv_delta_s = torch.stack(lv_delta_list, dim=0)
        mu_rew_s = torch.stack(mu_rew_list, dim=0)
        lv_rew_s = torch.stack(lv_rew_list, dim=0)
        done_logit_s = torch.stack(done_logit_list, dim=0)

        n_elites = mu_next_s.shape[0]
        bsz = mu_next_s.shape[1]
        model_idx = torch.randint(0, n_elites, (bsz,), device=mu_next_s.device)
        batch_idx = torch.arange(bsz, device=mu_next_s.device)

        # Sample (aleatoric uncertainty)
        if self.stochastic_dynamics:
            std_next_s = torch.exp(0.5 * lv_delta_s)
            std_rew_s = torch.exp(0.5 * lv_rew_s)
            next_obs_flat_s = torch.normal(mu_next_s, std_next_s)
            reward_flat_s = torch.normal(mu_rew_s, std_rew_s)
        else:
            next_obs_flat_s = mu_next_s
            reward_flat_s = mu_rew_s

        next_obs_flat = next_obs_flat_s[model_idx, batch_idx]
        reward_flat = reward_flat_s[model_idx, batch_idx]
        done_logit = done_logit_s[model_idx, batch_idx]

        if self.centralized_dynamics:
            next_obs = next_obs_flat.reshape(obs.shape)
            reward = reward_flat.reshape(*obs.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs.shape[:-1], 1)
        else:
            next_obs = next_obs_flat.reshape(obs.shape)
            reward = reward_flat.reshape(*obs.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs.shape[:-1], 1)

        return next_obs, reward, done_logit

    def _predict_next_for_vis(
        self, group: str, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next transition using MEAN predictions (no stochastic sampling).
        
        This is used for visualization to show model accuracy without sampling noise.
        For actual rollouts, use `_predict_next` which includes stochastic sampling.
        """
        encoded_action = self._encode_action(group, action)
        if self.centralized_dynamics:
            obs_flat = obs.reshape(obs.shape[0], -1)
            act_flat = encoded_action.reshape(encoded_action.shape[0], -1)
        else:
            obs_flat = obs.flatten(0, 1)
            act_flat = encoded_action.flatten(0, 1)

        elites = self._elite_indices.get(group, None)
        if elites is None or elites.numel() == 0:
            elites = torch.arange(self.ensemble_size, device=self.device)

        # Get predictions from all elite models
        mu_next_list = []
        mu_rew_list = []
        done_logit_list = []
        for i in elites.tolist():
            mu_next, _, mu_rew, _, done_logit = self._dynamics[group][i](
                obs_flat, act_flat
            )
            mu_next_list.append(mu_next)
            mu_rew_list.append(mu_rew)
            done_logit_list.append(done_logit)

        mu_next_s = torch.stack(mu_next_list, dim=0)  # [E, N, obs_dim]
        mu_rew_s = torch.stack(mu_rew_list, dim=0)  # [E, N, rew_dim]
        done_logit_s = torch.stack(done_logit_list, dim=0)  # [E, N, 1]

        # Use ensemble MEAN (not sampling) for visualization
        next_obs_flat = mu_next_s.mean(dim=0)  # [N, obs_dim]
        reward_flat = mu_rew_s.mean(dim=0)  # [N, rew_dim]
        done_logit = done_logit_s.mean(dim=0)  # [N, 1]

        if self.centralized_dynamics:
            next_obs = next_obs_flat.reshape(obs.shape)
            reward = reward_flat.reshape(*obs.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs.shape[:-1], 1)
        else:
            next_obs = next_obs_flat.reshape(obs.shape)
            reward = reward_flat.reshape(*obs.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs.shape[:-1], 1)

        return next_obs, reward, done_logit

    def _get_rollout_uncertainty_threshold(self, group: str) -> Optional[float]:
        """Return the active uncertainty threshold for this group (if any)."""
        ths = getattr(self, "rollout_uncertainty_thresholds", None)
        if isinstance(ths, dict) and group in ths and ths[group] is not None:
            try:
                return float(ths[group])
            except Exception:
                pass
        th = getattr(self, "rollout_uncertainty_threshold", None)
        if th is None:
            return None
        try:
            return float(th)
        except Exception:
            return None

    @torch.no_grad()
    def _predict_next_with_uncertainty(
        self, group: str, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict next transition and compute per-env uncertainty metrics.

        Returns:
            next_obs, reward, done_logit (same semantics as `_predict_next`)
            unc: dict of per-env tensors with shape [B]
        """
        encoded_action = self._encode_action(group, action)
        if self.centralized_dynamics:
            obs_flat = obs.reshape(obs.shape[0], -1)
            act_flat = encoded_action.reshape(encoded_action.shape[0], -1)
            B = int(obs.shape[0])
            n_agents = int(obs.shape[-2])
        else:
            # Flatten agent dimension: [B, n_agents, ...] -> [B*n_agents, ...]
            B = int(obs.shape[0])
            n_agents = int(obs.shape[-2])
            obs_flat = obs.flatten(0, 1)
            act_flat = encoded_action.flatten(0, 1)

        elites = self._elite_indices.get(group, None)
        if elites is None or elites.numel() == 0:
            elites = torch.arange(self.ensemble_size, device=self.device)

        mu_next_list = []
        lv_delta_list = []
        mu_rew_list = []
        lv_rew_list = []
        done_logit_list = []
        for i in elites.tolist():
            mu_next, lv_delta, mu_rew, lv_rew, done_logit = self._dynamics[group][i](
                obs_flat, act_flat
            )
            mu_next_list.append(mu_next)
            lv_delta_list.append(lv_delta)
            mu_rew_list.append(mu_rew)
            lv_rew_list.append(lv_rew)
            done_logit_list.append(done_logit)

        mu_next_s = torch.stack(mu_next_list, dim=0)  # [E, N, obs_dim]
        lv_delta_s = torch.stack(lv_delta_list, dim=0)  # [E, N, obs_dim]
        mu_rew_s = torch.stack(mu_rew_list, dim=0)  # [E, N, rew_dim]
        lv_rew_s = torch.stack(lv_rew_list, dim=0)  # [E, N, rew_dim]
        done_logit_s = torch.stack(done_logit_list, dim=0)  # [E, N, 1]

        n_elites = int(mu_next_s.shape[0])
        N = int(mu_next_s.shape[1])
        model_idx = torch.randint(0, n_elites, (N,), device=mu_next_s.device)
        batch_idx = torch.arange(N, device=mu_next_s.device)

        # Sample (aleatoric uncertainty)
        if self.stochastic_dynamics:
            std_next_s = torch.exp(0.5 * lv_delta_s)
            std_rew_s = torch.exp(0.5 * lv_rew_s)
            next_obs_flat_s = torch.normal(mu_next_s, std_next_s)
            reward_flat_s = torch.normal(mu_rew_s, std_rew_s)
        else:
            next_obs_flat_s = mu_next_s
            reward_flat_s = mu_rew_s

        next_obs_flat = next_obs_flat_s[model_idx, batch_idx]
        reward_flat = reward_flat_s[model_idx, batch_idx]
        done_logit = done_logit_s[model_idx, batch_idx]

        # --- Uncertainty metrics (computed across ensemble) ---
        epistemic_obs_var = torch.var(mu_next_s, dim=0).mean(dim=-1)  # [N]
        epistemic_rew_std = torch.std(mu_rew_s, dim=0).mean(dim=-1)  # [N]
        aleatoric_obs_logvar = lv_delta_s.mean(dim=0).mean(dim=-1)  # [N]
        aleatoric_rew_logvar = lv_rew_s.mean(dim=0).mean(dim=-1)  # [N]

        aleatoric_obs_std = torch.exp(0.5 * aleatoric_obs_logvar)
        aleatoric_rew_std = torch.exp(0.5 * aleatoric_rew_logvar)

        epistemic_obs_std = torch.sqrt(epistemic_obs_var.clamp_min(0.0))
        total_obs_unc = torch.sqrt(
            epistemic_obs_var + aleatoric_obs_std.pow(2).clamp_min(0.0)
        )
        total_rew_unc = torch.sqrt(
            epistemic_rew_std.pow(2) + aleatoric_rew_std.pow(2).clamp_min(0.0)
        )

        if self.centralized_dynamics:
            # Already per-env: [B]
            unc = {
                "epistemic_obs_var": epistemic_obs_var,
                "epistemic_rew_std": epistemic_rew_std,
                "aleatoric_obs_logvar": aleatoric_obs_logvar,
                "aleatoric_rew_logvar": aleatoric_rew_logvar,
                "total_obs_unc": total_obs_unc,
                "total_rew_unc": total_rew_unc,
            }
        else:
            # Aggregate per-agent values to per-env values: reshape [B*n_agents] -> [B, n_agents]
            def _per_env(x: torch.Tensor) -> torch.Tensor:
                return x.reshape(B, n_agents).mean(dim=-1)

            unc = {
                "epistemic_obs_var": _per_env(epistemic_obs_var),
                "epistemic_rew_std": _per_env(epistemic_rew_std),
                "aleatoric_obs_logvar": _per_env(aleatoric_obs_logvar),
                "aleatoric_rew_logvar": _per_env(aleatoric_rew_logvar),
                "total_obs_unc": _per_env(total_obs_unc),
                "total_rew_unc": _per_env(total_rew_unc),
            }

        # Reshape sampled outputs back to original shapes
        if self.centralized_dynamics:
            next_obs = next_obs_flat.reshape(obs.shape)
            reward = reward_flat.reshape(*obs.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs.shape[:-1], 1)
        else:
            next_obs = next_obs_flat.reshape(obs.shape)
            reward = reward_flat.reshape(*obs.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs.shape[:-1], 1)

        return next_obs, reward, done_logit, unc

    def _step_oracle_env(
        self, group: str, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step the environment (oracle) to get ground truth next observation, reward, and done.
        
        This bypasses the world model and uses the actual simulator/environment.
        Useful for sanity checking the MBPO algorithm.
        
        Args:
            group: Agent group name
            obs: Current observation tensor [B, n_agents, obs_dim] or [B, obs_dim] for centralized
            action: Action tensor [B, n_agents, action_dim] or [B, action_dim] for centralized
            
        Returns:
            next_obs: Next observation [B, n_agents, obs_dim] or [B, obs_dim]
            reward: Reward [B, n_agents, 1] or [B, 1]
            done: Done flag [B, n_agents, 1] or [B, 1]
        """
        if not hasattr(self, "experiment") or self.experiment is None:
            raise RuntimeError("Cannot use oracle: experiment not available")
        
        # Get or create oracle environment instance
        if not hasattr(self, "_oracle_env"):
            try:
                # Try to use the experiment's rollout_env
                if hasattr(self.experiment, "rollout_env") and self.experiment.rollout_env is not None:
                    # Create a copy/clone for oracle use to avoid interfering with training
                    # For now, we'll use it directly but in a way that doesn't interfere
                    self._oracle_env = self.experiment.rollout_env
                else:
                    # Create a new environment instance
                    if hasattr(self.experiment, "task"):
                        env_func = self.experiment.task.get_env_fun(
                            num_envs=obs.shape[0] if len(obs.shape) >= 2 else 1,
                            continuous_actions=self.experiment.continuous_actions,
                            seed=getattr(self.experiment, "seed", 0),
                            device=self.device,
                        )
                        self._oracle_env = env_func()
                        # Apply transforms if needed
                        if hasattr(self.experiment, "test_env") and hasattr(self.experiment.test_env, "transform"):
                            from torchrl.envs import TransformedEnv
                            transforms = self.experiment.test_env.transform
                            self._oracle_env = TransformedEnv(self._oracle_env, transforms.clone())
                    else:
                        raise RuntimeError("Cannot create oracle environment: task not available")
            except Exception as e:
                raise RuntimeError(f"Cannot use oracle: failed to get environment: {e}")
        
        # Prepare TensorDict for environment step
        batch_dims = obs.shape[:-2] if len(obs.shape) >= 3 else obs.shape[:-1]
        n_agents = obs.shape[-2] if len(obs.shape) >= 3 else 1
        B = int(batch_dims[0]) if len(batch_dims) > 0 else 1
        
        # Create input TensorDict matching environment format
        # The environment expects the current state/observation to be set
        td_in = TensorDict({}, batch_size=batch_dims, device=self.device)
        td_in.set(
            group,
            TensorDict(
                {
                    "observation": obs,
                },
                batch_size=(*batch_dims, n_agents) if len(obs.shape) >= 3 else batch_dims,
                device=self.device,
            ),
        )
        td_in.set((group, "action"), action)
        
        # Step the environment
        try:
            # Use the environment's step function
            # Note: This requires the environment to be in the state corresponding to `obs`
            # For most TorchRL environments, we need to set the state first or use reset
            # For now, we'll try stepping directly - this works if the environment
            # can step from the observation state (some environments support this)
            td_out = self._oracle_env.step(td_in)
            
            # Extract results
            next_obs = td_out.get((group, "observation"))
            reward = td_out.get(("next", group, "reward"), None)
            if reward is None:
                reward = td_out.get(("next", "reward"), None)
                if reward is not None:
                    # Expand to match group shape
                    reward = reward.expand(*next_obs.shape[:-1], 1)
            
            done = td_out.get(("next", group, "done"), None)
            if done is None:
                done = td_out.get(("next", "done"), None)
                if done is not None:
                    # Expand to match group shape
                    if len(next_obs.shape) >= 3:
                        done = done.unsqueeze(-2).expand(*next_obs.shape[:-1], 1)
                    else:
                        done = done.unsqueeze(-1)
            
            # Ensure done is boolean and has correct shape
            if done is not None:
                done = done.to(torch.bool)
                if done.dim() < next_obs.dim():
                    done = done.unsqueeze(-1)
            else:
                # Default: not done
                done = torch.zeros(*next_obs.shape[:-1], 1, device=self.device, dtype=torch.bool)
            
            # Ensure reward has correct shape
            if reward is None:
                reward = torch.zeros(*next_obs.shape[:-1], 1, device=self.device, dtype=torch.float32)
            elif reward.dim() < next_obs.dim():
                reward = reward.unsqueeze(-1)
            
            # Create done_logit (convert boolean to logit: 0.0 for False, large positive for True)
            done_logit = done.float() * 10.0 - 5.0  # Maps False->-5.0, True->5.0 (sigmoid(5.0) ≈ 0.99)
            
            return next_obs, reward, done_logit
            
        except Exception as e:
            # Fallback: if direct step doesn't work, try alternative approach
            # Some environments may require reset first
            raise RuntimeError(f"Oracle environment step failed: {e}. "
                             f"Note: Oracle mode requires environment to support stepping from arbitrary states.")

    def _sample_start_states(
        self, group: str, flat_batch: TensorDictBase, sample_size: int
    ) -> Optional[TensorDictBase]:
        buffer = self.experiment.replay_buffers[group]
        # If the replay buffer is wrapped (MBPO real+synthetic), sample starts from real only.
        if hasattr(buffer, "_real"):
            buffer = buffer._real
        if len(buffer) > 0:
            try:
                start = buffer.sample(sample_size).to(self.device)
                return start
            except Exception:
                pass
        if flat_batch.numel() == 0:
            return None
        return flat_batch[:sample_size].to(self.device)

    def _generate_model_rollouts(
        self, group: str, flat_batch: TensorDictBase
    ) -> Optional[TensorDictBase]:
        sr = getattr(self, "syn_ratio", None)
        if sr is not None:
            # New semantics: generate ~syn_ratio * N_real synthetic transitions per iteration.
            target_synth = int(math.ceil(float(flat_batch.batch_size[0]) * max(0.0, float(sr))))
        else:
            # Backward-compatible semantics: target_synth ~= (1-real_ratio) * N_real.
            rr = float(getattr(self, "real_ratio", 1.0))
            rr = min(max(rr, 0.0), 1.0)
            target_synth = int(math.ceil(float(flat_batch.batch_size[0]) * (1.0 - rr)))
        if target_synth <= 0:
            return None

        horizon = max(1, int(self._current_rollout_horizon(group)))
        start_states = max(1, int(math.ceil(target_synth / float(horizon))))

        start = self._sample_start_states(group, flat_batch, start_states)
        if start is None or start.numel() == 0:
            return None

        obs = start.get((group, "observation")).to(self.device)
        # Parent batch dims do NOT include the agent dimension. The group sub-tensordict
        # must include it (matching what the collector produces).
        batch_dims = obs.shape[:-2]  # e.g. [B]
        n_agents = obs.shape[-2]
        done_mask = torch.zeros((*batch_dims, n_agents, 1), device=self.device, dtype=torch.bool)

        # Each element in `rollouts` is one "time step" TensorDict with batch_size=[B].
        # We'll stack these along dim=0 to produce a proper trajectory TensorDict with
        # batch_size=[T, B], which is required for MAPPO's GAE computation.
        rollouts: List[TensorDictBase] = []
        # Try to preserve collector metadata when available (helps match replay buffer schema).
        traj_ids = None
        if ("collector", "traj_ids") in start.keys(True, True):
            traj_ids = start.get(("collector", "traj_ids")).to(self.device)
        else:
            traj_ids = torch.zeros(batch_dims, device=self.device, dtype=torch.long)

        # Episode reward is often logged/stored in the replay buffer; maintain it if present.
        if (group, "episode_reward") in start.keys(True, True):
            ep_rew = start.get((group, "episode_reward")).to(self.device)
        else:
            ep_rew = torch.zeros((*batch_dims, n_agents, 1), device=self.device, dtype=torch.float32)

        # Optional uncertainty thresholding to avoid using inaccurate synthetic samples.
        unc_threshold = self._get_rollout_uncertainty_threshold(group)
        unc_metric = str(
            getattr(self, "rollout_uncertainty_metric", "total_rew_unc") or "total_rew_unc"
        )

        for _ in range(horizon):
            td_in = TensorDict({}, batch_size=batch_dims, device=self.device)
            td_in.set(
                group,
                TensorDict(
                    {"observation": obs},
                    batch_size=(*batch_dims, n_agents),
                    device=self.device,
                ),
            )
            if (
                self.action_mask_spec is not None
                and (group, "action_mask") in start.keys(True, True)
            ):
                td_in.get(group).set(
                    "action_mask", start.get((group, "action_mask")).to(self.device)
                )

            policy_td = self._policies_for_collection[group](td_in)
            action = policy_td.get((group, "action")).detach()
            # Policy may also output logits/log_prob/loc/scale; keep them to match real replay schema.
            policy_group = policy_td.get(group)

            # Env-level termination mask so we can prevent adding transitions after termination.
            env_done = done_mask.any(dim=-2)  # [B, 1]
            env_active = (~env_done).squeeze(-1)  # [B]

            # Decide if this synthetic transition is valid (active and below uncertainty threshold).
            synthetic_valid_env = env_active.clone()
            invalid_agent = None
            # Use oracle (simulator) if enabled, otherwise use world model
            if getattr(self, "use_oracle", False):
                try:
                    next_obs, reward_pred, done_logit = self._step_oracle_env(group, obs, action)
                except Exception as e:
                    # If oracle fails, fall back to model prediction
                    import warnings
                    warnings.warn(f"Oracle step failed, falling back to model: {e}")
                    if unc_threshold is None:
                        next_obs, reward_pred, done_logit = self._predict_next(group, obs, action)
                    else:
                        next_obs, reward_pred, done_logit, unc = self._predict_next_with_uncertainty(
                            group, obs, action
                        )
                        unc_val = unc.get(unc_metric, None)
                        if unc_val is None:
                            unc_val = unc.get("total_rew_unc", None)
                        if unc_val is not None:
                            unc_val = unc_val.to(self.device)
                            invalid_unc = (~torch.isfinite(unc_val)) | (unc_val > float(unc_threshold))
                            synthetic_valid_env = synthetic_valid_env & (~invalid_unc)
            elif unc_threshold is None:
                next_obs, reward_pred, done_logit = self._predict_next(group, obs, action)
            else:
                next_obs, reward_pred, done_logit, unc = self._predict_next_with_uncertainty(
                    group, obs, action
                )
                unc_val = unc.get(unc_metric, None)
                if unc_val is None:
                    unc_val = unc.get("total_rew_unc", None)
                if unc_val is not None:
                    unc_val = unc_val.to(self.device)
                    invalid_unc = (~torch.isfinite(unc_val)) | (unc_val > float(unc_threshold))
                    synthetic_valid_env = synthetic_valid_env & (~invalid_unc)

            invalid_env = ~synthetic_valid_env
            if invalid_env.any():
                invalid_agent = invalid_env.view(*batch_dims, 1, 1).expand(
                    *batch_dims, n_agents, 1
                )
                # Do not propagate the model further for invalid envs.
                next_obs = torch.where(invalid_agent, obs, next_obs)
            # Optionally replace learned reward with oracle reward computed from predicted next state.
            if self._oracle_reward_enabled():
                try:
                    reward_pred = self._oracle_reward_from_predicted_next_obs(group, next_obs)
                except Exception:
                    # If oracle reward can't be computed, fall back to learned reward head.
                    pass

            if invalid_agent is not None:
                # Ensure invalid transitions contribute nothing and terminate immediately.
                reward_pred = reward_pred.masked_fill(invalid_agent, 0.0)
            done_prob = torch.sigmoid(done_logit)
            done_flag = done_prob > 0.5
            if invalid_agent is not None:
                done_flag = done_flag | invalid_agent
            done_flag = done_flag | done_mask

            # Update episode reward (per-agent cumulative)
            ep_rew_next = ep_rew + reward_pred.detach()

            # Compute env-level done/terminated (top-level keys in collector batch)
            env_done_next = done_flag.any(dim=-2)  # [B, 1]

            rollout_td = TensorDict({}, batch_size=batch_dims, device=self.device)
            rollout_td.set(
                group,
                TensorDict(
                    {
                        "observation": obs.detach(),
                        "action": action,
                        "episode_reward": ep_rew.detach(),
                    },
                    batch_size=(*batch_dims, n_agents),
                    device=self.device,
                ),
            )
            # Optional policy outputs
            for k in ("logits", "log_prob", "loc", "scale"):
                if k in policy_group.keys():
                    rollout_td.get(group).set(k, policy_group.get(k).detach())

            # Collector metadata
            rollout_td.set(
                "collector",
                TensorDict({"traj_ids": traj_ids}, batch_size=batch_dims, device=self.device),
            )

            # Top-level done/terminated (current transition). We mirror next for simplicity.
            rollout_td.set("done", env_done_next.detach())
            rollout_td.set("terminated", env_done_next.detach())
            rollout_td.set(
                "synthetic_valid",
                synthetic_valid_env.view(*batch_dims, 1).to(torch.bool).detach(),
            )

            rollout_td.set(
                "next",
                TensorDict(
                    {
                        group: TensorDict(
                            {
                                "observation": next_obs.detach(),
                                "reward": reward_pred.detach(),
                                "done": done_flag.detach(),
                                "terminated": done_flag.detach(),
                                "episode_reward": ep_rew_next.detach(),
                            },
                            batch_size=(*batch_dims, n_agents),
                            device=self.device,
                        )
                    },
                    batch_size=batch_dims,
                    device=self.device,
                ),
            )
            rollout_td.get("next").set("done", env_done_next.detach())
            rollout_td.get("next").set("terminated", env_done_next.detach())

            rollouts.append(rollout_td)

            obs = next_obs
            done_mask = done_flag
            ep_rew = ep_rew_next
            if torch.all(done_mask):
                break

        if not rollouts:
            return None

        # Important: use stack (not cat) to preserve [T,B] structure.
        out = torch.stack(rollouts, dim=0)

        # Enforce an approximate target synthetic dataset size for this iteration while
        # keeping the [T,B] structure (avoid flattening here; MAPPO needs time dimension).
        #
        # Total transitions ~= T * B. If we have too many, slice the batch dimension.
        try:
            T = int(out.batch_size[0])
            B = int(out.batch_size[1]) if len(out.batch_size) > 1 else 1
        except Exception:
            T, B = len(rollouts), int(batch_dims[0]) if batch_dims else 1
        if T > 0 and B > 1:
            total = T * B
            if total > target_synth:
                # Keep as many start states as possible without exceeding the target (floor).
                keep_B = max(1, min(B, int(target_synth // T)))
                if keep_B < B:
                    out = out[:, :keep_B]
        return out

    def save_world_model(self, filepath: str) -> None:
        """
        Save the world model state to a file.
        
        Args:
            filepath: Path where to save the world model state
        """
        import pathlib
        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            "dynamics": {
                group: [model.state_dict() for model in models]
                for group, models in self._dynamics.items()
            },
            "dyn_optimizers": {
                group: [opt.state_dict() for opt in optimizers]
                for group, optimizers in self._dyn_optimizers.items()
            },
            "model_losses": {
                group: losses.cpu() for group, losses in self._model_losses.items()
            },
            "train_steps": self._train_steps.copy(),
            "elite_indices": {
                group: indices.cpu() for group, indices in self._elite_indices.items()
            },
            "config": {
                "rollout_horizon": self.rollout_horizon,
                "rollout_horizon_min": getattr(self, "rollout_horizon_min", None),
                "rollout_horizon_max": getattr(self, "rollout_horizon_max", None),
                "rollout_horizon_ramp_steps": int(
                    getattr(self, "rollout_horizon_ramp_steps", 0) or 0
                ),
                "model_train_freq": self.model_train_freq,
                "ensemble_size": self.ensemble_size,
                "model_batch_size": self.model_batch_size,
                "real_ratio": self.real_ratio,
                "syn_ratio": getattr(self, "syn_ratio", None),
                "temperature": self.temperature,
                "model_lr": self.model_lr,
                "model_hidden_size": self.model_hidden_size,
                "model_num_layers": self.model_num_layers,
                # Optional recurrent-world-model config (used by MbpoRecurrent*).
                # Backward compatible: older checkpoints won't have these keys.
                "history_length": int(getattr(self, "history_length", 0) or 0),
                "future_length": int(getattr(self, "future_length", 1) or 1),
                "gru_num_layers": int(getattr(self, "world_model_gru_num_layers", 1) or 1),
                "reward_loss_coef": float(getattr(self, "reward_loss_coef", 1.0)),
                "logvar_loss_coef": float(getattr(self, "logvar_loss_coef", 1.0)),
                "state_normalize": bool(getattr(self, "state_normalize", False)),
                "reward_normalize": bool(getattr(self, "reward_normalize", False)),
                "reward_norm_eps": float(getattr(self, "reward_norm_eps", 1e-6)),
                "separate_reward_net": bool(getattr(self, "separate_reward_net", False)),
                "centralized_dynamics": self.centralized_dynamics,
                "stochastic_dynamics": self.stochastic_dynamics,
                "n_elites": self.n_elites,
                "min_log_var": self.min_log_var,
                "max_log_var": self.max_log_var,
                "warmup_steps": self.warmup_steps,
                "rollout_uncertainty_threshold": getattr(self, "rollout_uncertainty_threshold", None),
                "rollout_uncertainty_thresholds": getattr(self, "rollout_uncertainty_thresholds", None),
                "rollout_uncertainty_metric": str(
                    getattr(self, "rollout_uncertainty_metric", "total_rew_unc") or "total_rew_unc"
                ),
                "use_oracle": bool(getattr(self, "use_oracle", False)),
                "oracle_reward": bool(getattr(self, "oracle_reward", False)),
                "oracle_reward_mode": str(getattr(self, "oracle_reward_mode", "")),
                "oracle_reward_goal_rel_indices": getattr(self, "oracle_reward_goal_rel_indices", None),
                "oracle_reward_scale": float(getattr(self, "oracle_reward_scale", 1.0)),
                "oracle_reward_disable_reward_head_loss": bool(
                    getattr(self, "oracle_reward_disable_reward_head_loss", True)
                ),
            },
        }
        torch.save(state_dict, path)

    def load_world_model(self, filepath: str, strict: bool = True) -> None:
        """
        Load the world model state from a file.
        
        Args:
            filepath: Path to the saved world model state
            strict: If True, raise an error if the loaded config doesn't match current config.
                    If False, only load model parameters and optimizers.
        """
        import pathlib
        path = pathlib.Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"World model file not found: {filepath}")
        
        state_dict = torch.load(path, map_location=self.device)
        
        # Verify config matches if strict=True
        if strict:
            loaded_config = state_dict.get("config", {})
            current_config = {
                "rollout_horizon": self.rollout_horizon,
                "rollout_horizon_min": getattr(self, "rollout_horizon_min", None),
                "rollout_horizon_max": getattr(self, "rollout_horizon_max", None),
                "rollout_horizon_ramp_steps": int(
                    getattr(self, "rollout_horizon_ramp_steps", 0) or 0
                ),
                "model_train_freq": self.model_train_freq,
                "ensemble_size": self.ensemble_size,
                "model_batch_size": self.model_batch_size,
                "real_ratio": self.real_ratio,
                "syn_ratio": getattr(self, "syn_ratio", None),
                "temperature": self.temperature,
                "model_lr": self.model_lr,
                "model_hidden_size": self.model_hidden_size,
                "model_num_layers": self.model_num_layers,
                # Optional recurrent-world-model config (used by MbpoRecurrent*).
                "history_length": int(getattr(self, "history_length", 0) or 0),
                "future_length": int(getattr(self, "future_length", 1) or 1),
                "gru_num_layers": int(getattr(self, "world_model_gru_num_layers", 1) or 1),
                "reward_loss_coef": float(getattr(self, "reward_loss_coef", 1.0)),
                "logvar_loss_coef": float(getattr(self, "logvar_loss_coef", 1.0)),
                "state_normalize": bool(getattr(self, "state_normalize", False)),
                "reward_normalize": bool(getattr(self, "reward_normalize", False)),
                "reward_norm_eps": float(getattr(self, "reward_norm_eps", 1e-6)),
                "separate_reward_net": bool(getattr(self, "separate_reward_net", False)),
                "centralized_dynamics": self.centralized_dynamics,
                "stochastic_dynamics": self.stochastic_dynamics,
                "n_elites": self.n_elites,
                "min_log_var": self.min_log_var,
                "max_log_var": self.max_log_var,
                "warmup_steps": self.warmup_steps,
                "rollout_uncertainty_threshold": getattr(self, "rollout_uncertainty_threshold", None),
                "rollout_uncertainty_thresholds": getattr(self, "rollout_uncertainty_thresholds", None),
                "rollout_uncertainty_metric": str(
                    getattr(self, "rollout_uncertainty_metric", "total_rew_unc") or "total_rew_unc"
                ),
                "use_oracle": bool(getattr(self, "use_oracle", False)),
                "oracle_reward": bool(getattr(self, "oracle_reward", False)),
                "oracle_reward_mode": str(getattr(self, "oracle_reward_mode", "")),
                "oracle_reward_goal_rel_indices": getattr(self, "oracle_reward_goal_rel_indices", None),
                "oracle_reward_scale": float(getattr(self, "oracle_reward_scale", 1.0)),
                "oracle_reward_disable_reward_head_loss": bool(
                    getattr(self, "oracle_reward_disable_reward_head_loss", True)
                ),
            }
            for key, value in current_config.items():
                if key in loaded_config and loaded_config[key] != value:
                    raise ValueError(
                        f"Config mismatch for '{key}': loaded={loaded_config[key]}, current={value}. "
                        f"Set strict=False to load anyway (only model parameters will be loaded)."
                    )
        
        # Load dynamics models
        dynamics_state = state_dict.get("dynamics", {})
        for group in self._dynamics.keys():
            if group not in dynamics_state:
                if strict:
                    raise KeyError(f"Group '{group}' not found in saved world model")
                continue
            
            saved_models = dynamics_state[group]
            if len(saved_models) != len(self._dynamics[group]):
                if strict:
                    raise ValueError(
                        f"Ensemble size mismatch for group '{group}': "
                        f"loaded={len(saved_models)}, current={len(self._dynamics[group])}"
                    )
                # Load only matching models
                n_models = min(len(saved_models), len(self._dynamics[group]))
                saved_models = saved_models[:n_models]
                self._dynamics[group] = self._dynamics[group][:n_models]
                self._dyn_optimizers[group] = self._dyn_optimizers[group][:n_models]
            
            for model, saved_state in zip(self._dynamics[group], saved_models):
                model.load_state_dict(saved_state)
        
        # Load optimizers
        optimizers_state = state_dict.get("dyn_optimizers", {})
        for group in self._dyn_optimizers.keys():
            if group not in optimizers_state:
                continue
            saved_opts = optimizers_state[group]
            for opt, saved_state in zip(self._dyn_optimizers[group], saved_opts):
                opt.load_state_dict(saved_state)
        
        # Load training state
        if "model_losses" in state_dict:
            for group, losses in state_dict["model_losses"].items():
                if group in self._model_losses:
                    self._model_losses[group] = losses.to(self.device)
        
        if "train_steps" in state_dict:
            self._train_steps.update(state_dict["train_steps"])
        
        if "elite_indices" in state_dict:
            for group, indices in state_dict["elite_indices"].items():
                if group in self._elite_indices:
                    self._elite_indices[group] = indices.to(self.device)


class MbpoMasac(_MbpoWorldModelMixin, Masac):
    """Model-Based Policy Optimisation (MBPO) built on top of MASAC.

    This implementation keeps MASAC losses/policies and augments training with a
    learned ensemble dynamics model that generates short synthetic rollouts,
    mixed into the replay buffer.
    """

    def __init__(
        self,
        rollout_horizon: int,
        model_train_freq: int,
        ensemble_size: int,
        model_batch_size: int,
        real_ratio: float,
        temperature: float,
        model_lr: float,
        model_hidden_size: int,
        model_num_layers: int,
        syn_ratio: Optional[float] = None,
        rollout_horizon_min: Optional[int] = None,
        rollout_horizon_max: Optional[int] = None,
        rollout_horizon_ramp_steps: int = 0,
        reward_loss_coef: float = 1.0,
        logvar_loss_coef: float = 1.0,  # scaling coefficient for log-variance term in NLL loss
        state_normalize: bool = False,
        reward_normalize: bool = False,
        reward_norm_eps: float = 1e-6,
        normalization_stats_json: Optional[str] = None,
        separate_reward_net: bool = False,
        training_iterations: int = 0,
        centralized_dynamics: bool = False,
        stochastic_dynamics: bool = True,
        n_elites: Optional[int] = None,
        min_log_var: float = -10.0,
        max_log_var: float = -2.0,
        warmup_steps: int = 0,
        rollout_uncertainty_threshold: Optional[float] = None,
        rollout_uncertainty_thresholds: Optional[Dict[str, float]] = None,
        rollout_uncertainty_metric: str = "total_rew_unc",
        use_oracle: bool = False,
        load_world_model_path: Optional[str] = None,
        load_world_model_strict: bool = True,
        save_world_model_path: Optional[str] = None,
        save_world_model_interval: Optional[int] = None,
        oracle_reward: bool = False,
        oracle_reward_mode: str = "goal_distance_from_obs",
        oracle_reward_goal_rel_indices: Optional[List[int]] = None,
        oracle_reward_scale: float = 1.0,
        oracle_reward_disable_reward_head_loss: bool = True,
        world_model_debug_video: bool = False,
        world_model_debug_video_interval: int = 200,
        world_model_debug_video_horizon: int = 50,
        world_model_debug_video_fps: int = 20,
        world_model_debug_video_env_index: int = 0,
        world_model_debug_video_obs_xy_indices: Optional[List[int]] = None,
        world_model_debug_video_mode: str = "open_loop",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mbpo_init(
            rollout_horizon=rollout_horizon,
            rollout_horizon_min=rollout_horizon_min,
            rollout_horizon_max=rollout_horizon_max,
            rollout_horizon_ramp_steps=rollout_horizon_ramp_steps,
            model_train_freq=model_train_freq,
            ensemble_size=ensemble_size,
            model_batch_size=model_batch_size,
            real_ratio=real_ratio,
            temperature=temperature,
            model_lr=model_lr,
            model_hidden_size=model_hidden_size,
            model_num_layers=model_num_layers,
            syn_ratio=syn_ratio,
            reward_loss_coef=reward_loss_coef,
            logvar_loss_coef=logvar_loss_coef,
            state_normalize=state_normalize,
            reward_normalize=reward_normalize,
            reward_norm_eps=reward_norm_eps,
            normalization_stats_json=normalization_stats_json,
            separate_reward_net=separate_reward_net,
            training_iterations=training_iterations,
            centralized_dynamics=centralized_dynamics,
            stochastic_dynamics=stochastic_dynamics,
            n_elites=n_elites,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
            warmup_steps=warmup_steps,
            rollout_uncertainty_threshold=rollout_uncertainty_threshold,
            rollout_uncertainty_thresholds=rollout_uncertainty_thresholds,
            rollout_uncertainty_metric=rollout_uncertainty_metric,
            use_oracle=use_oracle,
            load_world_model_path=load_world_model_path,
            load_world_model_strict=load_world_model_strict,
            save_world_model_path=save_world_model_path,
            save_world_model_interval=save_world_model_interval,
            oracle_reward=oracle_reward,
            oracle_reward_mode=oracle_reward_mode,
            oracle_reward_goal_rel_indices=oracle_reward_goal_rel_indices,
            oracle_reward_scale=oracle_reward_scale,
            oracle_reward_disable_reward_head_loss=oracle_reward_disable_reward_head_loss,
            world_model_debug_video=world_model_debug_video,
            world_model_debug_video_interval=world_model_debug_video_interval,
            world_model_debug_video_horizon=world_model_debug_video_horizon,
            world_model_debug_video_fps=world_model_debug_video_fps,
            world_model_debug_video_env_index=world_model_debug_video_env_index,
            world_model_debug_video_obs_xy_indices=world_model_debug_video_obs_xy_indices,
            world_model_debug_video_mode=world_model_debug_video_mode,
        )

    #############################
    # Overridden abstract methods
    #############################

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return super()._get_parameters(group, loss)

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        return super()._get_policy_for_loss(group, model_config, continuous)

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        return super()._get_policy_for_collection(policy_for_loss, group, continuous)


class MbpoMappo(_MbpoWorldModelMixin, Mappo):
    """Model-Based Policy Optimisation (MBPO) built on top of MAPPO.

    Note: MAPPO is on-policy; MBPO-style synthetic rollouts will be mixed into the
    on-policy replay buffer used for PPO updates in BenchMARL.
    """

    def __init__(
        self,
        rollout_horizon: int,
        model_train_freq: int,
        ensemble_size: int,
        model_batch_size: int,
        real_ratio: float,
        temperature: float,
        model_lr: float,
        model_hidden_size: int,
        model_num_layers: int,
        syn_ratio: Optional[float] = None,
        rollout_horizon_min: Optional[int] = None,
        rollout_horizon_max: Optional[int] = None,
        rollout_horizon_ramp_steps: int = 0,
        reward_loss_coef: float = 1.0,
        logvar_loss_coef: float = 1.0,  # scaling coefficient for log-variance term in NLL loss
        state_normalize: bool = False,
        reward_normalize: bool = False,
        reward_norm_eps: float = 1e-6,
        normalization_stats_json: Optional[str] = None,
        separate_reward_net: bool = False,
        training_iterations: int = 0,
        centralized_dynamics: bool = False,
        stochastic_dynamics: bool = True,
        n_elites: Optional[int] = None,
        min_log_var: float = -10.0,
        max_log_var: float = -2.0,
        warmup_steps: int = 0,
        rollout_uncertainty_threshold: Optional[float] = None,
        rollout_uncertainty_thresholds: Optional[Dict[str, float]] = None,
        rollout_uncertainty_metric: str = "total_rew_unc",
        use_oracle: bool = False,
        load_world_model_path: Optional[str] = None,
        load_world_model_strict: bool = True,
        save_world_model_path: Optional[str] = None,
        save_world_model_interval: Optional[int] = None,
        oracle_reward: bool = False,
        oracle_reward_mode: str = "goal_distance_from_obs",
        oracle_reward_goal_rel_indices: Optional[List[int]] = None,
        oracle_reward_scale: float = 1.0,
        oracle_reward_disable_reward_head_loss: bool = True,
        world_model_debug_video: bool = False,
        world_model_debug_video_interval: int = 200,
        world_model_debug_video_horizon: int = 50,
        world_model_debug_video_fps: int = 20,
        world_model_debug_video_env_index: int = 0,
        world_model_debug_video_obs_xy_indices: Optional[List[int]] = None,
        world_model_debug_video_mode: str = "open_loop",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mbpo_init(
            rollout_horizon=rollout_horizon,
            rollout_horizon_min=rollout_horizon_min,
            rollout_horizon_max=rollout_horizon_max,
            rollout_horizon_ramp_steps=rollout_horizon_ramp_steps,
            model_train_freq=model_train_freq,
            ensemble_size=ensemble_size,
            model_batch_size=model_batch_size,
            real_ratio=real_ratio,
            temperature=temperature,
            model_lr=model_lr,
            model_hidden_size=model_hidden_size,
            model_num_layers=model_num_layers,
            syn_ratio=syn_ratio,
            reward_loss_coef=reward_loss_coef,
            logvar_loss_coef=logvar_loss_coef,
            state_normalize=state_normalize,
            reward_normalize=reward_normalize,
            reward_norm_eps=reward_norm_eps,
            normalization_stats_json=normalization_stats_json,
            separate_reward_net=separate_reward_net,
            training_iterations=training_iterations,
            centralized_dynamics=centralized_dynamics,
            stochastic_dynamics=stochastic_dynamics,
            n_elites=n_elites,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
            warmup_steps=warmup_steps,
            rollout_uncertainty_threshold=rollout_uncertainty_threshold,
            rollout_uncertainty_thresholds=rollout_uncertainty_thresholds,
            rollout_uncertainty_metric=rollout_uncertainty_metric,
            use_oracle=use_oracle,
            load_world_model_path=load_world_model_path,
            load_world_model_strict=load_world_model_strict,
            save_world_model_path=save_world_model_path,
            save_world_model_interval=save_world_model_interval,
            oracle_reward=oracle_reward,
            oracle_reward_mode=oracle_reward_mode,
            oracle_reward_goal_rel_indices=oracle_reward_goal_rel_indices,
            oracle_reward_scale=oracle_reward_scale,
            oracle_reward_disable_reward_head_loss=oracle_reward_disable_reward_head_loss,
            world_model_debug_video=world_model_debug_video,
            world_model_debug_video_interval=world_model_debug_video_interval,
            world_model_debug_video_horizon=world_model_debug_video_horizon,
            world_model_debug_video_fps=world_model_debug_video_fps,
            world_model_debug_video_env_index=world_model_debug_video_env_index,
            world_model_debug_video_obs_xy_indices=world_model_debug_video_obs_xy_indices,
            world_model_debug_video_mode=world_model_debug_video_mode,
        )

# Backwards-compatible name: default MBPO remains MASAC-based.
Mbpo = MbpoMasac


@dataclass
class MbpoConfig(MasacConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.MbpoMasac`."""

    rollout_horizon: int = MISSING
    model_train_freq: int = MISSING
    ensemble_size: int = MISSING
    model_batch_size: int = MISSING
    real_ratio: float = MISSING
    temperature: float = MISSING
    model_lr: float = MISSING
    model_hidden_size: int = MISSING
    model_num_layers: int = MISSING
    syn_ratio: Optional[float] = None
    rollout_horizon_min: Optional[int] = None
    rollout_horizon_max: Optional[int] = None
    rollout_horizon_ramp_steps: int = 0
    reward_loss_coef: float = 1.0
    state_normalize: bool = False
    reward_normalize: bool = False
    reward_norm_eps: float = 1e-6
    normalization_stats_json: Optional[str] = None
    use_oracle: bool = False
    separate_reward_net: bool = False
    training_iterations: int = 0
    centralized_dynamics: bool = False
    stochastic_dynamics: bool = True
    n_elites: Optional[int] = None
    min_log_var: float = -10.0
    max_log_var: float = -2.0
    logvar_loss_coef: float = 1.0
    warmup_steps: int = 0
    rollout_uncertainty_threshold: Optional[float] = None
    rollout_uncertainty_thresholds: Optional[Dict[str, float]] = None
    rollout_uncertainty_metric: str = "total_rew_unc"
    load_world_model_path: Optional[str] = None
    load_world_model_strict: bool = True
    save_world_model_path: Optional[str] = None
    save_world_model_interval: Optional[int] = None
    oracle_reward: bool = False
    oracle_reward_mode: str = "goal_distance_from_obs"
    oracle_reward_goal_rel_indices: Optional[List[int]] = None
    oracle_reward_scale: float = 1.0
    oracle_reward_disable_reward_head_loss: bool = True
    # Debug visualization: log a video comparing world-model predictions vs simulator transitions.
    world_model_debug_video: bool = False
    world_model_debug_video_interval: int = 200
    world_model_debug_video_horizon: int = 50
    world_model_debug_video_fps: int = 20
    world_model_debug_video_env_index: int = 0
    # Indices of obs dims to interpret as (x,y) for plotting; if None, uses the first two dims.
    world_model_debug_video_obs_xy_indices: Optional[List[int]] = None
    # 'open_loop' compares 1-step predictions from the simulator state (same start state each frame).
    # 'closed_loop' rolls out the world model by feeding its own predicted state back in.
    world_model_debug_video_mode: str = "open_loop"

    @staticmethod
    def associated_class() -> Type[Masac]:
        return MbpoMasac

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        return True


@dataclass
class MbpoMappoConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.MbpoMappo`."""

    rollout_horizon: int = MISSING
    model_train_freq: int = MISSING
    ensemble_size: int = MISSING
    model_batch_size: int = MISSING
    real_ratio: float = MISSING
    temperature: float = MISSING
    model_lr: float = MISSING
    model_hidden_size: int = MISSING
    model_num_layers: int = MISSING
    syn_ratio: Optional[float] = None
    rollout_horizon_min: Optional[int] = None
    rollout_horizon_max: Optional[int] = None
    rollout_horizon_ramp_steps: int = 0
    reward_loss_coef: float = 1.0
    state_normalize: bool = False
    reward_normalize: bool = False
    reward_norm_eps: float = 1e-6
    normalization_stats_json: Optional[str] = None
    use_oracle: bool = False
    separate_reward_net: bool = False
    training_iterations: int = 0
    centralized_dynamics: bool = False
    stochastic_dynamics: bool = True
    n_elites: Optional[int] = None
    min_log_var: float = -10.0
    max_log_var: float = -2.0
    logvar_loss_coef: float = 1.0
    warmup_steps: int = 0
    rollout_uncertainty_threshold: Optional[float] = None
    rollout_uncertainty_thresholds: Optional[Dict[str, float]] = None
    rollout_uncertainty_metric: str = "total_rew_unc"
    load_world_model_path: Optional[str] = None
    load_world_model_strict: bool = True
    save_world_model_path: Optional[str] = None
    save_world_model_interval: Optional[int] = None
    oracle_reward: bool = False
    oracle_reward_mode: str = "goal_distance_from_obs"
    oracle_reward_goal_rel_indices: Optional[List[int]] = None
    oracle_reward_scale: float = 1.0
    oracle_reward_disable_reward_head_loss: bool = True
    world_model_debug_video: bool = False
    world_model_debug_video_interval: int = 200
    world_model_debug_video_horizon: int = 50
    world_model_debug_video_fps: int = 20
    world_model_debug_video_env_index: int = 0
    world_model_debug_video_obs_xy_indices: Optional[List[int]] = None
    world_model_debug_video_mode: str = "open_loop"

    @staticmethod
    def associated_class() -> Type[Mappo]:
        return MbpoMappo

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True

