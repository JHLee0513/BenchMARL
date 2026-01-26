"""
Recurrent MBPO world model (GRU) with state history input.

This module implements a recurrent variant of MBPO that augments the world model input with
an observation history window of length (history_length + 1).

- history_length = 0: identical behavior to standard MBPO (feed-forward dynamics model).
- history_length > 0: the world model uses a GRU encoder over the observation sequence and
  predicts the next transition from the encoded hidden state + current action.

To avoid over-complicating `mbpo.py`, this is implemented as a separate algorithm module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from tensordict import TensorDict, TensorDictBase

from benchmarl.algorithms.mappo import Mappo, MappoConfig
from benchmarl.algorithms.masac import Masac, MasacConfig

# Reuse the existing MBPO implementation and override only world-model parts.
from benchmarl.algorithms.mbpo import (  # noqa: F401
    _DynamicsModel,
    _DynamicsTrainBuffer,
    _MbpoWorldModelMixin,
)


class _RecurrentDynamicsTrainBuffer(_DynamicsTrainBuffer):
    """Recurrent variant of _DynamicsTrainBuffer that supports time dimensions.
    
    Handles data with time dimensions:
    - For shards: (batch, max_horizon_len, n_agent, dim)
    - For input: (history_length, Batch size, n_agents, dim)
    - For next observation, reward, done, etc.: (forecast_length, Batch size, n_agents, dim)
    """

    def __init__(self, capacity: int, device: torch.device, history_length: int = 0, forecast_length: int = 1):
        super().__init__(capacity, device)
        self.history_length = max(0, int(history_length))
        self.forecast_length = max(1, int(forecast_length))

    def extend(self, td: TensorDictBase) -> None:
        """Extend buffer with recurrent-friendly data.

        This buffer accepts:
        1) Raw simulator trajectory batches with time dim (TorchRL-style):
           - (group,"observation"): [T, B, n_agents, obs_dim]
           - (group,"action"):      [T, B, n_agents, ...]
           - ("next",group,"observation"): [T, B, n_agents, obs_dim]
           - ("next",group,"reward") / ("next",group,"done") (or top-level next reward/done)
           and converts them into a flat recurrent format.

        2) Already-processed recurrent batches (e.g., saved datasets) containing:
           - (group,"observation_history"): [N, L, n_agents, obs_dim]
           - ("next_recurrent",group,"observation"): [N, L', n_agents, obs_dim] (preferred)

        Output format stored in the buffer uses batch_size [N] (agent is a tensor dim):
          - (group,"observation_history"): [N, L, n_agents, obs_dim]
          - (group,"observation"): [N, n_agents, obs_dim]
          - (group,"action"): [N, n_agents, ...]
          - (group,"next_observation"): [N, L', n_agents, obs_dim] (if forecast_length > 0)
          - (group,"next_reward"), (group,"next_done"): [N, L', n_agents, ...]
        """
        if td is None or td.numel() == 0:
            return
        td = td.detach()
        keys = td.keys(True, True)

        groups = self._infer_groups(keys)
        if not groups:
            return

        td_processed = self._preprocess(td, keys=keys, groups=groups)
        if td_processed is None or td_processed.numel() == 0:
            return

        # Ensure leading dimension exists
        if td_processed.batch_size is None or len(td_processed.batch_size) == 0:
            td_processed = td_processed.reshape(-1)

        # Call parent extend with processed data
        super().extend(td_processed)

    def _infer_groups(self, keys) -> set:
        """Infer group names from tensordict keys."""
        groups = set()
        for k in keys:
            if isinstance(k, tuple) and len(k) == 2 and k[1] == "observation":
                groups.add(k[0])
        # Keep only groups that look like proper agent groups (have action + next obs).
        out = set()
        for g in groups:
            if (g, "action") in keys and ("next", g, "observation") in keys:
                out.add(g)
        return out

    def _preprocess(
        self, td: TensorDictBase, *, keys, groups: set
    ) -> Optional[TensorDictBase]:
        """Return a recurrent-ready flat batch (batch_size [N]) or None if unsupported."""
        has_observation_history = any(
            isinstance(k, tuple) and len(k) == 2 and k[1] == "observation_history" for k in keys
        )
        if has_observation_history:
            return self._process_new_format(td, groups)

        # Raw sim trajectory with time dimension.
        if td.batch_size is not None and len(td.batch_size) >= 2:
            return self._process_trajectory_format(td, groups)

        # Flat transition batch without time dim: we can only support forecast_length==1
        # (otherwise we cannot build strict multi-step targets).
        if int(self.forecast_length) != 1:
            return None
        return self._process_flat_transition_format(td, groups)

    def _get_group_reward_done(
        self, td: TensorDictBase, *, group: str, keys
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Best-effort extraction of reward/done aligned with ('next', group, 'observation')."""
        rew = None
        done = None
        if ("next", group, "reward") in keys:
            try:
                rew = td.get(("next", group, "reward"))
            except Exception:
                rew = None
        elif ("next", "reward") in keys:
            try:
                r = td.get(("next", "reward"))
                lead = td.get(("next", group, "observation")).shape[:-1]
                rew = r.expand(*lead).unsqueeze(-1)
            except Exception:
                rew = None

        if ("next", group, "done") in keys:
            try:
                done = td.get(("next", group, "done"))
            except Exception:
                done = None
        elif ("next", "done") in keys:
            try:
                d = td.get(("next", "done"))
                lead = td.get(("next", group, "observation")).shape[:-1]
                done = d.expand(*lead).unsqueeze(-1)
            except Exception:
                done = None
        return rew, done

    def _process_flat_transition_format(
        self, td: TensorDictBase, groups: set
    ) -> Optional[TensorDictBase]:
        """Process a flat transition batch (no time dim) into recurrent format (forecast_length==1 only)."""
        keys = td.keys(True, True)
        processed_dict: Dict = {}
        for group in groups:
            if (group, "observation") not in keys or (group, "action") not in keys:
                continue
            if ("next", group, "observation") not in keys:
                continue
            reward, done = self._get_group_reward_done(td, group=group, keys=keys)
            if reward is None or done is None:
                continue

            obs = td.get((group, "observation"))
            action = td.get((group, "action"))
            next_obs = td.get(("next", group, "observation"))
            # forecast_length == 1
            group_dict = {
                "observation": obs,
                "action": action,
                "observation_history": obs.unsqueeze(1).repeat_interleave(self.history_length + 1, dim=1),
                "next_observation": next_obs.unsqueeze(1),
                "next_reward": reward.unsqueeze(1) if reward.dim() == 3 else reward,
                "next_done": done.unsqueeze(1) if done.dim() == 3 else done,
            }
            processed_dict[group] = TensorDict(
                group_dict, batch_size=[int(obs.shape[0])], device=obs.device
            )

        if not processed_dict:
            return None
        n = int(list(processed_dict.values())[0].batch_size[0])
        return TensorDict(processed_dict, batch_size=[n], device=list(processed_dict.values())[0].device)

    def _process_trajectory_format(self, td: TensorDictBase, groups: set) -> Optional[TensorDictBase]:
        """Convert a time-major trajectory batch into a flat recurrent batch (strict future horizon)."""
        processed_dict: Dict = {}
        keys = td.keys(True, True)
        H = int(self.history_length)
        L_hist = H + 1
        L_fut = int(self.forecast_length)
        if L_fut < 1:
            L_fut = 1

        for group in groups:
            if (group, "observation") not in keys or (group, "action") not in keys:
                continue
            if ("next", group, "observation") not in keys:
                continue
            reward, done = self._get_group_reward_done(td, group=group, keys=keys)
            if reward is None or done is None:
                continue

            obs = td.get((group, "observation"))  # [T, B, n_agents, obs_dim]
            action = td.get((group, "action"))  # [T, B, n_agents, ...]
            next_obs = td.get(("next", group, "observation"))  # [T, B, n_agents, obs_dim]

            if obs.dim() < 4 or next_obs.dim() < 4:
                continue

            T = int(obs.shape[0])
            B = int(obs.shape[1])
            if T < L_hist or T < (H + L_fut):
                # Not enough length for strict future horizon.
                continue

            # Number of valid transition indices t: t in [H, T - L_fut]
            TH_valid = int(T - H - L_fut + 1)
            if TH_valid <= 0:
                continue

            # Observation history windows: aligned with transitions t=H..T-1 (then trimmed to TH_valid).
            obs_win = obs.unfold(0, L_hist, 1)  # [T-H, L_hist, B, n_agents, obs_dim]
            obs_win = obs_win[:TH_valid].permute(0, 2, 1, 3, 4).contiguous()  # [TH_valid, B, L_hist, n_agents, obs_dim]

            # Current action at time t (t=H..H+TH_valid-1).
            action_sel = action[H : H + TH_valid]  # [TH_valid, B, n_agents, ...]

            # Strict future targets: window of length L_fut starting at time t on "next" tensors.
            next_obs_win = next_obs.unfold(0, L_fut, 1)  # [T-L_fut+1, L_fut, B, n_agents, obs_dim]
            next_obs_win = next_obs_win[H : H + TH_valid].permute(0, 2, 1, 3, 4).contiguous()  # [TH_valid, B, L_fut, n_agents, obs_dim]

            # Reward/done windows aligned to time t on next tensors.
            def _permute_unfold(x: torch.Tensor) -> torch.Tensor:
                # x: [K, L, B, ...] -> [K, B, L, ...]
                if x.dim() < 3:
                    return x
                dims = [0, 2, 1] + list(range(3, x.dim()))
                return x.permute(*dims).contiguous()

            reward_win = reward.unfold(0, L_fut, 1)  # [T-L_fut+1, L_fut, B, ...]
            reward_win = _permute_unfold(reward_win[H : H + TH_valid])  # [TH_valid, B, L_fut, ...]

            done_win = done.unfold(0, L_fut, 1)  # [T-L_fut+1, L_fut, B, ...]
            done_win = _permute_unfold(done_win[H : H + TH_valid])  # [TH_valid, B, L_fut, ...]

            # Episode-boundary filtering:
            # - history is invalid if any done=True appears in the history window excluding current step
            # - future is invalid if any done=True appears in the future window excluding last step
            d_any = done.to(torch.bool)
            while d_any.dim() > 2:
                d_any = d_any.any(dim=-1)
            # d_any: [T, B]
            invalid = torch.zeros((TH_valid, B), device=obs.device, dtype=torch.bool)
            if H > 0:
                d_hist = d_any.unfold(0, L_hist, 1)  # [T-H, L_hist, B]
                d_hist = d_hist[:TH_valid]  # align with obs_win
                invalid |= d_hist[:, :H, :].any(dim=1)
            if L_fut > 1:
                d_fut = d_any.unfold(0, L_fut, 1)  # [T-L_fut+1, L_fut, B]
                d_fut = d_fut[H : H + TH_valid]
                invalid |= d_fut[:, : L_fut - 1, :].any(dim=1)

            valid_flat = (~invalid).reshape(-1)

            # Flatten time and batch dims.
            obs_hist_flat = obs_win.reshape(TH_valid * B, L_hist, *obs_win.shape[3:])
            action_flat = action_sel.reshape(TH_valid * B, *action_sel.shape[2:])
            next_obs_flat = next_obs_win.reshape(TH_valid * B, L_fut, *next_obs_win.shape[3:])
            reward_flat = reward_win.reshape(TH_valid * B, L_fut, *reward_win.shape[3:])
            done_flat = done_win.reshape(TH_valid * B, L_fut, *done_win.shape[3:])

            if valid_flat.numel() == obs_hist_flat.shape[0] and valid_flat.any():
                obs_hist_flat = obs_hist_flat[valid_flat]
                action_flat = action_flat[valid_flat]
                next_obs_flat = next_obs_flat[valid_flat]
                reward_flat = reward_flat[valid_flat]
                done_flat = done_flat[valid_flat]
            else:
                # Nothing valid for this group.
                continue

            group_dict = {
                "observation_history": obs_hist_flat.contiguous(),
                "observation": obs_hist_flat[:, -1].contiguous(),
                "action": action_flat.contiguous(),
                "next_observation": next_obs_flat.contiguous(),
                "next_reward": reward_flat.contiguous(),
                "next_done": done_flat.contiguous(),
            }
            processed_dict[group] = TensorDict(
                group_dict, batch_size=[int(obs_hist_flat.shape[0])], device=obs_hist_flat.device
            )

        if not processed_dict:
            return None
        n = int(list(processed_dict.values())[0].batch_size[0])
        return TensorDict(processed_dict, batch_size=[n], device=list(processed_dict.values())[0].device)
    
    def _process_new_format(self, td: TensorDictBase, groups: set) -> TensorDictBase:
        """Process new format with observation_history, action_history, and next_recurrent."""
        processed_dict = {}
        
        for group in groups:
            group_dict = {}
            keys = td.keys(True, True)
            
            # Get observation_history: [N, L, n_agents, obs_dim]
            obs_hist_key = (group, "observation_history")
            if obs_hist_key in keys:
                obs_hist = td.get(obs_hist_key)  # [N, L_data, n_agents, obs_dim]
                # Slice to match training history_length if needed
                L_data = obs_hist.shape[1]
                L_train = self.history_length + 1
                if L_data >= L_train:
                    # Take the last L_train steps
                    obs_hist = obs_hist[:, -L_train:]
                elif L_data < L_train:
                    # Pad by repeating the first observation
                    pad_len = L_train - L_data
                    first_obs = obs_hist[:, 0:1].expand(-1, pad_len, -1, -1)
                    obs_hist = torch.cat([first_obs, obs_hist], dim=1)
                
                group_dict["observation"] = obs_hist[:, -1]  # Current observation
                group_dict["observation_history"] = obs_hist
            
            # Get action_history if available: [N, L, n_agents, action_dim]
            action_hist_key = (group, "action_history")
            if action_hist_key in keys:
                action_hist = td.get(action_hist_key)
                # Slice to match training history_length if needed
                L_data = action_hist.shape[1]
                L_train = self.history_length + 1
                if L_data >= L_train:
                    action_hist = action_hist[:, -L_train:]
                elif L_data < L_train:
                    pad_len = L_train - L_data
                    first_action = action_hist[:, 0:1].expand(-1, pad_len, -1, -1)
                    action_hist = torch.cat([first_action, action_hist], dim=1)
                # Store for potential future use (currently model uses current action)
                group_dict["action_history"] = action_hist
            
            # Get current action: [N, n_agents, action_dim]
            action_key = (group, "action")
            if action_key in keys:
                group_dict["action"] = td.get(action_key)
            
            # Get next_recurrent if available: [N, L', n_agents, ...]
            next_recurrent_key = ("next_recurrent", group, "observation")
            if next_recurrent_key in keys:
                next_obs_future = td.get(next_recurrent_key)  # [N, L_data, n_agents, obs_dim]
                # Slice to match training future_length if needed
                L_data = next_obs_future.shape[1]
                L_train = self.forecast_length
                if L_data >= L_train:
                    next_obs_future = next_obs_future[:, :L_train]
                elif L_data < L_train:
                    # Strict future horizon: do NOT pad.
                    raise ValueError(
                        f"Insufficient next_recurrent horizon for group={group}: "
                        f"have {L_data}, need {L_train}. Recollect data with strict future windows."
                    )
                group_dict["next_observation"] = next_obs_future
                
                # Get future reward and done
                reward_future_key = ("next_recurrent", group, "reward")
                done_future_key = ("next_recurrent", group, "done")
                if reward_future_key in keys:
                    reward_future = td.get(reward_future_key)
                    L_data = reward_future.shape[1]
                    if L_data >= L_train:
                        reward_future = reward_future[:, :L_train]
                    elif L_data < L_train:
                        raise ValueError(
                            f"Insufficient next_recurrent reward horizon for group={group}: "
                            f"have {L_data}, need {L_train}. Recollect data with strict future windows."
                        )
                    group_dict["next_reward"] = reward_future
                
                if done_future_key in keys:
                    done_future = td.get(done_future_key)
                    L_data = done_future.shape[1]
                    if L_data >= L_train:
                        done_future = done_future[:, :L_train]
                    elif L_data < L_train:
                        raise ValueError(
                            f"Insufficient next_recurrent done horizon for group={group}: "
                            f"have {L_data}, need {L_train}. Recollect data with strict future windows."
                        )
                    group_dict["next_done"] = done_future
            else:
                # Fallback to single-step next data
                if int(self.forecast_length) != 1:
                    raise ValueError(
                        f"Missing next_recurrent for group={group} but forecast_length={self.forecast_length}. "
                        "Cannot build strict multi-step targets from single-step next data."
                    )
                next_obs_key = ("next", group, "observation")
                next_reward_key = ("next", group, "reward")
                next_done_key = ("next", group, "done")
                if next_obs_key in keys:
                    next_obs = td.get(next_obs_key)  # [N, n_agents, obs_dim]
                    # Expand to [N, 1, n_agents, obs_dim] to match multi-step format
                    next_obs = next_obs.unsqueeze(1)
                    group_dict["next_observation"] = next_obs
                if next_reward_key in keys:
                    reward = td.get(next_reward_key)
                    reward = reward.unsqueeze(1) if reward.dim() == 2 else reward
                    group_dict["next_reward"] = reward
                if next_done_key in keys:
                    done = td.get(next_done_key)
                    done = done.unsqueeze(1) if done.dim() == 2 else done
                    group_dict["next_done"] = done
            
            if group_dict:
                processed_dict[group] = TensorDict(group_dict, batch_size=[group_dict["observation"].shape[0]])
        
        if processed_dict:
            return TensorDict(processed_dict, batch_size=[list(processed_dict.values())[0].batch_size[0]])
        return td


    def sample(self, batch_size: int, *, generator: Optional[torch.Generator]) -> TensorDictBase:
        """Sample a batch, preserving time dimensions for recurrent data."""
        if self._storage is None or self._len <= 0:
            raise RuntimeError("RecurrentDynamicsTrainBuffer is empty")
        batch_size = max(1, int(batch_size))
        # Sample indices using the provided generator on CPU to avoid global RNG
        idx = torch.randint(
            0,
            self._len,
            (batch_size,),
            device="cpu",
            generator=generator,
        ).to(self.device)
        return self._storage[idx]


class _RecurrentDynamicsModel(nn.Module):
    """GRU-based dynamics model.

    The GRU encodes an observation history window. The final hidden state is concatenated
    with the current action and fed to MLP heads to predict:
      - next observation mean (via delta to last observation in the window)
      - (optionally) observation log-variance
      - reward mean and log-variance
      - done logits
    """

    def __init__(
        self,
        *,
        obs_input_dim: int,
        action_dim: int,
        next_obs_dim: int,
        reward_dim: int,
        done_dim: int,
        hidden_size: int,
        mlp_num_layers: int,
        gru_num_layers: int,
        stochastic: bool,
        min_log_var: float,
        max_log_var: float,
        separate_reward_net: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(obs_input_dim),
            hidden_size=int(hidden_size),
            num_layers=max(1, int(gru_num_layers)),
            batch_first=True,
        )

        trunk_in = int(hidden_size) + int(action_dim)

        layers: List[nn.Module] = []
        last_dim = trunk_in
        for _ in range(max(0, int(mlp_num_layers))):
            layers += [nn.Linear(last_dim, int(hidden_size)), nn.ReLU()]
            last_dim = int(hidden_size)
        self.net = nn.Sequential(*layers)

        self._stochastic = bool(stochastic)
        self._separate_reward_net = bool(separate_reward_net)
        self.register_buffer("_min_log_var", torch.tensor(float(min_log_var)))
        self.register_buffer("_max_log_var", torch.tensor(float(max_log_var)))

        self.delta_mu_head = nn.Linear(last_dim, int(next_obs_dim))
        self.delta_log_var_head = nn.Linear(last_dim, int(next_obs_dim))

        if self._separate_reward_net:
            r_layers: List[nn.Module] = []
            r_last = trunk_in
            for _ in range(max(0, int(mlp_num_layers))):
                r_layers += [nn.Linear(r_last, int(hidden_size)), nn.ReLU()]
                r_last = int(hidden_size)
            self.reward_net = nn.Sequential(*r_layers)
            reward_last_dim = r_last
        else:
            self.reward_net = None
            reward_last_dim = last_dim

        self.rew_mu_head = nn.Linear(reward_last_dim, int(reward_dim))
        self.rew_log_var_head = nn.Linear(reward_last_dim, int(reward_dim))
        self.done_head = nn.Linear(last_dim, int(done_dim))

    def forward(
        self, obs_seq_flat: torch.Tensor, action_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs_seq_flat: [B, L, obs_dim]
        # action_flat:  [B, act_dim]
        if obs_seq_flat.dim() != 3:
            raise ValueError(
                f"Expected obs_seq_flat with shape [B, L, D], got {tuple(obs_seq_flat.shape)}"
            )
        # Use last hidden state from the last layer as representation.
        _, h_n = self.gru(obs_seq_flat)  # h_n: [num_layers, B, hidden]
        h_last = h_n[-1]  # [B, hidden]

        x = torch.cat([h_last, action_flat], dim=-1)
        feat = self.net(x) if len(self.net) > 0 else x

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

        # Predict next obs as (last obs in window) + delta
        last_obs = obs_seq_flat[:, -1, :]
        mu_next_obs = last_obs + delta_mu
        return mu_next_obs, delta_log_var, rew_mu, rew_log_var, done_logit


class _MbpoRecurrentWorldModelMixin(_MbpoWorldModelMixin):
    """Overrides MBPO world-model to support `history_length` and GRU dynamics."""

    def _mbpo_init(
        self,
        *,
        history_length: int = 0,
        future_length: int = 1,
        gru_num_layers: int = 1,
        **kwargs
    ) -> None:
        # Set attributes BEFORE calling parent init, since parent calls _get_dynamics_training_buffer()
        # which needs these attributes.
        self.history_length = max(0, int(history_length))
        self.future_length = max(1, int(future_length))
        self.world_model_gru_num_layers = max(1, int(gru_num_layers or 1))

        # IMPORTANT:
        # MBPO's _mbpo_init() will immediately call load_world_model(load_world_model_path, ...)
        # after constructing the (feed-forward) `_DynamicsModel` ensemble.
        #
        # For recurrent MBPO (history_length > 0) we must first swap the ensemble to
        # `_RecurrentDynamicsModel`, otherwise loading a recurrent checkpoint will fail
        # with unexpected `gru.*` keys / shape mismatches.
        load_world_model_path = kwargs.pop("load_world_model_path", None)
        load_world_model_strict = bool(kwargs.pop("load_world_model_strict", True))

        # Run the standard MBPO initialization but defer world-model loading until the
        # correct dynamics model class is instantiated.
        super()._mbpo_init(load_world_model_path=None, load_world_model_strict=load_world_model_strict, **kwargs)

        # If no history is requested, keep default feed-forward world model and load now.
        if self.history_length <= 0:
            if load_world_model_path is not None:
                self.load_world_model(load_world_model_path, strict=load_world_model_strict)
            return

        # Replace dynamics ensemble with recurrent models.
        for group in self.group_map.keys():
            obs_shape = self.observation_spec[group, "observation"].shape
            n_agents = obs_shape[0]
            obs_dim = int(math.prod(obs_shape[1:]))

            action_space = self.action_spec[group, "action"]
            if hasattr(action_space, "space"):
                if hasattr(action_space.space, "n"):
                    action_dim = int(action_space.space.n)
                else:
                    action_dim = int(math.prod(action_space.shape[1:]))
            else:
                action_dim = int(math.prod(action_space.shape[1:]))

            if self.centralized_dynamics:
                obs_in_dim = n_agents * obs_dim
                act_in_dim = n_agents * action_dim
                next_obs_dim_out = n_agents * obs_dim
                reward_dim_out = n_agents
                done_dim_out = n_agents
            else:
                obs_in_dim = obs_dim
                act_in_dim = action_dim
                next_obs_dim_out = obs_dim
                reward_dim_out = 1
                done_dim_out = 1

            models: List[nn.Module] = []
            opts: List[torch.optim.Optimizer] = []
            for _ in range(self.ensemble_size):
                model = _RecurrentDynamicsModel(
                    obs_input_dim=obs_in_dim,
                    action_dim=act_in_dim,
                    next_obs_dim=next_obs_dim_out,
                    reward_dim=reward_dim_out,
                    done_dim=done_dim_out,
                    hidden_size=self.model_hidden_size,
                    mlp_num_layers=self.model_num_layers,
                    gru_num_layers=self.world_model_gru_num_layers,
                    stochastic=self.stochastic_dynamics,
                    min_log_var=self.min_log_var,
                    max_log_var=self.max_log_var,
                    separate_reward_net=getattr(self, "separate_reward_net", False),
                ).to(self.device)
                models.append(model)
                opts.append(torch.optim.Adam(model.parameters(), lr=self.model_lr))

            self._dynamics[group] = models  # type: ignore[assignment]
            self._dyn_optimizers[group] = opts

        # Now that recurrent dynamics models exist, load the checkpoint (if requested).
        if load_world_model_path is not None:
            self.load_world_model(load_world_model_path, strict=load_world_model_strict)

    def _get_dynamics_training_buffer(self, memory_size: int) -> _RecurrentDynamicsTrainBuffer:
        """Override to return recurrent buffer with history and future length."""
        return _RecurrentDynamicsTrainBuffer(
            capacity=memory_size,
            device=torch.device(self.device),
            history_length=self.history_length,
            forecast_length=self.future_length,
        )

    def _history_key(self) -> str:
        return "observation_history"

    def _ensure_history_in_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        """Ensure `(group, observation_history)` exists when history_length > 0.

        Supported cases:
        1) The batch is already in recurrent dataset format (e.g., loaded from a saved dataset)
           and contains `(group, "observation_history")`.
           - If length mismatches `history_length + 1`, we pad/slice deterministically.

        2) The batch is raw simulation/collector trajectory data (two leading dims) and does
           NOT contain history. Important: **the first leading dim is NOT time**; it is the
           number of parallel environments (or parallel episode streams). The second leading
           dim is the rollout length. True episode boundaries must be recovered by splitting
           each environment stream using `done`.
           In this case we build contiguous episode windows using `done` (mirroring
           `benchmarl/collect.py`'s recurrence acquisition), and return a flattened TensorDict
           with `batch_size=[N]` containing history.

        Any other format is invalid and will raise (no silent repetition/padding of states).
        """
        if self.history_length <= 0:
            return batch

        keys = batch.keys(True, True)
        hk = (group, self._history_key())
        H = int(self.history_length)
        L_required = H + 1
        
        # Case 1: History exists - validate and adjust length if needed
        if hk in keys:
            obs_hist = batch.get(hk)  # [N, L_data, n_agents, obs_dim] or [N, L_data, ...]
            
            # Get current length (handle different shapes)
            if obs_hist.dim() >= 2:
                L_data = obs_hist.shape[1]
            else:
                # Invalid shape, will be handled by Case 3
                L_data = 0
            
            if L_data == L_required:
                # Correct length, return as-is
                return batch
            elif L_data > 0:
                # Wrong length - adjust it
                if L_data > L_required:
                    # Take the last L_required steps
                    obs_hist = obs_hist[:, -L_required:]
                else:
                    # Pad by repeating the first observation
                    pad_len = L_required - L_data
                    first_obs = obs_hist[:, 0:1]  # [N, 1, ...]
                    # Expand to [N, pad_len, ...]
                    if obs_hist.dim() == 4:
                        first_obs = first_obs.expand(-1, pad_len, -1, -1)
                    else:
                        # Handle other dimensions
                        expand_shape = [-1, pad_len] + [-1] * (obs_hist.dim() - 2)
                        first_obs = first_obs.expand(*expand_shape)
                    obs_hist = torch.cat([first_obs, obs_hist], dim=1)
                
                # Update the batch with corrected history
                try:
                    batch.get(group).set(self._history_key(), obs_hist)
                except Exception:
                    # If group TD isn't writable, create a shallow copy
                    out = batch.clone(False)
                    out.get(group).set(self._history_key(), obs_hist)
                    batch = out
                
                # Also update current observation to be the last in history
                if (group, "observation") in keys:
                    current_obs = obs_hist[:, -1]  # [N, n_agents, obs_dim] or [N, ...]
                    try:
                        batch.get(group).set("observation", current_obs)
                    except Exception:
                        if 'out' not in locals():
                            out = batch.clone(False)
                        out.get(group).set("observation", current_obs)
                        batch = out
                
                return batch

        # Case 2: raw simulation trajectory (two leading dims) -> build windows like collect.py.
        # If we don't have a time dimension, we *cannot* recover recurrent history online.
        if batch.batch_size is None or len(batch.batch_size) < 2:
            raise ValueError(
                f"Missing {(group, self._history_key())} and no time dimension to build it. "
                f"Expected either a recurrent dataset batch with {(group, self._history_key())} "
                f"or a trajectory batch with two leading dims like [B,T] (or [T,B]) plus ('next',*,done). "
                f"Got batch_size={getattr(batch, 'batch_size', None)} and keys={list(keys)[:25]}..."
            )

        if (group, "observation") not in keys or (group, "action") not in keys or ("next", group, "observation") not in keys:
            raise ValueError(
                f"Cannot build history for group={group}. Missing required keys among: "
                f"{(group,'observation')}, {(group,'action')}, {('next',group,'observation')}."
            )

        obs_raw = batch.get((group, "observation"))
        action_raw = batch.get((group, "action"))
        next_obs_raw = batch.get(("next", group, "observation"))

        # Reward/done can be stored at group-level or top-level. Expand to match next_obs leading dims.
        def _get_group_reward_done_like_next_obs() -> Tuple[torch.Tensor, torch.Tensor]:
            rew = None
            done = None
            if ("next", group, "reward") in keys:
                try:
                    rew = batch.get(("next", group, "reward"))
                except Exception:
                    rew = None
            elif ("next", "reward") in keys:
                try:
                    r = batch.get(("next", "reward"))
                    lead = next_obs_raw.shape[:-1]
                    rew = r.expand(*lead).unsqueeze(-1)
                except Exception:
                    rew = None

            if ("next", group, "done") in keys:
                try:
                    done = batch.get(("next", group, "done"))
                except Exception:
                    done = None
            elif ("next", "done") in keys:
                try:
                    d = batch.get(("next", "done"))
                    lead = next_obs_raw.shape[:-1]
                    done = d.expand(*lead).unsqueeze(-1)
                except Exception:
                    done = None

            if rew is None or done is None:
                raise ValueError(
                    f"Cannot build history for group={group}: missing reward/done. "
                    f"Expected ('next',{group},'reward'/'done') or ('next','reward'/'done')."
                )
            return rew, done

        reward_raw, done_raw = _get_group_reward_done_like_next_obs()

        # We implement the windowing logic from collect.py, which operates on tensors shaped [B, T, ...]
        # where B is the number of parallel envs (NOT time), and T is the rollout length along which we
        # split into episodes using `done`.
        # Support both [B, T, ...] and [T, B, ...] by trying both and picking the first that yields windows.
        def _try_build_windows_from_bt(
            obs_bt: torch.Tensor,
            action_bt: torch.Tensor,
            next_obs_bt: torch.Tensor,
            reward_bt: torch.Tensor,
            done_bt: torch.Tensor,
        ) -> Optional[TensorDict]:
            if obs_bt.dim() < 4:
                return None
            if obs_bt.shape[1] < L_required:
                return None

            # done_any: [B, T] (per-env termination stream over rollout axis)
            d_any = done_bt.squeeze(-1).to(torch.bool)
            while d_any.dim() > 2:
                d_any = d_any.any(dim=-1)
            if d_any.dim() != 2:
                return None

            all_windows = []
            B = int(d_any.shape[0])
            T = int(d_any.shape[1])

            for b in range(B):
                done1d = d_any[b]  # [T]
                ends = (done1d == True).nonzero(as_tuple=False).flatten().tolist()
                starts = [0] + [i + 1 for i in ends]
                ends = ends + [T - 1]

                for s, e in zip(starts, ends):
                    if s > e or s >= obs_bt.shape[1]:
                        continue
                    traj_len = int(e - s + 1)
                    if traj_len < L_required:
                        continue

                    traj_obs = obs_bt[b, s : e + 1]
                    traj_action = action_bt[b, s : e + 1]
                    traj_next_obs = next_obs_bt[b, s : e + 1]
                    traj_reward = reward_bt[b, s : e + 1]
                    traj_done = done_bt[b, s : e + 1]

                    # Create sliding windows (start where we have enough history).
                    for t in range(L_required - 1, traj_len):
                        if H > 0:
                            history_done = traj_done[max(0, t - L_required + 1) : t]
                            if history_done.numel() > 0 and history_done.any():
                                continue

                        hist_start = max(0, t - L_required + 1)
                        obs_hist = traj_obs[hist_start : t + 1]

                        # Pad history if needed (shouldn't happen due to t>=L-1, but keep deterministic behavior).
                        if obs_hist.shape[0] < L_required:
                            pad_len = L_required - int(obs_hist.shape[0])
                            first_obs = obs_hist[0:1].expand(pad_len, *obs_hist.shape[1:])
                            obs_hist = torch.cat([first_obs, obs_hist], dim=0)

                        curr_action = traj_action[t]

                        # Mirror collect.py's single-step extraction: use t+1 when available.
                        future_start = t + 1
                        if future_start < traj_len:
                            next_obs_single = traj_next_obs[future_start]
                            reward_single = traj_reward[future_start]
                            done_single = traj_done[future_start]
                        else:
                            next_obs_single = traj_obs[-1]
                            reward_single = traj_reward[-1]
                            done_single = traj_done[-1]

                        all_windows.append(
                            {
                                "obs_hist": obs_hist,
                                "action": curr_action,
                                "next_obs_single": next_obs_single,
                                "reward_single": reward_single,
                                "done_single": done_single,
                            }
                        )

            if not all_windows:
                return None

            N = len(all_windows)
            device = all_windows[0]["obs_hist"].device
            obs_hist_stack = torch.stack([w["obs_hist"] for w in all_windows], dim=0).contiguous()
            action_stack = torch.stack([w["action"] for w in all_windows], dim=0).contiguous()
            next_obs_stack = torch.stack([w["next_obs_single"] for w in all_windows], dim=0).contiguous()
            reward_stack = torch.stack([w["reward_single"] for w in all_windows], dim=0).contiguous()
            done_stack = torch.stack([w["done_single"] for w in all_windows], dim=0).contiguous()

            out = TensorDict({}, batch_size=[N], device=device)
            out.set(
                group,
                TensorDict(
                    {
                        "observation": obs_hist_stack[:, -1].contiguous(),
                        self._history_key(): obs_hist_stack,
                        "action": action_stack,
                    },
                    batch_size=[N],
                    device=device,
                ),
            )
            out.set(
                "next",
                TensorDict(
                    {
                        group: TensorDict(
                            {"observation": next_obs_stack, "reward": reward_stack, "done": done_stack},
                            batch_size=[N],
                            device=device,
                        )
                    },
                    batch_size=[N],
                    device=device,
                ),
            )
            return out

        # Attempt 1: assume already [B, T, ...]
        out_bt = _try_build_windows_from_bt(obs_raw, action_raw, next_obs_raw, reward_raw, done_raw)
        if out_bt is not None:
            return out_bt

        # Attempt 2: assume [T, B, ...] -> transpose to [B, T, ...]
        def _swap_tb(x: torch.Tensor) -> torch.Tensor:
            if x.dim() < 2:
                return x
            dims = list(range(x.dim()))
            dims[0], dims[1] = dims[1], dims[0]
            return x.permute(*dims).contiguous()

        out_tb = _try_build_windows_from_bt(
            _swap_tb(obs_raw), _swap_tb(action_raw), _swap_tb(next_obs_raw), _swap_tb(reward_raw), _swap_tb(done_raw)
        )
        if out_tb is not None:
            return out_tb

        raise ValueError(
            f"Unable to build {(group, self._history_key())} from trajectory batch for group={group}. "
            f"Need at least L={L_required} contiguous steps per episode segment and valid done/reward tensors. "
            f"obs_shape={tuple(obs_raw.shape)}, done_shape={tuple(done_raw.shape)}, batch_size={batch.batch_size}."
        )

    def _flatten_inputs_for_world_model(
        self, group: str, batch: TensorDictBase
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return (obs_seq_flat, last_obs_flat, action_flat, next_obs_flat, reward_flat, done_flat, is_multi_step).
        
        Returns is_multi_step=True if next_obs_flat has time dimension (multi-step future).
        """
        batch = self._ensure_history_in_batch(group, batch)
        keys = batch.keys(True, True)

        obs = batch.get((group, "observation")).to(self.device)
        action = batch.get((group, "action")).to(self.device)
        
        # Check for multi-step future data (next_recurrent) or single-step (next)
        is_multi_step = False
        next_obs_key = ("next", group, "observation")
        next_reward_key = ("next", group, "reward")
        next_done_key = ("next", group, "done")
        
        # Check if we have next_observation with time dimension (from buffer processing)
        if (group, "next_observation") in keys:
            next_obs = batch.get((group, "next_observation")).to(self.device)
            if next_obs.dim() == 4:  # [B, L', n_agents, obs_dim]
                is_multi_step = True
            else:
                next_obs = next_obs.unsqueeze(1)  # Add time dim
        elif next_obs_key in keys:
            next_obs = batch.get(next_obs_key).to(self.device)
            if next_obs.dim() == 3:  # [B, n_agents, obs_dim]
                next_obs = next_obs.unsqueeze(1)  # [B, 1, n_agents, obs_dim]
        else:
            raise ValueError(f"Next observation not found for group {group}")
        
        if (group, "next_reward") in keys:
            reward = batch.get((group, "next_reward")).to(self.device)
        elif next_reward_key in keys:
            reward = batch.get(next_reward_key).to(self.device)
            if reward.dim() == 2:  # [B, n_agents] or [B, 1]
                reward = reward.unsqueeze(1)
        else:
            raise ValueError(f"Next reward not found for group {group}")
        
        if (group, "next_done") in keys:
            done = batch.get((group, "next_done")).to(self.device)
        elif next_done_key in keys:
            done = batch.get(next_done_key).to(self.device)
            if done.dim() == 2:  # [B, n_agents] or [B, 1]
                done = done.unsqueeze(1)
        else:
            raise ValueError(f"Next done not found for group {group}")

        # Observation history: [B, L, n_agents, obs_dim] (or [B, L, ...])
        if self.history_length > 0:
            if (group, self._history_key()) not in keys:
                raise ValueError(
                    f"Missing {(group, self._history_key())} after _ensure_history_in_batch for group={group}. "
                    "This is a bug: history must be present or an error should have been raised earlier."
                )
            obs_hist = batch.get((group, self._history_key())).to(self.device)
        else:
            obs_hist = obs.unsqueeze(1)

        # Ensure expected shapes
        obs = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
        action = action.reshape(-1, action.shape[-2], *action.shape[-1:])

        encoded_action = self._encode_action(group, action)

        # obs_hist is expected to be [B, L, n_agents, obs_dim]
        if obs_hist.dim() == 3:
            # If somehow flattened: [B, n_agents, obs_dim] -> add L dim
            L = int(self.history_length) + 1
            obs_hist = obs_hist.unsqueeze(1).repeat_interleave(L, dim=1)
        if obs_hist.dim() != 4:
            raise ValueError(f"Unexpected obs_hist shape: {tuple(obs_hist.shape)}")

        if self.centralized_dynamics:
            # [B, L, n_agents*obs_dim]
            obs_seq_flat = obs_hist.reshape(obs_hist.shape[0], obs_hist.shape[1], -1)
            last_obs_flat = obs_seq_flat[:, -1, :]
            act_flat = encoded_action.reshape(encoded_action.shape[0], -1)
            
            # Handle multi-step or single-step next_obs
            if is_multi_step:
                # [B, L', n_agents*obs_dim]
                next_obs_flat = next_obs.reshape(next_obs.shape[0], next_obs.shape[1], -1)
            else:
                # [B, n_agents*obs_dim]
                next_obs_flat = next_obs.reshape(next_obs.shape[0], -1)
            
            # Reward and done: [B, L'] or [B]
            if reward.dim() >= 2:
                reward_flat = reward.squeeze(-1) if reward.shape[-1] == 1 else reward
            else:
                reward_flat = reward
            if done.dim() >= 2:
                done_flat = done.squeeze(-1) if done.shape[-1] == 1 else done
            else:
                done_flat = done
        else:
            # [B*n_agents, L, obs_dim]
            B = obs_hist.shape[0]
            L = obs_hist.shape[1]
            n_agents = obs_hist.shape[2]
            obs_dim = obs_hist.shape[3]
            obs_seq_flat = (
                obs_hist.permute(0, 2, 1, 3).contiguous().reshape(B * n_agents, L, obs_dim)
            )
            last_obs_flat = obs_seq_flat[:, -1, :]
            act_flat = encoded_action.flatten(0, 1)
            
            # Handle multi-step or single-step next_obs
            if is_multi_step:
                # [B*n_agents, L', obs_dim]
                next_obs_flat = next_obs.permute(0, 2, 1, 3).contiguous().reshape(B * n_agents, next_obs.shape[1], obs_dim)
            else:
                # [B*n_agents, obs_dim]
                next_obs_flat = next_obs.flatten(0, 1)
            
            # Reward and done: [B*n_agents, L'] or [B*n_agents]
            if is_multi_step:
                reward_flat = reward.unsqueeze(2).expand(-1, -1, n_agents, -1).flatten(0, 1)  # [B*n_agents, L']
                done_flat = done.unsqueeze(2).expand(-1, -1, n_agents, -1).flatten(0, 1)  # [B*n_agents, L']
            else:
                reward_flat = reward.flatten(0, 1)
                done_flat = done.flatten(0, 1)

        return obs_seq_flat, last_obs_flat, act_flat, next_obs_flat, reward_flat, done_flat, is_multi_step

    def _train_dynamics(self, group: str, flat_batch: TensorDictBase) -> None:

        if self.history_length <= 0:
            return super()._train_dynamics(group, flat_batch)

        keys = flat_batch.keys(True, True)
        if (group, "observation") not in keys:
            raise ValueError(f"Observation not found in flat_batch for group {group}")

        # Flattening + history injection.
        obs_seq_flat, last_obs_flat, act_flat, next_obs_flat, reward_flat, done_flat, is_multi_step = (
            self._flatten_inputs_for_world_model(group, flat_batch)
        )

        # Normalize observations/states if enabled
        if getattr(self, "state_normalize", False) and group in getattr(self, "_state_norm_mean", {}):
            obs_mean = self._state_norm_mean[group]
            obs_std = self._state_norm_std[group]
            eps = float(getattr(self, "reward_norm_eps", 1e-6))
            # Normalize each step in the history and the next-state target.
            if self.centralized_dynamics:
                obs_seq_flat = (obs_seq_flat - obs_mean) / (obs_std + eps)
                if is_multi_step:
                    # Normalize each step in the future sequence
                    next_obs_flat = (next_obs_flat - obs_mean.unsqueeze(0)) / (obs_std.unsqueeze(0) + eps)
                else:
                    next_obs_flat = (next_obs_flat - obs_mean) / (obs_std + eps)
            else:
                obs_seq_flat = (obs_seq_flat - obs_mean) / (obs_std + eps)
                if is_multi_step:
                    next_obs_flat = (next_obs_flat - obs_mean.unsqueeze(0)) / (obs_std.unsqueeze(0) + eps)
                else:
                    next_obs_flat = (next_obs_flat - obs_mean) / (obs_std + eps)

        use_rn = bool(getattr(self, "reward_normalize", False))
        if use_rn and group in getattr(self, "_reward_norm_mean", {}):
            eps = float(getattr(self, "reward_norm_eps", 1e-6))
            r_mean = self._reward_norm_mean[group]
            r_std = self._reward_norm_std[group] + eps

        logvar_coef = float(getattr(self, "logvar_loss_coef", 1.0))

        # Bootstrap sampling per model encourages ensemble diversity.
        batch_size = int(obs_seq_flat.shape[0])
        for i, model in enumerate(self._dynamics[group]):
            optimizer = self._dyn_optimizers[group][i]
            optimizer.zero_grad()
            # Sample with replacement
            idx = torch.randint(
                0,
                batch_size,
                (batch_size,),
                device="cpu",
                generator=getattr(self, "_world_model_rng", None),
            ).to(self.device)
            oseq = obs_seq_flat[idx]
            a = act_flat[idx]
            no = next_obs_flat[idx]
            r = reward_flat[idx]
            d = done_flat[idx]

            mu_next, lv_delta, mu_rew, lv_rew, done_logit = model(oseq, a)
            
            # Handle multi-step or single-step loss
            if is_multi_step:
                # Multi-step: compute loss for each future step and average
                # no: [B, L', ...], mu_next: [B, ...] (single-step prediction)
                # For now, we predict single-step and compare with first step of future
                # TODO: Could extend model to predict multi-step directly
                no_first = no[:, 0] if no.dim() > 2 else no
                inv_var_obs = torch.exp(-lv_delta)
                loss_obs = torch.mean((mu_next - no_first) ** 2 * inv_var_obs + logvar_coef * lv_delta)
                
                # Reward: use first step or average across steps
                r_first = r[:, 0] if r.dim() > 1 else r
                d_first = d[:, 0] if d.dim() > 1 else d
            else:
                # Single-step: original behavior
                inv_var_obs = torch.exp(-lv_delta)
                inv_var_rew = torch.exp(-lv_rew)
                loss_obs = torch.mean((mu_next - no) ** 2 * inv_var_obs + logvar_coef * lv_delta)
                r_first = r
                d_first = d

            if use_rn:
                if r_mean.dim() == 0:
                    r_n = (r_first - r_mean) / r_std
                    mu_rew_n = (mu_rew - r_mean) / r_std
                else:
                    r_n = (r_first - r_mean) / r_std
                    mu_rew_n = (mu_rew - r_mean) / r_std
                lv_rew_n = lv_rew - 2.0 * torch.log(r_std)
                inv_var_rew_n = torch.exp(-lv_rew_n)
                loss_rew = torch.mean((mu_rew_n - r_n) ** 2 * inv_var_rew_n + logvar_coef * lv_rew_n)
            else:
                inv_var_rew = torch.exp(-lv_rew)
                if is_multi_step:
                    # For multi-step, use simplified loss (model predicts single-step)
                    loss_rew = torch.mean((mu_rew - r_first) ** 2 * inv_var_rew + logvar_coef * lv_rew)
                else:
                    loss_rew = torch.mean((mu_rew - r_first) ** 2 * inv_var_rew + logvar_coef * lv_rew)

            if self._oracle_reward_enabled() and bool(
                getattr(self, "oracle_reward_disable_reward_head_loss", True)
            ):
                loss_rew = loss_rew * 0.0
            else:
                loss_rew = loss_rew * float(getattr(self, "reward_loss_coef", 1.0))

            loss_done = F.binary_cross_entropy_with_logits(done_logit, d_first.float())
            loss = loss_obs + loss_rew + loss_done * 0.1
            # print(f"loss_obs: {loss_obs.detach().cpu().item()}, loss_rew: {loss_rew.detach().cpu().item()}, loss_done: {loss_done.detach().cpu().item()}")
            loss.backward()
            optimizer.step()
            self._model_losses[group][i] = loss.detach()

        # Update elite indices (lowest loss are best)
        n_elites = self.n_elites or self.ensemble_size
        n_elites = max(1, min(int(n_elites), self.ensemble_size))
        self._elite_indices[group] = torch.argsort(self._model_losses[group])[:n_elites]

    @torch.no_grad()
    def _world_model_eval_losses(
        self, group: str, flat_batch: TensorDictBase
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.history_length <= 0:
            return super()._world_model_eval_losses(group, flat_batch)

        keys = flat_batch.keys(True, True)
        if (group, "observation") not in keys:
            return torch.full((self.ensemble_size,), float("nan"), device=self.device), {}

        obs_seq_flat, last_obs_flat, act_flat, next_obs_flat, reward_flat, done_flat, is_multi_step = (
            self._flatten_inputs_for_world_model(group, flat_batch)
        )

        # Normalize observations/states if enabled
        if getattr(self, "state_normalize", False) and group in getattr(self, "_state_norm_mean", {}):
            obs_mean = self._state_norm_mean[group]
            obs_std = self._state_norm_std[group]
            eps = float(getattr(self, "reward_norm_eps", 1e-6))
            obs_seq_flat = (obs_seq_flat - obs_mean) / (obs_std + eps)
            if is_multi_step:
                next_obs_flat = (next_obs_flat - obs_mean.unsqueeze(0)) / (obs_std.unsqueeze(0) + eps)
            else:
                next_obs_flat = (next_obs_flat - obs_mean) / (obs_std + eps)

        use_rn = bool(getattr(self, "reward_normalize", False))
        if use_rn and group in getattr(self, "_reward_norm_mean", {}):
            eps = float(getattr(self, "reward_norm_eps", 1e-6))
            r_mean = self._reward_norm_mean[group]
            r_std = self._reward_norm_std[group] + eps

        losses: List[torch.Tensor] = []
        total_loss_obs = 0.0
        total_loss_rew = 0.0
        total_loss_done = 0.0
        total_mse_obs = 0.0
        total_mse_rew = 0.0
        total_smape_obs = 0.0
        total_smape_rew = 0.0
        total_r2_obs = 0.0
        total_r2_rew = 0.0

        logvar_coef = float(getattr(self, "logvar_loss_coef", 1.0))
        for model in self._dynamics[group]:
            mu_next, lv_delta, mu_rew, lv_rew, done_logit = model(obs_seq_flat, act_flat)
            
            # Handle multi-step or single-step
            if is_multi_step:
                # Use first step of future for evaluation (model predicts single-step)
                no_first = next_obs_flat[:, 0] if next_obs_flat.dim() > 2 else next_obs_flat
                r_first = reward_flat[:, 0] if reward_flat.dim() > 1 else reward_flat
                d_first = done_flat[:, 0] if done_flat.dim() > 1 else done_flat
            else:
                no_first = next_obs_flat
                r_first = reward_flat
                d_first = done_flat
            
            inv_var_obs = torch.exp(-lv_delta)
            inv_var_rew = torch.exp(-lv_rew)
            loss_obs = torch.mean((mu_next - no_first) ** 2 * inv_var_obs + logvar_coef * lv_delta)

            rew_target = r_first.to(torch.float32)
            rew_pred = mu_rew.to(torch.float32)
            mse_rew = torch.mean((rew_pred - rew_target) ** 2)
            var_rew = torch.mean((rew_target - rew_target.mean()) ** 2)
            r2_rew = 1.0 - (mse_rew / (var_rew + 1e-12))

            obs_target = no_first.to(torch.float32)
            obs_pred = mu_next.to(torch.float32)
            mse_obs = torch.mean((obs_pred - obs_target) ** 2)
            var_obs = torch.mean((obs_target - obs_target.mean()) ** 2)
            r2_obs = 1.0 - (mse_obs / (var_obs + 1e-12))

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
                if r_mean.dim() == 0:
                    r_n = (r_first - r_mean) / r_std
                    mu_rew_n = (mu_rew - r_mean) / r_std
                else:
                    r_n = (r_first - r_mean) / r_std
                    mu_rew_n = (mu_rew - r_mean) / r_std
                lv_rew_n = lv_rew - 2.0 * torch.log(r_std)
                inv_var_rew_n = torch.exp(-lv_rew_n)
                loss_rew = torch.mean((mu_rew_n - r_n) ** 2 * inv_var_rew_n + logvar_coef * lv_rew_n)
            else:
                loss_rew = torch.mean((mu_rew - r_first) ** 2 * inv_var_rew + logvar_coef * lv_rew)
            loss_rew = loss_rew * float(getattr(self, "reward_loss_coef", 1.0))
            loss_done = F.binary_cross_entropy_with_logits(done_logit, d_first.float())
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

    @torch.no_grad()
    def _predict_next(
        self, group: str, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.history_length <= 0:
            return super()._predict_next(group, obs, action)

        # Build a sliding window if caller provides only the current observation.
        L = int(self.history_length) + 1
        if obs.dim() == 4:
            # [B, L, n_agents, obs_dim]
            obs_hist = obs
            obs_cur = obs_hist[:, -1]
        else:
            obs_cur = obs
            obs_hist = obs_cur.unsqueeze(1).repeat_interleave(L, dim=1)

        encoded_action = self._encode_action(group, action)
        if self.centralized_dynamics:
            obs_seq_flat = obs_hist.reshape(obs_hist.shape[0], obs_hist.shape[1], -1)
            act_flat = encoded_action.reshape(encoded_action.shape[0], -1)
        else:
            B = obs_hist.shape[0]
            L = obs_hist.shape[1]
            n_agents = obs_hist.shape[2]
            obs_dim = obs_hist.shape[3]
            obs_seq_flat = (
                obs_hist.permute(0, 2, 1, 3).contiguous().reshape(B * n_agents, L, obs_dim)
            )
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
                obs_seq_flat, act_flat
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
            next_obs = next_obs_flat.reshape(obs_cur.shape)
            reward = reward_flat.reshape(*obs_cur.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs_cur.shape[:-1], 1)
        else:
            next_obs = next_obs_flat.reshape(obs_cur.shape)
            reward = reward_flat.reshape(*obs_cur.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs_cur.shape[:-1], 1)

        return next_obs, reward, done_logit

    @torch.no_grad()
    def _predict_next_with_uncertainty(
        self, group: str, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.history_length <= 0:
            return super()._predict_next_with_uncertainty(group, obs, action)

        # Accept either current obs [B, n_agents, obs_dim] or history [B, L, n_agents, obs_dim]
        L = int(self.history_length) + 1
        if obs.dim() == 4:
            obs_hist = obs
            obs_cur = obs_hist[:, -1]
        else:
            obs_cur = obs
            obs_hist = obs_cur.unsqueeze(1).repeat_interleave(L, dim=1)

        encoded_action = self._encode_action(group, action)
        if self.centralized_dynamics:
            obs_seq_flat = obs_hist.reshape(obs_hist.shape[0], obs_hist.shape[1], -1)
            act_flat = encoded_action.reshape(encoded_action.shape[0], -1)
            B = int(obs_cur.shape[0])
            n_agents = int(obs_cur.shape[-2])
        else:
            B = int(obs_cur.shape[0])
            n_agents = int(obs_cur.shape[-2])
            obs_seq_flat = (
                obs_hist.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(B * n_agents, obs_hist.shape[1], obs_hist.shape[3])
            )
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
                obs_seq_flat, act_flat
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

        total_obs_unc = torch.sqrt(
            epistemic_obs_var + aleatoric_obs_std.pow(2).clamp_min(0.0)
        )
        total_rew_unc = torch.sqrt(
            epistemic_rew_std.pow(2) + aleatoric_rew_std.pow(2).clamp_min(0.0)
        )

        if self.centralized_dynamics:
            unc = {
                "epistemic_obs_var": epistemic_obs_var,
                "epistemic_rew_std": epistemic_rew_std,
                "aleatoric_obs_logvar": aleatoric_obs_logvar,
                "aleatoric_rew_logvar": aleatoric_rew_logvar,
                "total_obs_unc": total_obs_unc,
                "total_rew_unc": total_rew_unc,
            }
        else:
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

        if self.centralized_dynamics:
            next_obs = next_obs_flat.reshape(obs_cur.shape)
            reward = reward_flat.reshape(*obs_cur.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs_cur.shape[:-1], 1)
        else:
            next_obs = next_obs_flat.reshape(obs_cur.shape)
            reward = reward_flat.reshape(*obs_cur.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs_cur.shape[:-1], 1)

        return next_obs, reward, done_logit, unc

    @torch.no_grad()
    def _predict_next_for_vis(
        self, group: str, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next transition using MEAN predictions (no stochastic sampling) for visualization.
        
        This is used for visualization to show model accuracy without sampling noise.
        For actual rollouts, use `_predict_next` which includes stochastic sampling.
        
        Overrides base class to handle observation history for recurrent models.
        """
        if self.history_length <= 0:
            return super()._predict_next_for_vis(group, obs, action)

        # Build a sliding window if caller provides only the current observation.
        L = int(self.history_length) + 1
        if obs.dim() == 4:
            # [B, L, n_agents, obs_dim]
            obs_hist = obs
            obs_cur = obs_hist[:, -1]
        else:
            obs_cur = obs
            obs_hist = obs_cur.unsqueeze(1).repeat_interleave(L, dim=1)

        encoded_action = self._encode_action(group, action)
        if self.centralized_dynamics:
            obs_seq_flat = obs_hist.reshape(obs_hist.shape[0], obs_hist.shape[1], -1)
            act_flat = encoded_action.reshape(encoded_action.shape[0], -1)
        else:
            B = obs_hist.shape[0]
            L = obs_hist.shape[1]
            n_agents = obs_hist.shape[2]
            obs_dim = obs_hist.shape[3]
            obs_seq_flat = (
                obs_hist.permute(0, 2, 1, 3).contiguous().reshape(B * n_agents, L, obs_dim)
            )
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
                obs_seq_flat, act_flat
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
            next_obs = next_obs_flat.reshape(obs_cur.shape)
            reward = reward_flat.reshape(*obs_cur.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs_cur.shape[:-1], 1)
        else:
            next_obs = next_obs_flat.reshape(obs_cur.shape)
            reward = reward_flat.reshape(*obs_cur.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs_cur.shape[:-1], 1)

        return next_obs, reward, done_logit

    def _wm_vis_render_frames(
        self, *, group: str, traj: TensorDictBase, horizon: int
    ) -> Optional[List]:
        """Override to handle observation history for recurrent models.
        
        For open_loop mode: extracts history from trajectory.
        For closed_loop mode: maintains a sliding history window.
        """
        if self.history_length <= 0:
            return super()._wm_vis_render_frames(group=group, traj=traj, horizon=horizon)

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
        H = int(self.history_length)
        L_required = H + 1

        # Closed-loop state: maintain history window
        obs_pred_hist: Optional[torch.Tensor] = None
        if self._wm_vis_mode() == "closed_loop":
            obs0 = traj[0].get((group, "observation")).to(self.device)
            obs0_b = obs0.unsqueeze(0) if obs0.dim() == 2 else obs0
            # Initialize history by repeating the first observation
            obs_pred_hist = obs0_b.unsqueeze(1).repeat_interleave(L_required, dim=1)  # [B, L, n_agents, obs_dim]

        reward_true_hist: List[float] = []
        reward_pred_hist: List[float] = []
        frames: List = []

        # Stable limits from ground truth
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

            # Build observation history for input
            if self._wm_vis_mode() == "closed_loop":
                if obs_pred_hist is None:
                    obs_pred_hist = obs_true_b.unsqueeze(1).repeat_interleave(L_required, dim=1)
                obs_in_hist = obs_pred_hist
            else:
                # open_loop: extract history from trajectory
                if t >= H:
                    # We have enough history in the trajectory
                    obs_hist_list = []
                    for i in range(L_required):
                        obs_t_i = traj[t - H + i].get((group, "observation")).to(self.device)
                        obs_t_i_b = obs_t_i.unsqueeze(0) if obs_t_i.dim() == 2 else obs_t_i
                        obs_hist_list.append(obs_t_i_b)
                    obs_in_hist = torch.stack(obs_hist_list, dim=1)  # [B, L, n_agents, obs_dim]
                else:
                    # Not enough history: pad with first observation
                    obs_hist_list = [obs_true_b]
                    for i in range(H):
                        if t - i - 1 >= 0:
                            obs_t_i = traj[t - i - 1].get((group, "observation")).to(self.device)
                            obs_t_i_b = obs_t_i.unsqueeze(0) if obs_t_i.dim() == 2 else obs_t_i
                            obs_hist_list.insert(0, obs_t_i_b)
                        else:
                            obs_hist_list.insert(0, obs_true_b)
                    obs_in_hist = torch.stack(obs_hist_list, dim=1)  # [B, L, n_agents, obs_dim]

            oracle_vis = bool(getattr(self, "use_oracle", False))
            oracle_used = False
            oracle_failed = False
            with torch.no_grad():
                if oracle_vis:
                    oracle_used = True
                    next_obs_pred_b = next_obs_true_b
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
                    # Use _predict_next_for_vis with history
                    next_obs_pred_b, rew_pred_b, done_logit_b = self._predict_next_for_vis(
                        group, obs_in_hist, act_true_b
                    )

            # Extract ground truth reward
            r_true = 0.0
            if rew_true is not None:
                try:
                    r_true = float(rew_true.to(torch.float32).mean().detach().cpu().item())
                except Exception:
                    pass
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
            xy_pred = self._wm_vis_extract_xy(obs_in_hist[0, -1])  # Use last obs in history
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
                f"MBPO recurrent world-model debug ({self._wm_vis_mode()}) | group={group} | t={t} | history={H}"
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

            # Advance closed-loop state: slide history window
            if self._wm_vis_mode() == "closed_loop":
                # Slide window: remove first, add predicted next
                obs_pred_hist = torch.cat([obs_pred_hist[:, 1:], next_obs_pred_b.unsqueeze(1)], dim=1)

        return frames

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        # Keep the original MBPO behavior, but store history-augmented samples for training.
        if self.history_length <= 0:
            return super().process_batch(group, batch)

        # We largely mirror `_MbpoWorldModelMixin.process_batch`, but replace the
        # dynamics-buffer write path so the world model sees history windows.
        skip_world_model = False
        unc_threshold = self._get_rollout_uncertainty_threshold(group)
        if unc_threshold is not None and unc_threshold == 0.0:
            if not getattr(self, "use_oracle", False):
                skip_world_model = True

        if not skip_world_model:
            synth_buf = getattr(self, "_synthetic_replay_buffers", {}).get(group, None)
            if synth_buf is not None and hasattr(synth_buf, "empty"):
                synth_buf.empty()

        batch = super(_MbpoWorldModelMixin, self).process_batch(group, batch)

        if skip_world_model:
            return batch

        dyn_buf = getattr(self, "_dynamics_train_buffers", {}).get(group, None)
        dyn_src = batch  # keep time dimension if present (buffer will window + flatten)
        if dyn_buf is not None and dyn_src is not None and dyn_src.numel() > 0:
            try:
                dyn_buf.extend(dyn_src)
            except Exception:
                pass

        self._train_steps[group] += 1

        if self._train_steps[group] % self.model_train_freq == 0 and not skip_world_model:
            dyn_buf = getattr(self, "_dynamics_train_buffers", {}).get(group, None)
            for _ in range(int(getattr(self, "training_iterations", 0))):
                if dyn_buf is not None and len(dyn_buf) > 0:
                    try:
                        dyn_batch = dyn_buf.sample(
                            int(getattr(self, "model_batch_size", 256)),
                            generator=getattr(self, "_world_model_rng", None),
                        )
                    except Exception:
                        try:
                            dyn_batch = dyn_buf._preprocess(  # type: ignore[attr-defined]
                                dyn_src, keys=dyn_src.keys(True, True), groups={group}
                            ) or dyn_src
                        except Exception:
                            dyn_batch = dyn_src
                else:
                    try:
                        dyn_batch = dyn_buf._preprocess(  # type: ignore[attr-defined]
                            dyn_src, keys=dyn_src.keys(True, True), groups={group}
                        ) or dyn_src
                    except Exception:
                        dyn_batch = dyn_src
                self._train_dynamics(group, dyn_batch)

            try:
                self._maybe_log_world_model_debug_video(group=group, batch=batch)
            except Exception:
                pass

            if self.save_world_model_path is not None:
                min_train_step = min(self._train_steps.values())
                should_save = False
                if self.save_world_model_interval is None:
                    should_save = True
                else:
                    min_last_save = min(self._last_save_step.values())
                    steps_since_last_save = min_train_step - min_last_save
                    if steps_since_last_save >= self.save_world_model_interval:
                        should_save = True
                if should_save:
                    self.save_world_model(self.save_world_model_path)
                    for g in self._last_save_step.keys():
                        self._last_save_step[g] = self._train_steps[g]

        if self._train_steps[group] > self.warmup_steps and not skip_world_model:
            synthetic = self._generate_model_rollouts(group, batch.reshape(-1) if not self.has_rnn else batch)
            if synthetic is not None and synthetic.numel() > 0:
                synthetic = super(_MbpoWorldModelMixin, self).process_batch(group, synthetic)
                try:
                    if not self.has_rnn:
                        synthetic = synthetic.reshape(-1)
                    if "synthetic_valid" in synthetic.keys(True, True):
                        v = synthetic.get("synthetic_valid").to(torch.bool).reshape(-1)
                        if v.numel() == synthetic.numel():
                            synthetic = synthetic[v]
                except Exception:
                    pass
                synth_buf = getattr(self, "_synthetic_replay_buffers", {}).get(group, None)
                if synth_buf is not None:
                    try:
                        synth_buf.extend(synthetic.to(synth_buf.storage.device))
                    except Exception:
                        pass
        return batch

    def _generate_model_rollouts(
        self, group: str, flat_batch: TensorDictBase
    ) -> Optional[TensorDictBase]:
        if self.history_length <= 0:
            return super()._generate_model_rollouts(group, flat_batch)

        # Copy of MBPO rollouts but maintain a sliding observation history window.
        sr = getattr(self, "syn_ratio", None)
        if sr is not None:
            target_synth = int(math.ceil(float(flat_batch.batch_size[0]) * max(0.0, float(sr))))
        else:
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
        batch_dims = obs.shape[:-2]
        n_agents = obs.shape[-2]
        done_mask = torch.zeros((*batch_dims, n_agents, 1), device=self.device, dtype=torch.bool)

        # Initialize history by repeating the starting observation.
        L = int(self.history_length) + 1
        obs_hist = obs.unsqueeze(1).repeat_interleave(L, dim=1)  # [B, L, n_agents, obs_dim]

        rollouts: List[TensorDictBase] = []
        traj_ids = start.get(("collector", "traj_ids")).to(self.device) if ("collector", "traj_ids") in start.keys(True, True) else torch.zeros(batch_dims, device=self.device, dtype=torch.long)
        if (group, "episode_reward") in start.keys(True, True):
            ep_rew = start.get((group, "episode_reward")).to(self.device)
        else:
            ep_rew = torch.zeros((*batch_dims, n_agents, 1), device=self.device, dtype=torch.float32)

        unc_threshold = self._get_rollout_uncertainty_threshold(group)
        unc_metric = str(getattr(self, "rollout_uncertainty_metric", "total_rew_unc") or "total_rew_unc")

        for _ in range(horizon):
            td_in = TensorDict({}, batch_size=batch_dims, device=self.device)
            td_in.set(
                group,
                TensorDict({"observation": obs}, batch_size=(*batch_dims, n_agents), device=self.device),
            )
            if self.action_mask_spec is not None and (group, "action_mask") in start.keys(True, True):
                td_in.get(group).set("action_mask", start.get((group, "action_mask")).to(self.device))

            policy_td = self._policies_for_collection[group](td_in)
            action = policy_td.get((group, "action")).detach()
            policy_group = policy_td.get(group)

            env_done = done_mask.any(dim=-2)
            env_active = (~env_done).squeeze(-1)
            synthetic_valid_env = env_active.clone()
            invalid_agent = None

            if getattr(self, "use_oracle", False):
                # Oracle mode ignores recurrent history (uses simulator).
                next_obs, reward_pred, done_logit = self._step_oracle_env(group, obs, action)
            else:
                # Pass history to the recurrent world model via `_predict_next` by providing [B, L, ...].
                if unc_threshold is None:
                    next_obs, reward_pred, done_logit = self._predict_next(group, obs_hist, action)
                else:
                    next_obs, reward_pred, done_logit, unc = self._predict_next_with_uncertainty(  # type: ignore[misc]
                        group, obs_hist, action
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
                invalid_agent = invalid_env.view(*batch_dims, 1, 1).expand(*batch_dims, n_agents, 1)
                next_obs = torch.where(invalid_agent, obs, next_obs)

            if self._oracle_reward_enabled():
                try:
                    reward_pred = self._oracle_reward_from_predicted_next_obs(group, next_obs)
                except Exception:
                    pass

            if invalid_agent is not None:
                reward_pred = reward_pred.masked_fill(invalid_agent, 0.0)

            done_prob = torch.sigmoid(done_logit)
            done_flag = done_prob > 0.5
            if invalid_agent is not None:
                done_flag = done_flag | invalid_agent
            done_flag = done_flag | done_mask

            ep_rew_next = ep_rew + reward_pred.detach()
            env_done_next = done_flag.any(dim=-2)

            rollout_td = TensorDict({}, batch_size=batch_dims, device=self.device)
            rollout_td.set(
                group,
                TensorDict(
                    {"observation": obs.detach(), "action": action, "episode_reward": ep_rew.detach()},
                    batch_size=(*batch_dims, n_agents),
                    device=self.device,
                ),
            )
            for k in ("logits", "log_prob", "loc", "scale"):
                if k in policy_group.keys():
                    rollout_td.get(group).set(k, policy_group.get(k).detach())
            rollout_td.set("collector", TensorDict({"traj_ids": traj_ids}, batch_size=batch_dims, device=self.device))
            rollout_td.set("done", env_done_next.detach())
            rollout_td.set("terminated", env_done_next.detach())
            rollout_td.set("synthetic_valid", synthetic_valid_env.view(*batch_dims, 1).to(torch.bool).detach())
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

            # Slide history window.
            obs = next_obs
            obs_hist = torch.cat([obs_hist[:, 1:].detach(), obs.unsqueeze(1).detach()], dim=1)
            done_mask = done_flag
            ep_rew = ep_rew_next
            if torch.all(done_mask):
                break

        if not rollouts:
            return None
        out = torch.stack(rollouts, dim=0)
        try:
            total = int(out.numel())
            if total > target_synth and out.batch_size is not None and len(out.batch_size) >= 2:
                T, B = int(out.batch_size[0]), int(out.batch_size[1])
                keep_B = max(1, int(math.floor(target_synth / float(max(1, T)))))
                keep_B = min(keep_B, B)
                out = out[:, :keep_B]
        except Exception:
            pass
        return out


class MbpoRecurrentMasac(_MbpoRecurrentWorldModelMixin, Masac):
    """Recurrent MBPO (MASAC backbone)."""

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
        logvar_loss_coef: float = 1.0,
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
        history_length: int = 0,
        future_length: int = 1,
        gru_num_layers: int = 1,
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
            history_length=history_length,
            future_length=future_length,
            gru_num_layers=gru_num_layers,
        )


class MbpoRecurrentMappo(_MbpoRecurrentWorldModelMixin, Mappo):
    """Recurrent MBPO (MAPPO backbone)."""

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
        logvar_loss_coef: float = 1.0,
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
        history_length: int = 0,
        future_length: int = 1,
        gru_num_layers: int = 1,
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
            history_length=history_length,
            future_length=future_length,
            gru_num_layers=gru_num_layers,
        )

# Backwards-compatible name: default MBPO remains MASAC-based.
MbpoRecurrent = MbpoRecurrentMasac
@dataclass
class MbpoRecurrentConfig(MasacConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.MbpoRecurrentMasac`."""

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

    # Recurrent world-model knobs
    history_length: int = 0  # History window length for training (0 = no history)
    future_length: int = 1  # Future window length for multi-step prediction (minimum 1)
    gru_num_layers: int = 1

    @staticmethod
    def associated_class() -> Type[Masac]:
        return MbpoRecurrentMasac

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
class MbpoRecurrentMappoConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.MbpoRecurrentMappo`."""

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

    history_length: int = 0  # History window length for training (0 = no history)
    future_length: int = 1  # Future window length for multi-step prediction (minimum 1)
    gru_num_layers: int = 1

    @staticmethod
    def associated_class() -> Type[Mappo]:
        return MbpoRecurrentMappo

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

