import math
from dataclasses import MISSING, dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import Composite, Unbounded
from torchrl.objectives import LossModule

from benchmarl.algorithms.mappo import Mappo, MappoConfig
from benchmarl.algorithms.masac import Masac, MasacConfig
from benchmarl.models.common import ModelConfig


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
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(last_dim, hidden_size), nn.ReLU()]
            last_dim = hidden_size
        self.net = nn.Sequential(*layers)

        self._stochastic = stochastic
        # Bounds used to keep log-variances well-behaved (softplus clamp like PETS/MAMBPO)
        self.register_buffer("_min_log_var", torch.tensor(float(min_log_var)))
        self.register_buffer("_max_log_var", torch.tensor(float(max_log_var)))

        # Predict mean(delta_next_obs), mean(reward), and optionally log-variances; plus done logits.
        self.mu_head = nn.Linear(last_dim, next_obs_dim + reward_dim)
        self.log_var_head = nn.Linear(last_dim, next_obs_dim + reward_dim)
        self.done_head = nn.Linear(last_dim, done_dim)

        self._next_obs_dim = next_obs_dim
        self._reward_dim = reward_dim
        self._done_dim = done_dim

    def forward(
        self, obs_flat: torch.Tensor, action_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_flat, action_flat], dim=-1)
        feat = self.net(x)
        mu = self.mu_head(feat)
        log_var = self.log_var_head(feat)
        done_logit = self.done_head(feat)

        # Softplus-based clamp to [min_log_var, max_log_var]
        max_lv = self._max_log_var
        min_lv = self._min_log_var
        log_var = max_lv - F.softplus(max_lv - log_var)
        log_var = min_lv + F.softplus(log_var - min_lv)

        if not self._stochastic:
            log_var = log_var * 0.0

        next_obs_end = self._next_obs_dim
        reward_end = next_obs_end + self._reward_dim

        mu_delta = mu[..., :next_obs_end]
        mu_rew = mu[..., next_obs_end:reward_end]
        lv_delta = log_var[..., :next_obs_end]
        lv_rew = log_var[..., next_obs_end:reward_end]

        mu_next_obs = obs_flat + mu_delta
        return mu_next_obs, lv_delta, mu_rew, lv_rew, done_logit


class _MbpoWorldModelMixin:
    """MBPO world-model implementation shared across different base algorithms."""

    def _mbpo_init(
        self,
        *,
        rollout_horizon: int,
        model_train_freq: int,
        ensemble_size: int,
        model_batch_size: int,
        real_ratio: float,
        temperature: float,
        model_lr: float,
        model_hidden_size: int,
        model_num_layers: int,
        centralized_dynamics: bool = False,
        stochastic_dynamics: bool = True,
        n_elites: Optional[int] = None,
        min_log_var: float = -10.0,
        max_log_var: float = -2.0,
    ) -> None:
        self.rollout_horizon = rollout_horizon
        self.model_train_freq = model_train_freq
        self.ensemble_size = ensemble_size
        self.model_batch_size = model_batch_size
        self.real_ratio = real_ratio
        self.temperature = max(temperature, 1e-3)
        self.model_lr = model_lr
        self.model_hidden_size = model_hidden_size
        self.model_num_layers = model_num_layers
        self.centralized_dynamics = centralized_dynamics
        self.stochastic_dynamics = stochastic_dynamics
        self.n_elites = n_elites
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self._dynamics: Dict[str, List[_DynamicsModel]] = {}
        self._dyn_optimizers: Dict[str, List[torch.optim.Optimizer]] = {}
        self._model_losses: Dict[str, torch.Tensor] = {}
        self._train_steps: Dict[str, int] = {}
        self._elite_indices: Dict[str, torch.Tensor] = {}

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

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = super().process_batch(group, batch)
        flat_batch = batch.reshape(-1) if not self.has_rnn else batch

        self._train_steps[group] += 1
        if self._train_steps[group] % self.model_train_freq == 0:
            self._train_dynamics(group, flat_batch)
            synthetic = self._generate_model_rollouts(group, flat_batch)
            if synthetic is not None and synthetic.numel() > 0:
                synthetic = super().process_batch(group, synthetic)
                if not self.has_rnn:
                    synthetic = synthetic.reshape(-1)
                buffer = self.experiment.replay_buffers[group]
                buffer.extend(synthetic.to(buffer.storage.device))

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

        # Bootstrap sampling per model encourages ensemble diversity.
        n_samples = obs_flat.shape[0]

        for i, model in enumerate(self._dynamics[group]):
            optimizer = self._dyn_optimizers[group][i]
            optimizer.zero_grad()

            # Bootstrap indices (with replacement)
            idx = torch.randint(0, n_samples, (n_samples,), device=obs_flat.device)
            o = obs_flat[idx]
            a = action_flat[idx]
            no = next_obs_flat[idx]
            r = reward_flat[idx]
            d = done_flat[idx]

            mu_next, lv_delta, mu_rew, lv_rew, done_logit = model(o, a)

            # Gaussian NLL-style losses (per-dim): (x-mu)^2 * exp(-log_var) + log_var
            inv_var_obs = torch.exp(-lv_delta)
            inv_var_rew = torch.exp(-lv_rew)
            loss_obs = torch.mean((mu_next - no) ** 2 * inv_var_obs + lv_delta)
            loss_rew = torch.mean((mu_rew - r) ** 2 * inv_var_rew + lv_rew)
            loss_done = F.binary_cross_entropy_with_logits(done_logit, d.float())
            loss = loss_obs + loss_rew + loss_done
            loss.backward()
            optimizer.step()
            self._model_losses[group][i] = loss.detach()

        # Update elite indices (lowest loss are best)
        n_elites = self.n_elites or self.ensemble_size
        n_elites = max(1, min(int(n_elites), self.ensemble_size))
        self._elite_indices[group] = torch.argsort(self._model_losses[group])[:n_elites]

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

        # Compute outputs for elite models, then sample one model per transition.
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

        # Select which elite model to use per sample (epistemic uncertainty)
        n_elites = mu_next_s.shape[0]
        bsz = mu_next_s.shape[1]
        model_idx = torch.randint(0, n_elites, (bsz,), device=mu_next_s.device)
        batch_idx = torch.arange(bsz, device=mu_next_s.device)

        mu_next = mu_next_s[model_idx, batch_idx]
        lv_delta = lv_delta_s[model_idx, batch_idx]
        mu_rew = mu_rew_s[model_idx, batch_idx]
        lv_rew = lv_rew_s[model_idx, batch_idx]
        done_logit = done_logit_s[model_idx, batch_idx]

        # Sample (aleatoric uncertainty)
        if self.stochastic_dynamics:
            std_next = torch.exp(0.5 * lv_delta)
            std_rew = torch.exp(0.5 * lv_rew)
            next_obs_flat = torch.normal(mu_next, std_next)
            reward_flat = torch.normal(mu_rew, std_rew)
        else:
            next_obs_flat = mu_next
            reward_flat = mu_rew

        if self.centralized_dynamics:
            next_obs = next_obs_flat.reshape(obs.shape)
            reward = reward_flat.reshape(*obs.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs.shape[:-1], 1)
        else:
            next_obs = next_obs_flat.reshape(obs.shape)
            reward = reward_flat.reshape(*obs.shape[:-1], 1)
            done_logit = done_logit.reshape(*obs.shape[:-1], 1)

        return next_obs, reward, done_logit

    def _sample_start_states(
        self, group: str, flat_batch: TensorDictBase, sample_size: int
    ) -> Optional[TensorDictBase]:
        buffer = self.experiment.replay_buffers[group]
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
        model_batch = max(1, int(self.model_batch_size * (1.0 - self.real_ratio)))
        start = self._sample_start_states(group, flat_batch, model_batch)
        if start is None or start.numel() == 0:
            return None

        obs = start.get((group, "observation")).to(self.device)
        # Parent batch dims do NOT include the agent dimension. The group sub-tensordict
        # must include it (matching what the collector produces).
        batch_dims = obs.shape[:-2]  # e.g. [B]
        n_agents = obs.shape[-2]
        done_mask = torch.zeros((*batch_dims, n_agents, 1), device=self.device, dtype=torch.bool)

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

        for _ in range(self.rollout_horizon):
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

            next_obs, reward_pred, done_logit = self._predict_next(group, obs, action)
            done_prob = torch.sigmoid(done_logit)
            done_flag = done_prob > 0.5
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
        return torch.cat(rollouts, dim=0)


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
        centralized_dynamics: bool = False,
        stochastic_dynamics: bool = True,
        n_elites: Optional[int] = None,
        min_log_var: float = -10.0,
        max_log_var: float = -2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mbpo_init(
            rollout_horizon=rollout_horizon,
            model_train_freq=model_train_freq,
            ensemble_size=ensemble_size,
            model_batch_size=model_batch_size,
            real_ratio=real_ratio,
            temperature=temperature,
            model_lr=model_lr,
            model_hidden_size=model_hidden_size,
            model_num_layers=model_num_layers,
            centralized_dynamics=centralized_dynamics,
            stochastic_dynamics=stochastic_dynamics,
            n_elites=n_elites,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
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
        centralized_dynamics: bool = False,
        stochastic_dynamics: bool = True,
        n_elites: Optional[int] = None,
        min_log_var: float = -10.0,
        max_log_var: float = -2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mbpo_init(
            rollout_horizon=rollout_horizon,
            model_train_freq=model_train_freq,
            ensemble_size=ensemble_size,
            model_batch_size=model_batch_size,
            real_ratio=real_ratio,
            temperature=temperature,
            model_lr=model_lr,
            model_hidden_size=model_hidden_size,
            model_num_layers=model_num_layers,
            centralized_dynamics=centralized_dynamics,
            stochastic_dynamics=stochastic_dynamics,
            n_elites=n_elites,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
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
    centralized_dynamics: bool = False
    stochastic_dynamics: bool = True
    n_elites: Optional[int] = None
    min_log_var: float = -10.0
    max_log_var: float = -2.0

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
    centralized_dynamics: bool = False
    stochastic_dynamics: bool = True
    n_elites: Optional[int] = None
    min_log_var: float = -10.0
    max_log_var: float = -2.0

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

