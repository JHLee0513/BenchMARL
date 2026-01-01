import math
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
        on_policy: bool,
    ):
        self._real = real
        self._synthetic = synthetic
        self._get_real_ratio = get_real_ratio
        self._on_policy = on_policy

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
        return self._real.extend(data)

    def update_tensordict_priority(self, data: TensorDictBase):
        # Priorities apply to the real buffer only.
        if hasattr(self._real, "update_tensordict_priority"):
            return self._real.update_tensordict_priority(data)

    def _safe_sample(self, buffer: ReplayBuffer, n: int) -> Optional[TensorDictBase]:
        if n <= 0:
            return None
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

        rr = float(self._get_real_ratio() or 1.0)
        rr = min(max(rr, 0.0), 1.0)
        n_real = max(1, min(total, int(math.ceil(total * rr))))
        n_synth = max(0, total - n_real)

        # If no synthetic is needed, behave exactly like the real buffer.
        if n_synth <= 0:
            return self._safe_sample(self._real, total)

        # If synthetic buffer is empty, fall back to real-only in a *single* call
        # (avoids changing SamplerWithoutReplacement behavior).
        try:
            synth_len = len(self._synthetic)
        except Exception:
            synth_len = 0
        if synth_len <= 0:
            return self._safe_sample(self._real, total)

        n_synth = min(n_synth, synth_len)
        # One call to real buffer, one call to synthetic buffer.
        real_td = self._safe_sample(self._real, total - n_synth)
        synth_td = self._safe_sample(self._synthetic, n_synth)
        if synth_td is None:
            return real_td
        return torch.cat([real_td, synth_td], dim=0)


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

        assert not stochastic, "Stochastic dynamics are currently disabled on purpose"

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

    def get_replay_buffer(
        self, group: str, transforms: List = None
    ) -> ReplayBuffer:
        # Create the real buffer using the default BenchMARL logic.
        real_buffer = super().get_replay_buffer(group=group, transforms=transforms)

        # Create a dedicated synthetic buffer with enough capacity for model rollouts.
        # We size it relative to the on-policy real buffer size for this run.
        base_memory_size = self.experiment_config.replay_buffer_memory_size(self.on_policy)
        rr = float(getattr(self, "real_ratio", 1.0))
        rr = min(max(rr, 0.0), 1.0)
        horizon = max(0, int(getattr(self, "rollout_horizon", 0)))
        synthetic_multiplier = (1.0 - rr) * float(horizon)
        synthetic_memory_size = int(math.ceil(base_memory_size * max(0.0, synthetic_multiplier)))
        synthetic_memory_size = max(1, synthetic_memory_size)

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
            on_policy=getattr(self, "on_policy", False),
        )

    def replay_buffer_memory_size_multiplier(self) -> float:
        # Do NOT inflate the main replay buffer: synthetic rollouts are stored separately.
        return 1.0

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
        warmup_steps: int = 0,
        load_world_model_path: Optional[str] = None,
        load_world_model_strict: bool = True,
        save_world_model_path: Optional[str] = None,
        save_world_model_interval: Optional[int] = None,
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
        self.warmup_steps = warmup_steps
        self.save_world_model_path = save_world_model_path
        self.save_world_model_interval = save_world_model_interval
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
        
        # Load pretrained world model if path is provided
        if load_world_model_path is not None:
            self.load_world_model(load_world_model_path, strict=load_world_model_strict)
        
        # Track last save step for interval-based saving
        self._last_save_step: Dict[str, int] = {group: 0 for group in self.group_map.keys()}
        # Use a dedicated RNG for world-model training so we don't perturb the policy/replay sampling RNG.
        # This is critical for on-policy algorithms: consuming global RNG here can change minibatch order.
        seed = int(getattr(self.experiment, "seed", 0))
        self._world_model_rng = torch.Generator(device="cpu")
        self._world_model_rng.manual_seed(seed + 10_000_019)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        # Ensure the synthetic buffer exists (created via get_replay_buffer()).
        # We don't create buffers here to avoid touching Experiment setup logic.
        synth_buf = getattr(self, "_synthetic_replay_buffers", {}).get(group, None)
        if synth_buf is not None and hasattr(synth_buf, "empty"):
            # Synthetic rollouts are per-iteration; clear at the start of the step.
            synth_buf.empty()

        batch = super().process_batch(group, batch)
        flat_batch = batch.reshape(-1) if not self.has_rnn else batch

        self._train_steps[group] += 1
        if self._train_steps[group] % self.model_train_freq == 0:
            self._train_dynamics(group, flat_batch)
            if self._train_steps[group] > self.warmup_steps:
                pass
                # synthetic = self._generate_model_rollouts(group, flat_batch)
                # if synthetic is not None and synthetic.numel() > 0:
                #     synthetic = super().process_batch(group, synthetic)
                #     if not self.has_rnn:
                #         synthetic = synthetic.reshape(-1)
                #         # Store synthetic rollouts in the dedicated buffer.
                #         synth_buf = getattr(self, "_synthetic_replay_buffers", {}).get(
                #             group, None
                #         )
                #         if synth_buf is not None:
                #             synth_buf.extend(synthetic.to(synth_buf.storage.device))
                #     else:
                #         # RNN batches have sequence structure; mixing is not supported yet.
                #         pass
            
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
            
            # Compute range metrics: mean ± std (averaged across samples)
            pred_reward_min = (mean_pred_flat - std_pred_flat).mean().detach().item()
            pred_reward_max = (mean_pred_flat + std_pred_flat).mean().detach().item()

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
                f"world_model/{group}/bias/reward_range_min": pred_reward_min,
                f"world_model/{group}/bias/reward_range_max": pred_reward_max,
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
        model_batch = max(1, int(float(flat_batch.batch_size[0]) * (1.0 - self.real_ratio)))
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
                "model_train_freq": self.model_train_freq,
                "ensemble_size": self.ensemble_size,
                "model_batch_size": self.model_batch_size,
                "real_ratio": self.real_ratio,
                "temperature": self.temperature,
                "model_lr": self.model_lr,
                "model_hidden_size": self.model_hidden_size,
                "model_num_layers": self.model_num_layers,
                "centralized_dynamics": self.centralized_dynamics,
                "stochastic_dynamics": self.stochastic_dynamics,
                "n_elites": self.n_elites,
                "min_log_var": self.min_log_var,
                "max_log_var": self.max_log_var,
                "warmup_steps": self.warmup_steps,
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
                "model_train_freq": self.model_train_freq,
                "ensemble_size": self.ensemble_size,
                "model_batch_size": self.model_batch_size,
                "real_ratio": self.real_ratio,
                "temperature": self.temperature,
                "model_lr": self.model_lr,
                "model_hidden_size": self.model_hidden_size,
                "model_num_layers": self.model_num_layers,
                "centralized_dynamics": self.centralized_dynamics,
                "stochastic_dynamics": self.stochastic_dynamics,
                "n_elites": self.n_elites,
                "min_log_var": self.min_log_var,
                "max_log_var": self.max_log_var,
                "warmup_steps": self.warmup_steps,
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
        centralized_dynamics: bool = False,
        stochastic_dynamics: bool = True,
        n_elites: Optional[int] = None,
        min_log_var: float = -10.0,
        max_log_var: float = -2.0,
        warmup_steps: int = 0,
        load_world_model_path: Optional[str] = None,
        load_world_model_strict: bool = True,
        save_world_model_path: Optional[str] = None,
        save_world_model_interval: Optional[int] = None,
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
            warmup_steps=warmup_steps,
            load_world_model_path=load_world_model_path,
            load_world_model_strict=load_world_model_strict,
            save_world_model_path=save_world_model_path,
            save_world_model_interval=save_world_model_interval,
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
        warmup_steps: int = 0,
        load_world_model_path: Optional[str] = None,
        load_world_model_strict: bool = True,
        save_world_model_path: Optional[str] = None,
        save_world_model_interval: Optional[int] = None,
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
            warmup_steps=warmup_steps,
            load_world_model_path=load_world_model_path,
            load_world_model_strict=load_world_model_strict,
            save_world_model_path=save_world_model_path,
            save_world_model_interval=save_world_model_interval,
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
    warmup_steps: int = 0
    load_world_model_path: Optional[str] = None
    load_world_model_strict: bool = True
    save_world_model_path: Optional[str] = None
    save_world_model_interval: Optional[int] = None

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
    warmup_steps: int = 0
    load_world_model_path: Optional[str] = None
    load_world_model_strict: bool = True
    save_world_model_path: Optional[str] = None
    save_world_model_interval: Optional[int] = None

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

