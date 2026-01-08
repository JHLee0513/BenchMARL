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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector
from torchrl.data import Categorical, OneHot
from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type

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
    save_every_n_batches: int = 1
    # Avoid creating W&B / CSV logs for the auxiliary policy experiments.
    disable_loggers: bool = True


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

    with set_exploration_type(exploration):
        for batch in collector:
            if batch is None:
                continue
            batch = batch.detach()
            if collect_cfg.flatten_batch:
                try:
                    batch = batch.reshape(-1)
                except Exception:
                    pass

            # Save shard(s)
            if n_batches % int(collect_cfg.save_every_n_batches) == 0:
                shard_path = out_dir / f"batch_{n_batches:06d}.pt"
                # Save on CPU to make dataset portable.
                try:
                    to_save = batch.to("cpu")
                except Exception:
                    to_save = batch
                torch.save(to_save, shard_path)
                rec = {
                    "batch_idx": n_batches,
                    "path": shard_path.name,
                    "numel": int(batch.numel()),
                }
                with index_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

            # Best-effort frame count
            try:
                bs0 = int(batch.batch_size[0]) if hasattr(batch, "batch_size") and len(batch.batch_size) else int(batch.numel())
            except Exception:
                bs0 = int(batch.numel())
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


