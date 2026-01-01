import math
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Type, Any, Union

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

from benchmarl.algorithms.mbpo import _DynamicsModel
from benchmarl.utils import _class_from_name

import torch_geometric

class _GNNDynamicsModel(_DynamicsModel):
    """GNN-based dynamics model that predicts next observations, rewards and done logits for a whole agent group."""

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
        topology: str = "full",
        gnn_class: Type[torch_geometric.nn.MessagePassing] = torch_geometric.nn.conv.GraphConv,
        gnn_kwargs: dict = {},
        n_agents: int = 0,
        self_loops: bool = False,
        edge_radius: float = 1.0,
        pos_dim: int = 2,
        obs_dim_per_agent: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(
            input_dim=input_dim,
            next_obs_dim=next_obs_dim,
            reward_dim=reward_dim,
            done_dim=done_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            stochastic=stochastic,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
        )

        try:
            import torch_geometric
            from torch_geometric.utils import dense_to_sparse, remove_self_loops
        except ImportError:
            raise ImportError(
                "torch_geometric is required for GNN-based MBPO. "
                "Install it with: pip install torch-geometric"
            )

        # GNN-Dynamics model needs to replace self.net with GNN layers
        # GNN always works with per-agent/node features
        # Compute per-agent input dimension:
        # - In centralized mode: input_dim = n_agents * (obs_dim + action_dim), so per-agent = input_dim // n_agents
        # - In per-agent mode: input_dim = obs_dim + action_dim, so per-agent = input_dim
        # We can infer the mode: if next_obs_dim is divisible by n_agents, it's likely centralized
        if n_agents is not None and n_agents > 0:
            # Check if centralized mode (next_obs_dim = n_agents * obs_dim_per_agent)
            if next_obs_dim % n_agents == 0:
                # Centralized mode: input_dim is total, divide by n_agents
                current_dim = input_dim // n_agents
            else:
                # Per-agent mode: input_dim is already per-agent
                current_dim = input_dim
        else:
            current_dim = input_dim

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_kwargs = gnn_kwargs.copy()
            layer_kwargs.update({
                "in_channels": current_dim,
                "out_channels": hidden_size,
            })
            gnn_layer = gnn_class(**layer_kwargs)
            self.gnn_layers.append(gnn_layer)
            current_dim = hidden_size

        # Create edge index based on topology
        if topology == "full" and n_agents is not None:
            adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)
            edge_index, _ = dense_to_sparse(adjacency)
            if not self_loops:
                edge_index, _ = remove_self_loops(edge_index)
            self.register_buffer("edge_index", edge_index)
        elif topology == "empty":
            if self_loops and n_agents is not None:
                edge_index = (
                    torch.arange(n_agents, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .repeat(2, 1)
                )
            else:
                edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
            self.register_buffer("edge_index", edge_index)
        else:
            # For "from_pos" topology, edge_index will be computed dynamically
            self.edge_index = None

        self.n_agents = n_agents
        self.topology = topology
        self.edge_radius = edge_radius
        self.pos_dim = pos_dim
        self.self_loops = self_loops

        self._stochastic = stochastic
        # Bounds used to keep log-variances well-behaved (softplus clamp like PETS/MAMBPO)
        self.register_buffer("_min_log_var", torch.tensor(float(min_log_var)))
        self.register_buffer("_max_log_var", torch.tensor(float(max_log_var)))

        # Compute per-agent observation dimension
        # For centralized: next_obs_dim = n_agents * obs_dim_per_agent
        # For per-agent: next_obs_dim = obs_dim_per_agent
        if obs_dim_per_agent is not None:
            self._obs_dim_per_agent = obs_dim_per_agent
        elif n_agents is not None and n_agents > 0:
            # Infer from next_obs_dim: if it's divisible by n_agents, it's centralized
            if next_obs_dim % n_agents == 0:
                self._obs_dim_per_agent = next_obs_dim // n_agents
            else:
                # Per-agent mode: next_obs_dim is already per-agent
                self._obs_dim_per_agent = next_obs_dim
        else:
            # Fallback: assume per-agent mode
            self._obs_dim_per_agent = next_obs_dim

        # Each agent predicts only its own: obs_dim + reward (1)
        # So per-agent output is: obs_dim_per_agent + 1
        per_agent_output_dim = self._obs_dim_per_agent + 1
        
        # Predict mean(delta_next_obs), mean(reward), and optionally log-variances; plus done logits.
        # Each agent node predicts only its own state, reward, and done
        self.mu_head = nn.Linear(current_dim, per_agent_output_dim)
        self.log_var_head = nn.Linear(current_dim, per_agent_output_dim)
        self.done_head = nn.Linear(current_dim, 1)  # Each agent predicts its own done

        # Store the expected output dimensions (for reshaping later)
        self._next_obs_dim = next_obs_dim  # Total output obs dim (may be centralized)
        self._reward_dim = reward_dim  # Total output reward dim (may be centralized)
        self._done_dim = done_dim  # Total output done dim (may be centralized)

    def _compute_dynamic_edges(
        self, obs: torch.Tensor, batch_size: int, n_agents: int
    ) -> torch.Tensor:
        """
        Compute edge_index dynamically based on agent positions.
        
        Args:
            obs: [B*n_agents, obs_dim] or [B, n_agents, obs_dim] observations
            batch_size: Batch size
            n_agents: Number of agents
            
        Returns:
            edge_index: [2, num_edges] tensor of edge indices
        """
        from torch_geometric.utils import dense_to_sparse
        
        # Check that observation has enough dimensions for position extraction
        obs_dim = obs.shape[-1]
        if obs_dim < self.pos_dim:
            raise ValueError(
                f"Observation dimension ({obs_dim}) is less than pos_dim ({self.pos_dim}). "
                f"Cannot extract position information for dynamic graph construction."
            )
        
        # Extract position information (first pos_dim dimensions of observation)
        if obs.dim() == 3:
            # [B, n_agents, obs_dim]
            pos = obs[..., :self.pos_dim]  # [B, n_agents, pos_dim]
        else:
            # [B*n_agents, obs_dim]
            pos = obs[:, :self.pos_dim]  # [B*n_agents, pos_dim]
            pos = pos.view(batch_size, n_agents, self.pos_dim)  # [B, n_agents, pos_dim]
        
        # Compute pairwise distances: [B, n_agents, n_agents]
        # pos: [B, n_agents, pos_dim]
        pos_i = pos.unsqueeze(2)  # [B, n_agents, 1, pos_dim]
        pos_j = pos.unsqueeze(1)  # [B, 1, n_agents, pos_dim]
        distances = torch.norm(pos_i - pos_j, dim=-1)  # [B, n_agents, n_agents]
        
        # Create adjacency matrix based on edge_radius
        adjacency = distances <= self.edge_radius  # [B, n_agents, n_agents]
        
        # Remove self-loops if not desired
        if not self.self_loops:
            mask = ~torch.eye(n_agents, dtype=torch.bool, device=obs.device).unsqueeze(0)
            adjacency = adjacency & mask
        
        # Convert to edge_index format for each batch
        edge_indices = []
        for b in range(batch_size):
            # Get edges for this batch
            adj_b = adjacency[b]  # [n_agents, n_agents]
            edge_index_b, _ = dense_to_sparse(adj_b.long())
            # Add batch offset
            edge_index_b = edge_index_b + b * n_agents
            edge_indices.append(edge_index_b)
        
        # Concatenate all edges
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]
        else:
            edge_index = torch.empty((2, 0), device=obs.device, dtype=torch.long)
        
        return edge_index

    def forward(
        self, obs_flat: torch.Tensor, action_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN dynamics model.
        
        Args:
            obs_flat: [B, n_agents*obs_dim] or [B, n_agents, obs_dim] observations
            action_flat: [B, n_agents*action_dim] or [B, n_agents, action_dim] actions
            
        Returns:
            mu_next_obs, lv_delta, mu_rew, lv_rew, done_logit
        """


        # Handle input shapes.
        # Expected inputs are:
        # - [B, n_agents, dim] (per-agent structure), or
        # - [B, n_agents * dim] (centralized flattened joint vectors).
        input_was_3d = obs_flat.dim() == 3
        if input_was_3d:
            batch_size = obs_flat.shape[0]
            n_agents = obs_flat.shape[1]
            obs_original = obs_flat  # Keep original for position extraction
            obs_flat = obs_flat.view(-1, obs_flat.shape[-1])  # [B*n_agents, obs_dim]
            action_flat = action_flat.view(-1, action_flat.shape[-1])  # [B*n_agents, action_dim]
        else:
            if self.n_agents is None:
                raise ValueError(
                    f"n_agents must be specified when input is 2D, but got {self.n_agents}"
                )
            batch_size = obs_flat.shape[0]  # B
            n_agents = self.n_agents
            obs_original = obs_flat.view(batch_size, n_agents, -1)  # [B, n_agents, obs_dim]
            action_original = action_flat.view(batch_size, n_agents, -1)  # [B, n_agents, action_dim]
            obs_flat = obs_original.view(-1, obs_original.shape[-1])  # [B*n_agents, obs_dim]
            action_flat = action_original.view(-1, action_original.shape[-1])  # [B*n_agents, action_dim]

        # Concatenate obs and action for node features
        x = torch.cat([obs_flat, action_flat], dim=-1)  # [B*n_agents, obs_dim + action_dim]

        # Get or create edge index
        if self.edge_index is not None:
            edge_index = self.edge_index
            # Expand edge_index for batch
            if batch_size > 1:
                n_edges = edge_index.shape[1]
                # Create batch offsets: [0, n_agents, 2*n_agents, ...]
                batch_offsets = torch.arange(batch_size, device=x.device) * n_agents
                # Expand edge_index: [2, n_edges] -> [2, n_edges * batch_size]
                edge_index_expanded = edge_index.repeat(1, batch_size)
                # Add offsets to each batch's edges
                offsets = batch_offsets.repeat_interleave(n_edges).unsqueeze(0).repeat(2, 1)
                edge_index = edge_index_expanded + offsets
        else:
            # For "from_pos" topology, compute edges dynamically based on positions
            if self.topology == "from_pos":
                edge_index = self._compute_dynamic_edges(obs_original, batch_size, n_agents)
            else:
                raise ValueError(f"Unknown topology: {self.topology}")

        # Process through GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)

        # Reshape back to [B, n_agents, hidden_size] - preserve per-agent features
        # x is [B*n_agents, hidden_size] after GNN
        x = x.view(n_agents, batch_size, -1).permute(1,0,2) #x.view(batch_size, n_agents, -1)  # [B, n_agents, hidden_size]

        # Apply output heads per-agent
        # Each agent predicts only its own: obs_dim_per_agent + reward (1) + done (1)
        x_flat = x.reshape(-1, x.shape[-1])  # [B*n_agents, hidden_size]
        mu_all = self.mu_head(x_flat)  # [B*n_agents, obs_dim_per_agent + 1 + 1]
        log_var_all = self.log_var_head(x_flat)  # [B*n_agents, obs_dim_per_agent + 1 + 1]
        done_logit = self.done_head(x_flat)  # [B*n_agents, 1]
        
        # Split per-agent predictions: each agent predicts [obs_dim, reward, done]
        mu_delta_per_agent = mu_all[..., :self._obs_dim_per_agent]  # [B*n_agents, obs_dim_per_agent]
        mu_rew_per_agent = mu_all[..., self._obs_dim_per_agent:self._obs_dim_per_agent + 1]  # [B*n_agents, 1]
        lv_delta_per_agent = log_var_all[..., :self._obs_dim_per_agent]  # [B*n_agents, obs_dim_per_agent]
        lv_rew_per_agent = log_var_all[..., self._obs_dim_per_agent:self._obs_dim_per_agent + 1]  # [B*n_agents, 1]

        # Softplus-based clamp to [min_log_var, max_log_var]
        max_lv = self._max_log_var
        min_lv = self._min_log_var
        lv_delta_per_agent = max_lv - F.softplus(max_lv - lv_delta_per_agent)
        lv_delta_per_agent = min_lv + F.softplus(lv_delta_per_agent - min_lv)
        lv_rew_per_agent = max_lv - F.softplus(max_lv - lv_rew_per_agent)
        lv_rew_per_agent = min_lv + F.softplus(lv_rew_per_agent - min_lv)

        if not self._stochastic:
            lv_delta_per_agent = lv_delta_per_agent * 0.0
            lv_rew_per_agent = lv_rew_per_agent * 0.0

        # Reshape to [B, n_agents, ...] for aggregation
        mu_delta = mu_delta_per_agent.view(batch_size, n_agents, self._obs_dim_per_agent)  # [B, n_agents, obs_dim_per_agent]
        mu_rew = mu_rew_per_agent.view(batch_size, n_agents, 1)  # [B, n_agents, 1]
        lv_delta = lv_delta_per_agent.view(batch_size, n_agents, self._obs_dim_per_agent)  # [B, n_agents, obs_dim_per_agent]
        lv_rew = lv_rew_per_agent.view(batch_size, n_agents, 1)  # [B, n_agents, 1]
        done_logit = done_logit.view(batch_size, n_agents, 1)  # [B, n_agents, 1]

        # Predict next obs as delta
        # Reshape obs_flat to [B, n_agents, obs_dim_per_agent] for per-agent addition
        # Verify that obs_flat has the correct per-agent dimension
        obs_flat = obs_flat.reshape(batch_size, n_agents, -1)
        obs_dim_from_input = obs_flat.shape[-1]
        if obs_dim_from_input != self._obs_dim_per_agent:
            raise ValueError(
                f"Observation dimension mismatch: obs_flat.shape[-1]={obs_dim_from_input}, "
                f"but model expects obs_dim_per_agent={self._obs_dim_per_agent}. "
                f"This suggests a mismatch between model initialization and input data."
            )
        obs_reshaped = obs_flat.view(batch_size, n_agents, self._obs_dim_per_agent)  # [B, n_agents, obs_dim_per_agent]
        mu_next_obs_per_agent = obs_reshaped + mu_delta  # [B, n_agents, obs_dim_per_agent]

        # Aggregate per-agent predictions to match expected output format
        # For centralized: flatten to [B, n_agents * obs_dim_per_agent] = [B, next_obs_dim]
        # For per-agent: flatten to [B*n_agents, obs_dim_per_agent] = [B*n_agents, next_obs_dim]
        if self._next_obs_dim == n_agents * self._obs_dim_per_agent:
            # Centralized mode: concatenate all agents' predictions
            mu_next_obs = mu_next_obs_per_agent.reshape(batch_size, -1)  # [B, n_agents * obs_dim_per_agent]
            lv_delta = lv_delta.reshape(batch_size, -1)  # [B, n_agents * obs_dim_per_agent]
            mu_rew = mu_rew.reshape(batch_size, n_agents)  # [B, n_agents] (squeeze last dim)
            lv_rew = lv_rew.reshape(batch_size, n_agents)  # [B, n_agents]
            done_logit = done_logit.reshape(batch_size, n_agents)  # [B, n_agents]
        else:
            # Per-agent mode: keep as [B*n_agents, obs_dim_per_agent]
            mu_next_obs = mu_next_obs_per_agent.reshape(-1, self._obs_dim_per_agent)  # [B*n_agents, obs_dim_per_agent]
            lv_delta = lv_delta.reshape(-1, self._obs_dim_per_agent)  # [B*n_agents, obs_dim_per_agent]
            mu_rew = mu_rew.reshape(-1, 1)  # [B*n_agents, 1]
            lv_rew = lv_rew.reshape(-1, 1)  # [B*n_agents, 1]
            done_logit = done_logit.reshape(-1, 1)  # [B*n_agents, 1]

        return mu_next_obs, lv_delta, mu_rew, lv_rew, done_logit


class _GmpoWorldModelMixin:
    """GNN-MBPO world-model implementation shared across different base algorithms.
    
    This is similar to _MbpoWorldModelMixin but always uses GNN-based dynamics models.
    """

    def replay_buffer_memory_size_multiplier(self) -> float:
        # If we are not mixing synthetic data (e.g., RNN case), don't inflate.
        if getattr(self, "has_rnn", False):
            return 1.0
        rr = float(getattr(self, "real_ratio", 1.0))
        rr = min(max(rr, 0.0), 1.0)
        horizon = int(getattr(self, "rollout_horizon", 0))
        horizon = max(horizon, 0)
        return 1.0 + (1.0 - rr) * float(horizon)

    def _gmpo_init(
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
        # GNN-specific parameters
        topology: str = "full",
        gnn_class: Union[Type[torch_geometric.nn.MessagePassing], str] = torch_geometric.nn.conv.GraphConv,
        gnn_kwargs: dict = {},
        self_loops: bool = False,
        edge_radius: float = 1.0,
        pos_dim: int = 2,
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
        # GNN-specific parameters
        self.topology = topology
        # Convert string class name to actual class if needed (e.g., from YAML config)
        if isinstance(gnn_class, str):
            self.gnn_class = _class_from_name(gnn_class)
        else:
            self.gnn_class = gnn_class
        self.gnn_kwargs = gnn_kwargs if gnn_kwargs is not None else {}
        self.self_loops = self_loops
        self.edge_radius = edge_radius
        self.pos_dim = pos_dim

        # Always use GNN dynamics
        self.gnn_dynamics = True
        self._dynamics: Dict[str, List[_GNNDynamicsModel]] = {}
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

            models: List[_GNNDynamicsModel] = []
            opts: List[torch.optim.Optimizer] = []
            for _ in range(self.ensemble_size):
                model = _GNNDynamicsModel(
                    input_dim=dyn_input_dim,
                    next_obs_dim=next_obs_dim_out,
                    reward_dim=reward_dim_out,
                    done_dim=done_dim_out,
                    hidden_size=self.model_hidden_size,
                    num_layers=self.model_num_layers,
                    stochastic=self.stochastic_dynamics,
                    min_log_var=self.min_log_var,
                    max_log_var=self.max_log_var,
                    topology=self.topology,
                    gnn_class=self.gnn_class,
                    gnn_kwargs=self.gnn_kwargs,
                    n_agents=n_agents,
                    self_loops=self.self_loops,
                    edge_radius=self.edge_radius,
                    pos_dim=self.pos_dim,
                    device=self.device,
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

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = super().process_batch(group, batch)
        flat_batch = batch.reshape(-1) if not self.has_rnn else batch

        self._train_steps[group] += 1
        if self._train_steps[group] % self.model_train_freq == 0:
            self._train_dynamics(group, flat_batch)
            if self._train_steps[group] > self.warmup_steps:
                synthetic = self._generate_model_rollouts(group, flat_batch)
                if synthetic is not None and synthetic.numel() > 0:
                    synthetic = super().process_batch(group, synthetic)
                    if not self.has_rnn:
                        synthetic = synthetic.reshape(-1)
                    # Mix synthetic transitions into training by returning an augmented batch.
                    # The Experiment loop will extend the replay buffer with whatever we return.
                    if not self.has_rnn:
                        batch = torch.cat([batch.reshape(-1), synthetic], dim=0)
                    else:
                        # RNN batches have sequence structure; mixing is not supported yet.
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


class Gmpo(_GmpoWorldModelMixin, Masac):
    """GNN-based Model-Based Policy Optimisation (GNN-MBPO) built on top of MASAC.

    This implementation keeps MASAC losses/policies and augments training with a
    learned GNN-based ensemble dynamics model that generates short synthetic rollouts,
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
        # GNN-specific parameters
        topology: str = "full",
        gnn_class: Optional[Type] = None,
        gnn_kwargs: Optional[dict] = None,
        self_loops: bool = False,
        edge_radius: float = 1.0,
        pos_dim: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gmpo_init(
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
            topology=topology,
            gnn_class=gnn_class,
            gnn_kwargs=gnn_kwargs,
            self_loops=self_loops,
            edge_radius=edge_radius,
            pos_dim=pos_dim,
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


class GmpoMappo(_GmpoWorldModelMixin, Mappo):
    """GNN-based Model-Based Policy Optimisation (GNN-MBPO) built on top of MAPPO.

    Note: MAPPO is on-policy; GNN-MBPO-style synthetic rollouts will be mixed into the
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
        # GNN-specific parameters
        topology: str = "full",
        gnn_class: Optional[Type] = None,
        gnn_kwargs: Optional[dict] = None,
        edge_radius: float = 1.0,
        pos_dim: int = 2,
        self_loops: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gmpo_init(
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
            topology=topology,
            gnn_class=gnn_class,
            gnn_kwargs=gnn_kwargs,
            edge_radius=edge_radius,
            pos_dim=pos_dim,
            self_loops=self_loops,
        )


@dataclass
class GmpoConfig(MasacConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Gmpo`."""

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
    # GNN-specific parameters
    topology: str = "full"
    gnn_class: Optional[Any] = None  # Type annotation changed to Any for OmegaConf compatibility
    gnn_kwargs: Optional[dict] = None
    edge_radius: float = 1.0
    pos_dim: int = 2
    self_loops: bool = False

    @staticmethod
    def associated_class() -> Type[Masac]:
        return Gmpo

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
class GmpoMappoConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.GmpoMappo`."""

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
    # GNN-specific parameters
    topology: str = "full"
    gnn_class: Optional[Any] = None  # Type annotation changed to Any for OmegaConf compatibility
    gnn_kwargs: Optional[dict] = None
    edge_radius: float = 1.0
    pos_dim: int = 2
    self_loops: bool = False

    @staticmethod
    def associated_class() -> Type[Mappo]:
        return GmpoMappo

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

