"""Double-critic actor and PPO for rsl-rl 3.0.1 (BeamDojo paper).

Inject into ``rsl_rl.runners.on_policy_runner`` before constructing OnPolicyRunner
so ``eval(class_name)`` can see ``ActorCriticDouble`` / ``PPODoubleCritic``.

Paper: critic 1 = dense locomotion, critic 2 = sparse foothold,
``A = w1 * n(A1) + w2 * n(A2)`` with w1=1.0, w2=0.25. MLP [512, 216, 128].
"""

from __future__ import annotations

from beamdojo_mdp.advantage import W1, W2, bootstrap_timeouts, combine_advantages, gae_advantages
from beamdojo_mdp.foothold_extras import foothold_term_from_extras

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    from rsl_rl.networks import MLP
except ImportError:  # CPU unit tests / no Isaac
    PPO = object  # type: ignore
    ActorCritic = object  # type: ignore
    MLP = None  # type: ignore
    nn = None  # type: ignore
    torch = None  # type: ignore
    optim = None  # type: ignore


class ActorCriticDouble(ActorCritic):
    """Actor + two critics (locomotion and foothold). rsl-rl 3.0.1 signature."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 216, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 216, 128]
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1]
        self.critic_foothold = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Foothold critic MLP: {self.critic_foothold}")

    def evaluate_foothold(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic_foothold(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load actor + critics. Missing critic_foothold → random init, no optimizer resume.

        rsl-rl 3.0.1 ``OnPolicyRunner.load`` skips the optimizer when this returns False.
        That is required when the checkpoint is a single-critic Stage 1 smoke: the new
        loco-only Adam param groups would not match the saved optimizer.
        """
        has_foot = any(str(key).startswith("critic_foothold") for key in state_dict)
        if not has_foot:
            print(
                "[WARN] Checkpoint has no critic_foothold. Loading actor/critic 1 only; "
                "critic 2 stays random and the optimizer will not resume."
            )
            nn.Module.load_state_dict(self, state_dict, strict=False)
            return False
        nn.Module.load_state_dict(self, state_dict, strict=strict)
        return True


class PPODoubleCritic(PPO):
    """PPO with two value heads and mixed normalized advantages."""

    def __init__(self, policy, w1: float = W1, w2: float = W2, **kwargs):
        super().__init__(policy, **kwargs)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.foot_rewards = None
        self.foot_values = None
        self.foot_returns = None
        self._pending_foot_value = None
        self.foot_optimizer = None
        if hasattr(policy, "critic_foothold") and optim is not None:
            self.optimizer = optim.Adam(
                [p for n, p in policy.named_parameters() if not n.startswith("critic_foothold")],
                lr=self.learning_rate,
            )
            self.foot_optimizer = optim.Adam(policy.critic_foothold.parameters(), lr=self.learning_rate)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        dev = self.device
        self.foot_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=dev)
        self.foot_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=dev)
        self.foot_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=dev)

    def act(self, obs):
        actions = super().act(obs)
        if hasattr(self.policy, "evaluate_foothold"):
            self._pending_foot_value = self.policy.evaluate_foothold(obs).detach()
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        foot = _foothold_from_extras(extras, rewards)
        # Split the env reward first. Timeout bootstrap is not part of R; rsl-rl
        # adds gamma * V * timeout onto the stored reward (Stage 1 is timeout-only).
        loco = rewards - foot
        step = self.storage.step
        if self.foot_rewards is not None:
            stored = _timeout_bootstrap_reward(foot, self._pending_foot_value, extras, self.gamma)
            self.foot_rewards[step].copy_(stored.reshape(-1, 1))
            if self._pending_foot_value is not None:
                self.foot_values[step].copy_(self._pending_foot_value.reshape(-1, 1))
        super().process_env_step(obs, loco, dones, extras)

    def compute_returns(self, obs):
        last_loco = self.policy.evaluate(obs).detach()
        last_foot = (
            self.policy.evaluate_foothold(obs).detach()
            if hasattr(self.policy, "evaluate_foothold")
            else last_loco
        )
        super().compute_returns(obs)
        if self.foot_rewards is None:
            return
        last_loco = last_loco.reshape(-1, 1)
        last_foot = last_foot.reshape(-1, 1)
        a2, r2 = gae_advantages(
            self.foot_rewards,
            self.foot_values,
            self.storage.dones,
            last_foot,
            gamma=self.gamma,
            lam=self.lam,
        )
        self.foot_returns.copy_(r2)
        # Recompute loco GAE without in-storage normalization, then mix.
        a1, r1 = gae_advantages(
            self.storage.rewards,
            self.storage.values,
            self.storage.dones,
            last_loco,
            gamma=self.gamma,
            lam=self.lam,
        )
        mixed = combine_advantages(a1, a2, w1=self.w1, w2=self.w2)
        self.storage.advantages.copy_(mixed)
        self.storage.returns.copy_(r1)

    def _update_foothold_critic(self):
        """Fit critic 2 before PPO.update() clears rollout bookkeeping."""
        if self.foot_returns is None or self.foot_optimizer is None:
            return None
        obs = self.storage.observations.flatten(0, 1)
        target = self.foot_returns.flatten(0, 1)
        pred = self.policy.evaluate_foothold(obs)
        extra = (pred.reshape_as(target) - target).pow(2).mean()
        self.foot_optimizer.zero_grad()
        extra.backward()
        nn.utils.clip_grad_norm_(self.policy.critic_foothold.parameters(), self.max_grad_norm)
        self.foot_optimizer.step()
        return float(extra.item())

    def update(self):
        extra = self._update_foothold_critic()
        loss_dict = super().update()
        if extra is not None:
            loss_dict["value_foothold"] = extra
        return loss_dict


def _foothold_from_extras(extras, rewards):
    foot = foothold_term_from_extras(extras)
    if foot is None:
        return torch.zeros_like(rewards)
    if not torch.is_tensor(foot):
        foot = torch.as_tensor(foot, device=rewards.device, dtype=rewards.dtype)
    return foot.to(device=rewards.device, dtype=rewards.dtype).reshape_as(rewards)


def _flat_n(tensor, n, *, device, dtype):
    flat = torch.as_tensor(tensor, device=device, dtype=dtype).reshape(-1)
    if flat.numel() == 1:
        return flat.expand(n).clone()
    if flat.numel() != n:
        raise ValueError(f"timeout-bootstrap length {flat.numel()} != {n}")
    return flat


def _timeout_bootstrap_reward(rewards, values, extras, gamma: float):
    """Same formula as rsl-rl 3.0.1 PPO.process_env_step timeout bootstrap."""
    if extras is None or "time_outs" not in extras or values is None:
        return rewards
    n = int(rewards.numel())
    device = rewards.device
    dtype = rewards.dtype
    boot = bootstrap_timeouts(
        rewards.reshape(n),
        _flat_n(values, n, device=device, dtype=dtype),
        _flat_n(extras["time_outs"], n, device=device, dtype=dtype),
        gamma,
    )
    return boot.reshape_as(rewards)
