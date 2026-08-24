"""Double-critic actor and PPO for rsl-rl 3.0.1 (BeamDojo paper).

Inject into ``rsl_rl.runners.on_policy_runner`` before constructing OnPolicyRunner
so ``eval(class_name)`` can see ``ActorCriticDouble`` / ``PPODoubleCritic``.

Paper: critic 1 = dense locomotion, critic 2 = sparse foothold,
``A = w1 * n(A1) + w2 * n(A2)`` with w1=1.0, w2=0.25. MLP [512, 216, 128].
"""

from __future__ import annotations

from beamdojo_mdp.advantage import W1, W2, combine_advantages, gae_advantages

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
        loco = rewards - foot
        step = self.storage.step
        if self.foot_rewards is not None:
            self.foot_rewards[step].copy_(foot.view(-1, 1))
            if self._pending_foot_value is not None:
                self.foot_values[step].copy_(self._pending_foot_value.view(-1, 1))
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

    def update(self):
        loss_dict = super().update()
        if self.foot_returns is None or self.foot_optimizer is None:
            return loss_dict
        obs = self.storage.observations.flatten(0, 1)
        target = self.foot_returns.flatten(0, 1)
        pred = self.policy.evaluate_foothold(obs)
        extra = (pred - target).pow(2).mean()
        self.foot_optimizer.zero_grad()
        extra.backward()
        nn.utils.clip_grad_norm_(self.policy.critic_foothold.parameters(), self.max_grad_norm)
        self.foot_optimizer.step()
        loss_dict["value_foothold"] = float(extra.item())
        return loss_dict


def _foothold_from_extras(extras, rewards):
    if extras is None:
        return torch.zeros_like(rewards)
    foot = extras.get("foothold_reward")
    if foot is None and isinstance(extras.get("log"), dict):
        foot = extras["log"].get("foothold_reward")
    if foot is None:
        return torch.zeros_like(rewards)
    if not torch.is_tensor(foot):
        foot = torch.as_tensor(foot, device=rewards.device, dtype=rewards.dtype)
    return foot.to(device=rewards.device, dtype=rewards.dtype).reshape_as(rewards)
