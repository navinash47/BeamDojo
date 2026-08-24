"""Double-critic advantage mix (BeamDojo paper: w1=1.0, w2=0.25). Isaac-free."""

from __future__ import annotations

W1 = 1.0
W2 = 0.25


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def normalize_adv(adv, eps: float = 1e-8):
    return (adv - adv.mean()) / (adv.std() + eps)


def combine_advantages(adv_loco, adv_foot, w1: float = W1, w2: float = W2, eps: float = 1e-8):
    """A = w1 · normalize(A_loco) + w2 · normalize(A_foot)."""
    return w1 * normalize_adv(adv_loco, eps=eps) + w2 * normalize_adv(adv_foot, eps=eps)


def bootstrap_timeouts(rewards, values, timeouts, gamma: float = 0.99):
    """Match rsl-rl 3.0.1 PPO.process_env_step: R += gamma * V * timeout.

    Inputs must already be the same shape (e.g. both length-N). Do not pass a
    length-N vector against an [N, 1] column — numpy/torch will broadcast wrong.
    """
    return rewards + gamma * values * timeouts


def gae_advantages(rewards, values, dones, last_values, gamma: float = 0.99, lam: float = 0.95):
    """Generalized advantage estimation. Tensors/arrays shaped [T, N] or [T, N, 1]."""
    if _is_torch(rewards):
        import torch

        t_steps = rewards.shape[0]
        advantage = torch.zeros((), device=rewards.device)
        advantages = torch.zeros_like(rewards)
        for step in reversed(range(t_steps)):
            next_values = last_values if step == t_steps - 1 else values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            advantages[step] = advantage
        returns = advantages + values
        return advantages, returns

    import numpy as np

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)
    t_steps = rewards.shape[0]
    advantage = np.zeros(rewards.shape[1:], dtype=np.float64)
    advantages = np.zeros_like(rewards)
    for step in reversed(range(t_steps)):
        next_values = last_values if step == t_steps - 1 else values[step + 1]
        next_is_not_terminal = 1.0 - dones[step]
        delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        advantages[step] = advantage
    returns = advantages + values
    return advantages, returns
