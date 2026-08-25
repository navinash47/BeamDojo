"""Pull the sparse foothold term out of Isaac / RSL-RL extras dicts."""


def foothold_term_from_extras(extras):
    """Return the per-env foothold tensor/array, or None if extras has no term.

    Isaac Lab logs the episode mean as ``Episode_Reward/foothold_penalty``.
    The gym wrapper writes the per-env term on the extras dict (not under
    ``extras['log']``, which rsl-rl 3.0.1 treats as episode scalars).
    """
    if not isinstance(extras, dict):
        return None
    keys = ("foothold_reward", "foothold_penalty")
    for key in keys:
        val = extras.get(key)
        if val is not None:
            return val
    log = extras.get("log")
    if isinstance(log, dict):
        for key in keys:
            val = log.get(key)
            if val is not None:
                return val
    return None
