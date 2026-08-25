"""Shared BeamDojo runtime helpers: GPU gate, env registration, NFS logs, W&B status."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
for _path in (REPO_ROOT, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from live_metrics import append_history, extract_live_metrics

_REGISTERED = False

TASK_IDS = {
    (1, "h1", "beam"): "Isaac-BeamDojo-Stage1-H1-v0",
    (2, "h1", "beam"): "Isaac-BeamDojo-Stage2-H1-v0",
    (2, "h1", "stones"): "Isaac-BeamDojo-Stage2-H1-Stones-v0",
    (1, "g1", "beam"): "Isaac-BeamDojo-Stage1-G1-v0",
    (2, "g1", "beam"): "Isaac-BeamDojo-Stage2-G1-v0",
    (2, "g1", "stones"): "Isaac-BeamDojo-Stage2-G1-Stones-v0",
}

PLAY_IDS = {
    (1, "h1", "beam"): "Isaac-BeamDojo-Stage1-H1-Play-v0",
    (2, "h1", "beam"): "Isaac-BeamDojo-Stage2-H1-Play-v0",
    (2, "h1", "stones"): "Isaac-BeamDojo-Stage2-H1-Stones-Play-v0",
    (1, "g1", "beam"): "Isaac-BeamDojo-Stage1-G1-Play-v0",
    (2, "g1", "beam"): "Isaac-BeamDojo-Stage2-G1-Play-v0",
    (2, "g1", "stones"): "Isaac-BeamDojo-Stage2-G1-Stones-Play-v0",
}


def require_gpu_device(device: str | None) -> None:
    """Refuse CPU before Isaac Sim starts. Visualization and training must be CUDA."""
    if device is None:
        return
    if str(device).strip().lower().startswith("cpu"):
        raise SystemExit(
            "BeamDojo refuses CPU. Pass --device cuda:0 on an NVIDIA GPU with RT cores (A10/A6000/L40S)."
        )


def require_cuda() -> None:
    """Abort if PyTorch cannot see CUDA. Call after torch is imported."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. BeamDojo will not train or render on CPU. "
            "Run on the Lambda A10 (or another RT-core NVIDIA GPU)."
        )
    if torch.cuda.device_count() < 1:
        raise SystemExit("No CUDA devices found. Refusing to continue.")


_DISTRIBUTED_ENV_KEYS = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "GROUP_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def clear_stale_distributed_env(*, distributed: bool = False) -> list[str]:
    """Drop leftover torchrun env so OnPolicyRunner does not NCCL-init.

    rsl-rl 3.0.1 treats ``WORLD_SIZE > 1`` as multi-GPU and calls
    ``init_process_group`` in ``OnPolicyRunner.__init__`` — before ``learn()``
    and ``wandb.init``. A stale Lambda/Docker ``WORLD_SIZE`` kills the 10k
    with no live W&B page.
    """
    if distributed:
        return []
    world = os.environ.get("WORLD_SIZE")
    try:
        world_n = int(str(world).strip()) if world is not None and str(world).strip() != "" else 1
    except ValueError:
        world_n = 1
        os.environ.pop("WORLD_SIZE", None)
    if world_n <= 1:
        return []
    cleared: list[str] = []
    for key in _DISTRIBUTED_ENV_KEYS:
        if key not in os.environ:
            continue
        print(
            f"[WARN] Unsetting leftover {key}={os.environ[key]!r} (not --distributed). "
            "rsl-rl 3.0.1 would NCCL-init and die before wandb.init."
        )
        os.environ.pop(key, None)
        cleared.append(key)
    return cleared


def write_boot_status(
    *,
    stage: int = 1,
    robot: str = "h1",
    terrain: str = "beam",
    task: str | None = None,
    note: str | None = None,
    **extra,
):
    """Isaac-free status write so Research Lab is not Idle during AppLauncher."""
    robot = str(robot).lower()
    terrain = str(terrain).lower()
    if not task:
        try:
            task = resolve_task(int(stage), robot, terrain)
        except ValueError:
            task = None
    return write_training_status(
        {
            "status": "unknown",
            "robot": robot,
            "stage": int(stage),
            "terrain": terrain,
            "task": task,
            "iteration": 0,
            "logger": "wandb" if os.environ.get("WANDB_API_KEY", "").strip() else "tensorboard",
            "note": note
            or (
                "Isaac Sim AppLauncher starting on CUDA. Not a live W&B run yet — "
                "status becomes running when learn() opens the logger."
            ),
            **extra,
        }
    )


def resolve_task(stage: int, robot: str, terrain: str = "beam", *, play: bool = False) -> str:
    table = PLAY_IDS if play else TASK_IDS
    key = (int(stage), str(robot).lower(), str(terrain).lower())
    if key not in table:
        raise ValueError(
            f"No gym id for stage={stage} robot={robot} terrain={terrain} play={play}. "
            f"Known: {sorted(table)}"
        )
    return table[key]


def experiment_name(stage: int, robot: str) -> str:
    return f"beamdojo_{str(robot).lower()}_stage{int(stage)}"


def resolve_load_experiment(stage: int, robot: str, *, load_experiment: str | None = None) -> str:
    """Which ``logs/rsl_rl/<name>`` folder to read checkpoints from.

    Stage 2 fine-tunes Stage 1, so the default load experiment is Stage 1.
    Set ``LOAD_EXPERIMENT`` (or ``load_experiment``) to continue an interrupted
    Stage 2 run from ``beamdojo_<robot>_stage2``.
    """
    override = (load_experiment or os.environ.get("LOAD_EXPERIMENT") or "").strip()
    if override:
        return override
    if int(stage) >= 2:
        return experiment_name(1, robot)
    return experiment_name(stage, robot)


DEFAULT_LOAD_RUN = ".*"
DEFAULT_LOAD_CHECKPOINT = r"model_.*.pt"


def stage2_fine_tunes_stage1(
    stage: int,
    robot: str,
    *,
    load_experiment: str | None = None,
) -> bool:
    """True when Stage 2 is loading Stage 1 weights (not continuing Stage 2)."""
    if int(stage) < 2:
        return False
    return resolve_load_experiment(stage, robot, load_experiment=load_experiment) == experiment_name(
        1, robot
    )


def inactive_rsl_optional_cfg(name: str, value) -> bool:
    """True when rsl-rl 3.0.1 must see ``None`` instead of a Hydra leftover.

    OnPolicyRunner enables RND/symmetry with ``if cfg is not None``. An empty
    dict, or Isaac Lab's default ``RslRlRndCfg(weight=0.0)`` dump, is not None
    and then looks up a missing ``rnd_state`` obs group after Isaac boot.
    """
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    if not value:
        return True
    if name == "rnd_cfg":
        try:
            weight = float(value.get("weight") or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        return weight == 0.0 and not value.get("weight_schedule")
    if name == "symmetry_cfg":
        if not value.get("use_data_augmentation") and not value.get("use_mirror_loss"):
            return True
        # Leftover "enabled" flags without a callable still KeyError in PPO.__init__.
        return not value.get("data_augmentation_func")
    return False


DEFAULT_OBS_GROUPS = {"policy": ["policy"], "critic": ["policy"]}


def valid_obs_groups(value) -> bool:
    """True when rsl-rl 3.0.1 ``resolve_obs_groups`` can read this mapping."""
    if not isinstance(value, dict) or not value:
        return False
    policy = value.get("policy")
    if not isinstance(policy, (list, tuple)) or not policy:
        return False
    return all(isinstance(name, str) and name for name in policy)


def beamdojo_obs_groups_ok(value) -> bool:
    """Isaac Lab 2.3.2 locomotion exposes only a ``policy`` obs group.

    Official H1 ``H1RoughPPORunnerCfg`` leaves ``obs_groups = MISSING``. A leftover
    ``critic: ["critic"]`` / ``rnd_state`` / ``privileged`` list passes a shape
    check, then ``resolve_obs_groups`` ValueErrors in ``OnPolicyRunner.__init__``
    — before ``learn()`` / ``wandb.init``.
    """
    if not valid_obs_groups(value):
        return False
    extra = set(value) - {"policy", "critic"}
    if extra:
        return False
    policy = list(value.get("policy") or [])
    critic = list(value["critic"]) if "critic" in value else policy
    return policy == ["policy"] and critic == ["policy"]


def leftover_all_joints(names) -> bool:
    """True for parent locomotion ``joint_names=[".*"]`` (full-body leftover)."""
    if names is None:
        return False
    items = names_as_list(names)
    return len(items) == 1 and items[0] == ".*"


def names_as_list(names) -> list[str]:
    if names is None:
        return []
    items = names if isinstance(names, (list, tuple)) else [names]
    return [str(item) for item in items]


def names_match(left, right) -> bool:
    return tuple(names_as_list(left)) == tuple(names_as_list(right))


# Official H1 regexes that do not ``re.fullmatch`` G1 ``*_joint`` names.
H1_JOINTS_MISS_G1 = {
    ".*_hip_yaw",
    ".*_hip_roll",
    ".*_ankle",
    "torso",
    ".*_elbow",
}
H1_BODIES_MISS_G1 = {".*_ankle_link", ".*ankle_link"}
G1_JOINTS_MISS_H1 = {
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_roll_joint",
    "torso_joint",
}
G1_BODIES_MISS_H1 = {".*_ankle_roll_link"}


def leftover_h1_joints_for_g1(names) -> bool:
    """True when leftover H1 regexes fail Isaac 2.3.2 ``re.fullmatch`` on G1."""
    return any(item in H1_JOINTS_MISS_G1 for item in names_as_list(names))


def leftover_h1_bodies_for_g1(names) -> bool:
    return any(item in H1_BODIES_MISS_G1 for item in names_as_list(names))


def leftover_g1_joints_for_h1(names) -> bool:
    """True when leftover G1 ``*_joint`` regexes miss official H1 joint names."""
    return any(item in G1_JOINTS_MISS_H1 for item in names_as_list(names))


def leftover_g1_bodies_for_h1(names) -> bool:
    return any(item in G1_BODIES_MISS_H1 for item in names_as_list(names))


def leftover_h1_torso_name(names) -> bool:
    return names_as_list(names) == ["torso"]


QUAD_ROBOT_MARKERS = (
    "anymal",
    "anybotics",
    "unitree_a1",
    "/a1/",
    "go1",
    "go2",
    "spot",
    "cassie",
    "digit",
    "aliengo",
)
BEAM_TOP_Z = 0.28  # BEAM_CENTER_Z + BEAM_THICKNESS/2; scene_props imports Isaac.


def leftover_quadruped_robot(env_cfg) -> bool:
    """True when leftover scene.robot is ANYmal/quad USD H1/G1 cannot resolve."""
    blob = _robot_identity_blob(env_cfg)
    return any(marker in blob for marker in QUAD_ROBOT_MARKERS)


def leftover_full_unitree_usd(env_cfg) -> bool:
    """Full H1/G1 USD (not *_minimal) OOMs 1024 envs on A10 24GB before W&B."""
    blob = _robot_identity_blob(env_cfg)
    if "usd" not in blob or "minimal" in blob:
        return False
    return bool(re.search(r"(?:^|[^a-z0-9])(h1|g1)(?:[^a-z0-9]|$)", blob))


def desired_train_robot(env_cfg) -> str | None:
    """``g1`` / ``h1`` from the env class, else USD / leftover fingers."""
    name = type(env_cfg).__name__.lower()
    if re.search(r"(?:^|[^a-z0-9])g1(?:[^a-z0-9]|$)", name):
        return "g1"
    if re.search(r"(?:^|[^a-z0-9])h1(?:[^a-z0-9]|$)", name):
        return "h1"
    kind = env_cfg_robot_kind(env_cfg)
    if kind in ("g1", "h1"):
        return kind
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is not None and getattr(rewards, "joint_deviation_fingers", None) is not None:
        return "g1"
    return None


def _ensure_obs_groups(train_cfg: dict) -> None:
    """Isaac Lab 2.3.2 ``obs_groups`` defaults to ``MISSING``.

    ``OnPolicyRunner.__init__`` calls ``resolve_obs_groups`` *before*
    ``learn()`` / ``wandb.init``. A leftover None / empty / ``MISSING`` dump
    KeyErrors, and a leftover ``critic: ["critic"]`` ValueErrors, so the live
    W&B page never opens.
    """
    groups = train_cfg.get("obs_groups")
    if beamdojo_obs_groups_ok(groups):
        if isinstance(groups, dict) and "critic" not in groups:
            groups["critic"] = list(groups["policy"])
        return
    print(
        "[WARN] Restoring leftover obs_groups for rsl-rl 3.0.1 "
        "(Isaac 2.3.2 locomotion only has 'policy')."
    )
    train_cfg["obs_groups"] = dict(DEFAULT_OBS_GROUPS)


def _reassert_double_critic_class_names(train_cfg: dict) -> None:
    """Hydra leftover restores parent ``ActorCritic`` / ``PPO``.

    That still boots and logs, but it is not BeamDojo double-critic. Distillation
    keeps its own class names. ``ActorCriticRecurrent`` leftover crashes
    ``OnPolicyRunner.__init__`` when ``rnn_type`` is missing — before W&B.
    """
    algorithm = train_cfg.get("algorithm")
    if not isinstance(algorithm, dict):
        return
    if algorithm.get("class_name") == "Distillation":
        return
    if algorithm.get("class_name") != "PPODoubleCritic":
        print("[WARN] Restoring leftover algorithm.class_name to PPODoubleCritic.")
        algorithm["class_name"] = "PPODoubleCritic"
    policy = train_cfg.get("policy")
    if isinstance(policy, dict) and policy.get("class_name") != "ActorCriticDouble":
        print("[WARN] Restoring leftover policy.class_name to ActorCriticDouble.")
        policy["class_name"] = "ActorCriticDouble"


PAPER_MLP_DIMS = [512, 216, 128]


def _reassert_paper_mlp(train_cfg: dict) -> None:
    """Official H1 leftover is ``[512, 256, 128]``. Paper / double critic is ``[512, 216, 128]``."""
    algorithm = train_cfg.get("algorithm")
    if isinstance(algorithm, dict) and algorithm.get("class_name") == "Distillation":
        return
    policy = train_cfg.get("policy")
    if not isinstance(policy, dict):
        return
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        dims = policy.get(key)
        try:
            current = [int(x) for x in list(dims)]
        except (TypeError, ValueError):
            current = []
        if current != PAPER_MLP_DIMS:
            print(f"[WARN] Restoring leftover policy.{key} to paper {PAPER_MLP_DIMS}.")
            policy[key] = list(PAPER_MLP_DIMS)


def _drop_on_policy_runner_kwarg_collisions(train_cfg: dict) -> None:
    """rsl-rl 3.0.1 ``OnPolicyRunner._construct_algorithm`` passes ``device=`` and
    ``multi_gpu_cfg=`` explicitly. A leftover key in the Hydra dump is
    ``TypeError: multiple values for keyword argument`` — before ``wandb.init``.

    Empty ``multi_gpu_cfg: {}`` is also not None, so PPO then KeyErrors
    ``global_rank`` in ``__init__``.
    """
    algorithm = train_cfg.get("algorithm")
    if isinstance(algorithm, dict):
        for key in ("device", "multi_gpu_cfg"):
            if key in algorithm:
                print(f"[WARN] Dropping leftover algorithm.{key} (OnPolicyRunner supplies it).")
                algorithm.pop(key, None)
    policy = train_cfg.get("policy")
    if isinstance(policy, dict):
        for key in ("obs", "obs_groups", "num_actions"):
            if key in policy:
                print(f"[WARN] Dropping leftover policy.{key} (ActorCritic is called positionally).")
                policy.pop(key, None)


def _ensure_runner_intervals(train_cfg: dict) -> None:
    """``OnPolicyRunner.__init__`` KeyErrors if these are leftover-MISSING."""
    if not train_cfg.get("num_steps_per_env"):
        print("[WARN] Restoring leftover num_steps_per_env=24.")
        train_cfg["num_steps_per_env"] = 24
    if not train_cfg.get("save_interval"):
        print("[WARN] Restoring leftover save_interval=100.")
        train_cfg["save_interval"] = 100


def _reassert_noise_std_type(train_cfg: dict) -> None:
    """ActorCritic 3.0.1 ValueErrors on leftover ``noise_std_type`` before W&B."""
    policy = train_cfg.get("policy")
    if not isinstance(policy, dict):
        return
    nst = policy.get("noise_std_type")
    if nst in (None, "scalar", "log"):
        return
    print(f"[WARN] Restoring leftover policy.noise_std_type={nst!r} to 'scalar'.")
    policy["noise_std_type"] = "scalar"


def reassert_runner_class(agent_cfg) -> None:
    """Hydra leftover ``DistillationRunner`` / empty class_name dies at runner construct."""
    name = getattr(agent_cfg, "class_name", None)
    if name in ("OnPolicyRunner", "DistillationRunner"):
        return
    print(f"[WARN] Restoring leftover agent class_name={name!r} to OnPolicyRunner.")
    try:
        agent_cfg.class_name = "OnPolicyRunner"
    except Exception as exc:
        print(f"[WARN] Runner class reassert skipped ({type(exc).__name__}: {exc})")


def _ensure_train_cfg_sections(train_cfg: dict) -> None:
    """``OnPolicyRunner.__init__`` KeyErrors if algorithm/policy dumps are missing."""
    if not isinstance(train_cfg.get("algorithm"), dict):
        print("[WARN] Restoring leftover train_cfg.algorithm dict (OnPolicyRunner requires it).")
        train_cfg["algorithm"] = {}
    if not isinstance(train_cfg.get("policy"), dict):
        print("[WARN] Restoring leftover train_cfg.policy dict (OnPolicyRunner requires it).")
        train_cfg["policy"] = {}


KNOWN_ACTIVATIONS = {"elu", "relu", "selu", "tanh", "sigmoid", "lrelu", "leaky_relu"}


def _reassert_activation(train_cfg: dict) -> None:
    policy = train_cfg.get("policy")
    if not isinstance(policy, dict):
        return
    act = policy.get("activation")
    if act is None:
        return
    if str(act).strip().lower() in KNOWN_ACTIVATIONS:
        return
    print(f"[WARN] Restoring leftover policy.activation={act!r} to 'elu'.")
    policy["activation"] = "elu"


def _reassert_init_noise_std(train_cfg: dict) -> None:
    """ActorCritic 3.0.1 does ``init_noise_std * ones``; leftover None TypeErrors before W&B."""
    policy = train_cfg.get("policy")
    if not isinstance(policy, dict):
        return
    raw = policy.get("init_noise_std", 1.0)
    try:
        std = float(raw)
    except (TypeError, ValueError):
        std = 0.0
    if std != std or std <= 0:
        print(f"[WARN] Restoring leftover policy.init_noise_std={raw!r} to 1.0.")
        policy["init_noise_std"] = 1.0


def _reassert_logger(train_cfg: dict) -> None:
    """Invalid leftover logger ValueErrors at the start of ``learn()`` — no W&B page."""
    logger = train_cfg.get("logger")
    if logger is None:
        return
    text = str(logger).strip().lower()
    if text in {"wandb", "tensorboard", "neptune"}:
        train_cfg["logger"] = text
        return
    fallback = "wandb" if os.environ.get("WANDB_API_KEY", "").strip() else "tensorboard"
    print(f"[WARN] Restoring leftover logger={logger!r} to {fallback}.")
    train_cfg["logger"] = fallback


def leftover_cpu_device(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower().startswith("cpu")


def leftover_unusable_device(value) -> bool:
    """CPU / MPS / leftover ``cuda:1`` on a single A10 dies at gym.make / runner init."""
    if leftover_cpu_device(value):
        return True
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"mps", "xpu", "meta"}:
        return True
    if not text.startswith("cuda"):
        return False
    if text in {"cuda", "cuda:0"}:
        return False
    try:
        world = int(str(os.environ.get("WORLD_SIZE") or "1").strip() or "1")
    except ValueError:
        world = 1
    return world <= 1


def leftover_quadruped_joint_key(key) -> bool:
    """ANYmal/Go2 leftover ``.*HAA`` / ``LF_HFE`` keys miss H1/G1 ``re.fullmatch``."""
    return bool(
        re.search(
            r"(HAA|HFE|KFE|(?:^|[._*])(?:LF|RF|LH|RH|FL|FR|RL|RR)(?:[._*]|$))",
            str(key),
            flags=re.IGNORECASE,
        )
    )


def sanitize_clip_actions(value):
    """Isaac Lab 2.3.2 ``RslRlVecEnvWrapper`` wants ``float | None``.

    Older leftover ``clip_actions=False`` is not None, so the wrapper builds
    ``Box(0, 0)`` and zeros every action. A leftover string/dict TypeErrors in
    ``_modify_action_space`` — after gym.make, before ``wandb.init``.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        clip = float(value)
    except (TypeError, ValueError):
        return None
    if clip != clip or clip <= 0:
        return None
    return clip


def reassert_clip_actions(agent_cfg) -> None:
    if agent_cfg is None or not hasattr(agent_cfg, "clip_actions"):
        return
    raw = getattr(agent_cfg, "clip_actions", None)
    cleaned = sanitize_clip_actions(raw)
    if cleaned == raw:
        return
    print(f"[WARN] Restoring leftover clip_actions={raw!r} to {cleaned!r}.")
    try:
        agent_cfg.clip_actions = cleaned
    except Exception as exc:
        print(f"[WARN] clip_actions reassert skipped ({type(exc).__name__}: {exc})")


def reassert_agent_cuda(agent_cfg) -> None:
    """Leftover ``agent.device=cpu`` / ``cuda:1`` builds ActorCritic off the A10."""
    if agent_cfg is None or not leftover_unusable_device(getattr(agent_cfg, "device", None)):
        return
    print("[WARN] Forcing leftover agent.device onto cuda:0 (single-GPU A10).")
    try:
        agent_cfg.device = "cuda:0"
    except Exception as exc:
        print(f"[WARN] agent.device reassert skipped ({type(exc).__name__}: {exc})")


def sanitize_rsl_rl_train_cfg(train_cfg: dict) -> dict:
    """Drop empty Hydra RND/symmetry dicts before OnPolicyRunner / PPO 3.0.1.

    ``RslRlPpoAlgorithmCfg.rnd_cfg`` defaults to None. OmegaConf sometimes turns
    that into ``{}`` or a default ``RslRlRndCfg`` dump. rsl-rl then treats it as
    enabled (``is not None``), looks up a missing ``rnd_state`` obs group, and
    ``PPO.__init__`` TypeErrors.
    """
    if not isinstance(train_cfg, dict):
        return train_cfg
    _ensure_train_cfg_sections(train_cfg)
    algorithm = train_cfg.get("algorithm")
    if isinstance(algorithm, dict):
        for key in ("rnd_cfg", "symmetry_cfg"):
            if inactive_rsl_optional_cfg(key, algorithm.get(key)):
                algorithm[key] = None
    _ensure_obs_groups(train_cfg)
    _reassert_double_critic_class_names(train_cfg)
    _reassert_paper_mlp(train_cfg)
    _drop_on_policy_runner_kwarg_collisions(train_cfg)
    _ensure_runner_intervals(train_cfg)
    _reassert_noise_std_type(train_cfg)
    _reassert_activation(train_cfg)
    _reassert_init_noise_std(train_cfg)
    _reassert_logger(train_cfg)
    if "clip_actions" in train_cfg:
        train_cfg["clip_actions"] = sanitize_clip_actions(train_cfg.get("clip_actions"))
    return train_cfg


def runner_cfg_dict(agent_cfg) -> dict:
    """``OnPolicyRunner(..., train_cfg, ...)`` payload after Hydra sanitize."""
    cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    if not isinstance(cfg, dict):
        raise TypeError(f"agent cfg to_dict() must return a dict, got {type(cfg)}")
    return sanitize_rsl_rl_train_cfg(cfg)


def _sensor_cfg_name(sensor) -> str:
    if sensor is None:
        return ""
    name = getattr(sensor, "name", None)
    if name:
        return str(name)
    if isinstance(sensor, dict):
        return str(sensor.get("name") or "")
    return ""


def parent_raycast_height_scan(term) -> bool:
    """True when ``observations.policy.height_scan`` still needs ``scene.height_scanner``."""
    if term is None:
        return False
    func = getattr(term, "func", None)
    if getattr(func, "__name__", "") == "height_scan":
        return True
    params = getattr(term, "params", None)
    sensor = None
    if isinstance(params, dict):
        sensor = params.get("sensor_cfg")
    elif params is not None:
        sensor = getattr(params, "sensor_cfg", None)
    return _sensor_cfg_name(sensor) == "height_scanner"


def anymal_parent_body_names(names) -> bool:
    """True for ANYmal leftovers H1/G1 cannot ``resolve_matching_names``.

    Parent locomotion uses ``body_names="base"``, ``.*THIGH``, ``.*FOOT``.
    Official H1/G1 retarget those. A Hydra ``from_dict`` leftover crashes
    Reward/Event/Termination managers at ``gym.make`` — before ``wandb.init``.
    """
    if names is None:
        return False
    items = list(names) if isinstance(names, (list, tuple)) else [names]
    for item in items:
        text = str(item)
        if text in {"base", ".*FOOT", ".*THIGH", "FOOT", "THIGH"}:
            return True
        if text.endswith("FOOT") or text.endswith("THIGH"):
            return True
    return False


def _term_entity_cfg(term, key: str):
    if term is None:
        return None
    params = getattr(term, "params", None)
    if params is None:
        return None
    if isinstance(params, dict):
        return params.get(key)
    return getattr(params, key, None)


def _entity_body_names(entity):
    if entity is None:
        return None
    if isinstance(entity, dict):
        return entity.get("body_names")
    return getattr(entity, "body_names", None)


def _set_entity_body_names(entity, names) -> None:
    if entity is None:
        return
    if isinstance(entity, dict):
        entity["body_names"] = names
    elif hasattr(entity, "body_names"):
        entity.body_names = names


def _entity_joint_names(entity):
    if entity is None:
        return None
    if isinstance(entity, dict):
        return entity.get("joint_names")
    return getattr(entity, "joint_names", None)


def _set_entity_joint_names(entity, names) -> None:
    if entity is None:
        return
    if isinstance(entity, dict):
        entity["joint_names"] = names
    elif hasattr(entity, "joint_names"):
        entity.joint_names = names


def _robot_identity_blob(env_cfg) -> str:
    parts = [type(env_cfg).__name__]
    scene = getattr(env_cfg, "scene", None)
    robot = getattr(scene, "robot", None)
    objs = [robot, getattr(robot, "spawn", None) if robot is not None else None]
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, str):
            parts.append(obj)
            continue
        mapping = obj if isinstance(obj, dict) else None
        for attr in ("usd_path", "usd_file", "asset_path", "prim_path"):
            value = mapping.get(attr) if mapping is not None else getattr(obj, attr, None)
            if value:
                parts.append(str(value))
    return " ".join(parts).lower()


def _init_pelvis_z(env_cfg):
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    state = getattr(robot, "init_state", None)
    pos = getattr(state, "pos", None)
    if pos is None and isinstance(state, dict):
        pos = state.get("pos")
    if pos is None:
        return None
    try:
        return float(pos[2])
    except (TypeError, ValueError, IndexError):
        return None


def env_cfg_robot_kind(env_cfg) -> str | None:
    """``g1`` / ``h1`` from USD, class name, action joints, or pelvis height."""
    blob = _robot_identity_blob(env_cfg)
    has_g1 = bool(re.search(r"(?:^|[^a-z0-9])g1(?:[^a-z0-9]|$)", blob))
    has_h1 = bool(re.search(r"(?:^|[^a-z0-9])h1(?:[^a-z0-9]|$)", blob))
    if has_g1 and not has_h1:
        return "g1"
    if has_h1 and not has_g1:
        return "h1"

    actions = getattr(env_cfg, "actions", None)
    joint_pos = getattr(actions, "joint_pos", None)
    action_names = getattr(joint_pos, "joint_names", None) if joint_pos is not None else None
    if action_names and not leftover_all_joints(action_names):
        items = set(names_as_list(action_names))
        if items & G1_JOINTS_MISS_H1:
            return "g1"

    z = _init_pelvis_z(env_cfg)
    if z is not None:
        # spawn_robot: pelvis_z + beam_top (0.24 + 0.04).
        if abs(z - 1.02) <= 0.12:
            return "g1"
        if abs(z - 1.33) <= 0.12:
            return "h1"
    return None


def _treat_as_g1(env_cfg) -> bool:
    kind = env_cfg_robot_kind(env_cfg)
    if kind == "g1":
        return True
    if kind == "h1":
        return False
    rewards = getattr(env_cfg, "rewards", None)
    return rewards is not None and getattr(rewards, "joint_deviation_fingers", None) is not None


def _treat_as_h1(env_cfg) -> bool:
    return env_cfg_robot_kind(env_cfg) == "h1"


def _scene_uses_stones(scene) -> bool:
    return scene is not None and getattr(scene, "task_stone_0", None) is not None


def _reassert_physx_floors(env_cfg) -> None:
    """Parent locomotion leftover is ``10 * 2**15`` patches — too small for cloned beams."""
    try:
        from h1_cfg.physx_gpu import PHYSX_A10_UNSAFE_FLOOR, apply_physx_gpu_capacity
    except ImportError as exc:
        print(f"[WARN] PhysX floor reassert skipped ({type(exc).__name__}: {exc})")
        return
    scene = getattr(env_cfg, "scene", None)
    try:
        apply_physx_gpu_capacity(env_cfg, stones=_scene_uses_stones(scene))
    except Exception as exc:
        print(f"[WARN] PhysX floor reassert skipped ({type(exc).__name__}: {exc})")
    physx = getattr(getattr(env_cfg, "sim", None), "physx", None)
    if physx is None:
        return
    # Attribute name, not a quoted key in physx_gpu.py (after_relaunch refuses that).
    current = getattr(physx, "gpu_max_rigid_contact_count", None)
    try:
        value = int(current)
    except (TypeError, ValueError):
        return
    if value > PHYSX_A10_UNSAFE_FLOOR:
        print(
            "[WARN] Clamping leftover PhysX contact stream to the Isaac 8M default. "
            "A 16M leftover OOMs A10 24GB before wandb.init."
        )
        setattr(physx, "gpu_max_rigid_contact_count", PHYSX_A10_UNSAFE_FLOOR)


def expected_pelvis_z(env_cfg, spec) -> float:
    if getattr(getattr(env_cfg, "scene", None), "catcher", None) is not None:
        return float(spec.pelvis_z) + BEAM_TOP_Z
    return float(spec.pelvis_z)


def _set_init_pelvis_z(env_cfg, z: float) -> None:
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot is None:
        return
    state = getattr(robot, "init_state", None)
    if state is None and isinstance(robot, dict):
        state = robot.get("init_state")
    pos = None
    if isinstance(state, dict):
        pos = state.get("pos")
    elif state is not None:
        pos = getattr(state, "pos", None)
    try:
        xy = (float(pos[0]), float(pos[1])) if pos is not None else (0.0, 0.0)
    except (TypeError, ValueError, IndexError):
        xy = (0.0, 0.0)
    new_pos = (xy[0], xy[1], float(z))
    if isinstance(state, dict):
        state["pos"] = new_pos
    elif state is not None and hasattr(state, "pos"):
        state.pos = new_pos


def _stamp_unitree_usd(env_cfg, spec_name: str) -> None:
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    usd = f"/Isaac/Robots/Unitree/{spec_name.upper()}/{spec_name}_minimal.usd"
    robot = getattr(scene, "robot", None)
    if robot is None:
        scene.robot = type("Robot", (), {"usd_path": usd})()
        return
    spawn = getattr(robot, "spawn", None)
    if spawn is not None and hasattr(spawn, "usd_path"):
        spawn.usd_path = usd
    elif isinstance(robot, dict):
        robot["usd_path"] = usd
    elif hasattr(robot, "usd_path"):
        robot.usd_path = usd


def _reassert_unitree_robot(env_cfg) -> None:
    """Leftover ANYmal / full H1 USD crashes or OOMs gym.make before wandb.init."""
    desired = desired_train_robot(env_cfg)
    kind = env_cfg_robot_kind(env_cfg)
    need = leftover_quadruped_robot(env_cfg) or leftover_full_unitree_usd(env_cfg)
    if desired and kind and desired != kind:
        need = True
    spec_name = desired or kind
    if not need:
        if spec_name:
            _reassert_pelvis_height(env_cfg, spec_name)
        return
    spec_name = spec_name or "h1"
    print(f"[WARN] Restoring leftover scene.robot to Unitree {spec_name} minimal USD.")
    try:
        from h1_cfg.beamdojo_common import spawn_robot
        from h1_cfg.robot_spec import G1, H1

        spec = G1 if spec_name == "g1" else H1
        spawn_robot(env_cfg, spec)
        if getattr(getattr(env_cfg, "scene", None), "catcher", None) is None:
            env_cfg.scene.robot.init_state.pos = (0.0, 0.0, spec.pelvis_z)
        return
    except ImportError:
        pass
    from h1_cfg.robot_spec import G1, H1

    spec = G1 if spec_name == "g1" else H1
    _stamp_unitree_usd(env_cfg, spec_name)
    _set_init_pelvis_z(env_cfg, expected_pelvis_z(env_cfg, spec))


def _robot_init_state(env_cfg):
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot is None:
        return None
    state = getattr(robot, "init_state", None)
    if state is None and isinstance(robot, dict):
        state = robot.get("init_state")
    return state


def _joint_map(state, field: str):
    if state is None:
        return None
    value = state.get(field) if isinstance(state, dict) else getattr(state, field, None)
    return value if isinstance(value, dict) else None


def _set_joint_map(state, field: str, value) -> None:
    if state is None:
        return
    if isinstance(state, dict):
        state[field] = value
    elif hasattr(state, field):
        setattr(state, field, value)


def _reassert_init_joint_state(env_cfg) -> None:
    """Hydra leftover ANYmal ``.*HAA`` keys ValueError on H1/G1 at gym.make."""
    state = _robot_init_state(env_cfg)
    if state is None:
        return
    for field in ("joint_pos", "joint_vel"):
        mapping = _joint_map(state, field)
        if not mapping:
            continue
        cleaned = {key: mapping[key] for key in mapping if not leftover_quadruped_joint_key(key)}
        if len(cleaned) == len(mapping):
            continue
        dropped = [key for key in mapping if key not in cleaned]
        print(f"[WARN] Dropping leftover quadruped init_state.{field} keys {dropped}.")
        _set_joint_map(state, field, cleaned)


def _range_pair_ok(value) -> bool:
    try:
        lo, hi = value[0], value[1]
        return float(lo) == float(lo) and float(hi) == float(hi)
    except (TypeError, ValueError, IndexError, KeyError):
        return False


def _reassert_velocity_ranges(env_cfg) -> None:
    """Leftover ``ranges.lin_vel_x=None`` TypeErrors UniformVelocityCommand at gym.make."""
    cmd = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    ranges = getattr(cmd, "ranges", None) if cmd is not None else None
    if ranges is None:
        return
    stage2 = getattr(getattr(env_cfg, "scene", None), "catcher", None) is not None
    defaults = {
        "lin_vel_x": (0.2, 0.8) if stage2 else (-1.0, 1.0),
        "lin_vel_y": (-0.15, 0.15) if stage2 else (-1.0, 1.0),
        "ang_vel_z": (-0.4, 0.4) if stage2 else (-1.0, 1.0),
    }
    for key, default in defaults.items():
        current = ranges.get(key) if isinstance(ranges, dict) else getattr(ranges, key, None)
        if _range_pair_ok(current):
            continue
        print(f"[WARN] Restoring leftover commands.base_velocity.ranges.{key}={current!r}.")
        if isinstance(ranges, dict):
            ranges[key] = default
        else:
            setattr(ranges, key, default)


def _iter_obs_groups(obs):
    if obs is None:
        return
    names: list[str] = []
    data = getattr(obs, "__dict__", None) or {}
    names.extend(key for key in data if not str(key).startswith("_"))
    for known in ("policy", "critic", "teacher", "privileged", "rnd_state"):
        if known not in names and hasattr(obs, known):
            names.append(known)
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        group = getattr(obs, name, None)
        if group is None or callable(group):
            continue
        yield name, group


def _iter_obs_terms(group):
    if group is None:
        return
    if isinstance(group, dict):
        yield from group.items()
        return
    data = getattr(group, "__dict__", None) or {}
    keys = [key for key in data if not str(key).startswith("_")]
    if "height_scan" not in keys and hasattr(group, "height_scan"):
        keys.append("height_scan")
    for key in keys:
        yield key, getattr(group, key, None)


def _reassert_obs_height_scan(env_cfg) -> None:
    """Leftover ``mdp.height_scan`` on any group looks up the nulled RayCaster."""
    obs = getattr(env_cfg, "observations", None)
    policy = getattr(obs, "policy", None)
    if policy is not None and hasattr(policy, "concatenate_terms"):
        policy.concatenate_terms = True
    if policy is not None and hasattr(policy, "flatten_history_dim"):
        policy.flatten_history_dim = True
    for group_name, group in _iter_obs_groups(obs):
        for term_name, term in _iter_obs_terms(group):
            if not parent_raycast_height_scan(term):
                continue
            if group_name == "policy" and term_name == "height_scan":
                print("[WARN] Replacing parent mdp.height_scan with task_height_scan.")
                _install_task_height_scan(group)
                continue
            print(f"[WARN] Clearing leftover observations.{group_name}.{term_name} (height_scanner).")
            if isinstance(group, dict):
                group[term_name] = None
            else:
                setattr(group, term_name, None)


def _reassert_pelvis_height(env_cfg, spec_name: str) -> None:
    """ANYmal leftover z≈0.6 buries H1/G1; PhysX explodes at the first reset."""
    try:
        from h1_cfg.robot_spec import G1, H1
    except ImportError:
        return
    spec = G1 if spec_name == "g1" else H1
    expected = expected_pelvis_z(env_cfg, spec)
    current = _init_pelvis_z(env_cfg)
    if current is None or abs(current - expected) <= 0.10:
        return
    print(f"[WARN] Restoring leftover robot init z {current:.2f} → {expected:.2f} ({spec_name}).")
    _set_init_pelvis_z(env_cfg, expected)


def _reassert_env_spacing(env_cfg) -> None:
    """Parent leftover ``env_spacing=2.5`` overlaps 1024 H1s at gym.make reset."""
    scene = getattr(env_cfg, "scene", None)
    if scene is None or not hasattr(scene, "env_spacing"):
        return
    try:
        spacing = float(scene.env_spacing)
    except (TypeError, ValueError):
        return
    if spacing < 6.0:
        print("[WARN] Restoring leftover env_spacing to 8.0 (parent 2.5 overlaps 1024 envs).")
        scene.env_spacing = 8.0


def _reassert_contact_history(env_cfg) -> None:
    """``feet_air_time`` reads history at the first reset — before wandb.init."""
    contact = getattr(getattr(env_cfg, "scene", None), "contact_forces", None)
    if contact is None:
        return
    hist = getattr(contact, "history_length", None)
    try:
        length = int(hist)
    except (TypeError, ValueError):
        length = 0
    if length < 3:
        print("[WARN] Restoring leftover contact_forces.history_length=3.")
        contact.history_length = 3
    if hasattr(contact, "track_air_time") and not contact.track_air_time:
        print("[WARN] Enabling leftover contact_forces.track_air_time.")
        contact.track_air_time = True


def _reassert_sim_timing(env_cfg) -> None:
    """Leftover ``dt=0`` / ``decimation=0`` / CPU sim dies at gym.make before W&B."""
    dec_n = 0
    if hasattr(env_cfg, "decimation"):
        try:
            dec_n = int(env_cfg.decimation)
        except (TypeError, ValueError):
            dec_n = 0
        if dec_n < 1:
            print("[WARN] Restoring leftover decimation=4.")
            env_cfg.decimation = 4
            dec_n = 4
    if hasattr(env_cfg, "episode_length_s"):
        try:
            ep_n = float(env_cfg.episode_length_s)
        except (TypeError, ValueError):
            ep_n = 0.0
        if ep_n <= 0:
            print("[WARN] Restoring leftover episode_length_s=20.")
            env_cfg.episode_length_s = 20.0
    sim = getattr(env_cfg, "sim", None)
    if sim is None:
        return
    if leftover_unusable_device(getattr(sim, "device", None)):
        print("[WARN] Forcing leftover sim.device onto cuda:0 (single-GPU A10).")
        sim.device = "cuda:0"
    if hasattr(sim, "dt"):
        try:
            dt = float(sim.dt)
        except (TypeError, ValueError):
            dt = 0.0
        if dt <= 0:
            print("[WARN] Restoring leftover sim.dt=0.005.")
            sim.dt = 0.005
    if hasattr(sim, "render_interval"):
        try:
            ri = int(sim.render_interval)
        except (TypeError, ValueError):
            ri = 0
        if ri < 1 and dec_n >= 1:
            print(f"[WARN] Restoring leftover sim.render_interval={dec_n}.")
            sim.render_interval = dec_n


def _reassert_stage1_timeout_only(env_cfg) -> None:
    """Stage 1 is timeout-only. Leftover parent base_contact fires on the plane."""
    if getattr(getattr(env_cfg, "scene", None), "catcher", None) is not None:
        return
    terms = getattr(env_cfg, "terminations", None)
    if terms is None:
        return
    for name in ("base_contact", "base_height", "base_orientation"):
        if getattr(terms, name, None) is not None:
            print(f"[WARN] Clearing leftover terminations.{name} (Stage 1 is timeout-only).")
            setattr(terms, name, None)


def _reassert_g1_action_joints(env_cfg) -> None:
    """Parent leftover ``joint_names=[".*"]`` puts G1 arms/fingers back in the action."""
    if not _treat_as_g1(env_cfg):
        return
    actions = getattr(env_cfg, "actions", None)
    joint_pos = getattr(actions, "joint_pos", None)
    if joint_pos is None or not leftover_all_joints(getattr(joint_pos, "joint_names", None)):
        return
    try:
        from h1_cfg.robot_spec import G1
    except ImportError as exc:
        print(f"[WARN] G1 action-joint reassert skipped ({type(exc).__name__}: {exc})")
        return
    print("[WARN] Restoring leftover G1 action joints off parent '.*' (paper: 12 lower-body).")
    joint_pos.joint_names = list(G1.action_joints)


def _restore_entity_names(entity, kind: str, expected, label: str) -> None:
    if entity is None or expected is None:
        return
    current = _entity_joint_names(entity) if kind == "joint" else _entity_body_names(entity)
    if current is None or names_match(current, expected):
        return
    print(f"[WARN] Restoring leftover {label} so Isaac 2.3.2 re.fullmatch can resolve joints/bodies.")
    if kind == "joint":
        _set_entity_joint_names(entity, expected)
    else:
        _set_entity_body_names(entity, expected)


def _restore_robot_name_filters(env_cfg, spec, *, finger_joints=None) -> None:
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is not None:
        hip = [spec.hip_yaw, spec.hip_roll]
        mapping = (
            ("dof_pos_limits", "asset_cfg", "joint", spec.ankle_joints),
            ("joint_deviation_hip", "asset_cfg", "joint", hip),
            ("joint_deviation_arms", "asset_cfg", "joint", spec.arm_joints),
            ("joint_deviation_torso", "asset_cfg", "joint", spec.torso_joint),
            ("feet_air_time", "sensor_cfg", "body", spec.feet_body),
            ("feet_slide", "sensor_cfg", "body", spec.feet_body),
            ("feet_slide", "asset_cfg", "body", spec.feet_body),
            ("foothold_penalty", "sensor_cfg", "body", spec.feet_body),
        )
        if finger_joints is not None:
            mapping += (("joint_deviation_fingers", "asset_cfg", "joint", finger_joints),)
        if spec.name == "g1":
            mapping += (
                ("dof_acc_l2", "asset_cfg", "joint", [".*_hip_.*", ".*_knee_joint"]),
                ("dof_torques_l2", "asset_cfg", "joint", [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            )
        for term_name, key, kind, expected in mapping:
            term = getattr(rewards, term_name, None)
            _restore_entity_names(
                _term_entity_cfg(term, key),
                kind,
                expected,
                f"rewards.{term_name}.{key}",
            )

    terms = getattr(env_cfg, "terminations", None)
    if terms is not None:
        off_terrain = getattr(terms, "off_terrain", None)
        _restore_entity_names(
            _term_entity_cfg(off_terrain, "sensor_cfg"),
            "body",
            spec.feet_body,
            "terminations.off_terrain.sensor_cfg",
        )


def _reassert_g1_joint_fullmatch(env_cfg) -> None:
    """H1 leftover ``torso`` / ``.*_hip_yaw`` does not fullmatch G1 ``*_joint`` names.

    RewardManager raises at ``gym.make`` — before ``wandb.init``.
    """
    if not _treat_as_g1(env_cfg):
        return
    try:
        from h1_cfg.robot_spec import G1, G1_FINGER_JOINTS
    except ImportError as exc:
        print(f"[WARN] G1 joint-name reassert skipped ({type(exc).__name__}: {exc})")
        return
    _restore_robot_name_filters(env_cfg, G1, finger_joints=list(G1_FINGER_JOINTS))


def _drop_h1_leftover_fingers(env_cfg) -> None:
    """H1 has no finger joints. A leftover G1 term crashes ``resolve_matching_names``."""
    if not _treat_as_h1(env_cfg):
        return
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None or getattr(rewards, "joint_deviation_fingers", None) is None:
        return
    print("[WARN] Clearing leftover rewards.joint_deviation_fingers (H1 has no finger joints).")
    rewards.joint_deviation_fingers = None


def _reassert_h1_joint_fullmatch(env_cfg) -> None:
    """G1 leftover ``*_joint`` regexes miss official H1 names (no ``_joint`` suffix)."""
    if not _treat_as_h1(env_cfg):
        return
    try:
        from h1_cfg.robot_spec import H1
    except ImportError as exc:
        print(f"[WARN] H1 joint-name reassert skipped ({type(exc).__name__}: {exc})")
        return
    _restore_robot_name_filters(env_cfg, H1)


def _pose_range_from_reset(reset_base):
    params = getattr(reset_base, "params", None)
    if params is None:
        return None, None
    if isinstance(params, dict):
        pose = params.get("pose_range")
        current = dict(pose) if isinstance(pose, dict) else {}
        return params, current
    pose = getattr(params, "pose_range", None)
    current = dict(pose) if isinstance(pose, dict) else {}
    return params, current


def _set_pose_range(params, pose_range) -> None:
    if isinstance(params, dict):
        params["pose_range"] = pose_range
    elif hasattr(params, "pose_range"):
        params.pose_range = pose_range


def _range_too_wide(span, limit: float) -> bool:
    if span is None:
        return False
    try:
        lo, hi = float(span[0]), float(span[1])
    except (TypeError, ValueError, IndexError):
        return False
    return abs(lo) > limit or abs(hi) > limit


def _params_get(params, key):
    if params is None:
        return None
    if isinstance(params, dict):
        return params.get(key)
    return getattr(params, key, None)


def _params_set(params, key, value) -> None:
    if params is None:
        return
    if isinstance(params, dict):
        params[key] = value
    elif hasattr(params, key):
        setattr(params, key, value)


def _reassert_official_reset_events(env_cfg) -> None:
    """Official H1/G1 2.3.2: no interval push, identity joints, zero root twist.

    Parent ANYmal leftover ``push_robot`` / ``velocity_range=±0.5`` /
    ``position_range=(0.5, 1.5)`` runs at ``RslRlVecEnvWrapper`` reset — before
    ``wandb.init``. Wild joint scale can NaN PhysX; leftover push is eval-only
    in the paper.
    """
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    if getattr(events, "push_robot", None) is not None:
        print("[WARN] Clearing leftover events.push_robot (interval push is eval-only).")
        events.push_robot = None

    reset_base = getattr(events, "reset_base", None)
    params = getattr(reset_base, "params", None) if reset_base is not None else None
    vel = _params_get(params, "velocity_range")
    vel = dict(vel) if isinstance(vel, dict) else {}
    if any(_range_too_wide(vel.get(key), 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")):
        print("[WARN] Zeroing leftover reset_base velocity (official H1/G1).")
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            vel[key] = (0.0, 0.0)
        _params_set(params, "velocity_range", vel)

    reset_joints = getattr(events, "reset_robot_joints", None)
    jparams = getattr(reset_joints, "params", None) if reset_joints is not None else None
    span = _params_get(jparams, "position_range")
    try:
        lo, hi = float(span[0]), float(span[1])
    except (TypeError, ValueError, IndexError):
        lo, hi = 1.0, 1.0
    if lo < 0.99 or hi > 1.01:
        print("[WARN] Restoring leftover reset_robot_joints position_range to identity (official H1/G1).")
        _params_set(jparams, "position_range", (1.0, 1.0))


def _reassert_infinite_horizon(env_cfg) -> None:
    """``RslRlVecEnvWrapper`` only copies ``time_outs`` when horizon is infinite.

    Stage 1 is timeout-only. A leftover ``is_finite_horizon=True`` drops
    ``extras['time_outs']``, so PPO 3.0.1 / foothold GAE never bootstrap and
    the 10k is not the paper algorithm (and can NaN value targets).
    """
    if getattr(env_cfg, "is_finite_horizon", None) is True:
        print("[WARN] Forcing is_finite_horizon=False so RslRlVecEnvWrapper sets time_outs.")
        env_cfg.is_finite_horizon = False


def _reassert_stage2_reset_on_beam(env_cfg) -> None:
    """Parent leftover ``y=(-0.5, 0.5)`` / ``yaw=±π`` spawns beside a 20 cm beam."""
    scene = getattr(env_cfg, "scene", None)
    if scene is None or getattr(scene, "catcher", None) is None:
        return
    events = getattr(env_cfg, "events", None)
    reset_base = getattr(events, "reset_base", None)
    if reset_base is None:
        return
    params, pose = _pose_range_from_reset(reset_base)
    if params is None:
        return
    if not (
        _range_too_wide(pose.get("y"), 0.1)
        or _range_too_wide(pose.get("yaw"), 0.5)
        or _range_too_wide(pose.get("x"), 0.6)
    ):
        return
    print("[WARN] Tightening leftover reset_base pose so Stage 2 does not spawn off the beam.")
    pose["x"] = (-0.2, 0.5)
    pose["y"] = (-0.08, 0.08)
    pose["yaw"] = (-0.3, 0.3)
    _set_pose_range(params, pose)


def _reassert_anymal_body_names(env_cfg) -> None:
    """Null or retarget leftover ANYmal base/THIGH/FOOT body filters."""
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is not None:
        if getattr(rewards, "undesired_contacts", None) is not None:
            print("[WARN] Clearing leftover rewards.undesired_contacts (ANYmal .*THIGH).")
            rewards.undesired_contacts = None
        for name in ("feet_air_time", "feet_slide"):
            term = getattr(rewards, name, None)
            for key in ("sensor_cfg", "asset_cfg"):
                entity = _term_entity_cfg(term, key)
                if anymal_parent_body_names(_entity_body_names(entity)):
                    print(f"[WARN] Retargeting leftover rewards.{name}.{key} off ANYmal FOOT/base.")
                    _set_entity_body_names(entity, ".*ankle.*")

    events = getattr(env_cfg, "events", None)
    if events is not None:
        ext = getattr(events, "base_external_force_torque", None)
        if ext is not None and anymal_parent_body_names(
            _entity_body_names(_term_entity_cfg(ext, "asset_cfg"))
        ):
            print("[WARN] Clearing leftover events.base_external_force_torque (ANYmal body 'base').")
            events.base_external_force_torque = None
        for name in ("add_base_mass", "base_com"):
            term = getattr(events, name, None)
            entity = _term_entity_cfg(term, "asset_cfg")
            if anymal_parent_body_names(_entity_body_names(entity)):
                print(f"[WARN] Retargeting leftover events.{name} from 'base' to torso_link.")
                _set_entity_body_names(entity, "torso_link")

    terms = getattr(env_cfg, "terminations", None)
    if terms is not None:
        contact = getattr(terms, "base_contact", None)
        entity = _term_entity_cfg(contact, "sensor_cfg")
        if anymal_parent_body_names(_entity_body_names(entity)):
            scene = getattr(env_cfg, "scene", None)
            if scene is not None and getattr(scene, "catcher", None) is not None:
                print("[WARN] Retargeting leftover terminations.base_contact from 'base' to torso_link.")
                _set_entity_body_names(entity, "torso_link")
            else:
                print("[WARN] Clearing leftover terminations.base_contact (ANYmal body 'base').")
                terms.base_contact = None


def _install_task_height_scan(policy) -> None:
    """Swap parent ``mdp.height_scan`` for the dual-terrain task map. Isaac-only."""
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg

    from h1_cfg.mdp import task_height_scan

    policy.height_scan = ObsTerm(
        func=task_height_scan,
        params={"sensor_cfg": SceneEntityCfg("robot"), "offset": 0.5, "grid_n": 15, "extent": 1.4},
        clip=(-1.0, 1.0),
    )


def reassert_gpu_env_cfg(env_cfg) -> None:
    """Undo parent / Hydra leftovers that crash ``gym.make`` on the A10.

    ``LocomotionVelocityRoughEnvCfg.__post_init__`` runs *before* ``apply_stage1``
    and copies the rough-generator material onto ``sim.physics_material``. Isaac
    Lab 2.3.2 ``hydra_task_config`` then ``from_dict``s CLI/compose onto that
    instance. A leftover ANYmal RayCaster plus ``mdp.height_scan`` looks up
    ``scene['height_scanner']`` on the first observation.
    """
    scene = getattr(env_cfg, "scene", None)
    if scene is not None and getattr(scene, "height_scanner", None) is not None:
        print(
            "[WARN] Clearing leftover scene.height_scanner (ANYmal RayCaster). "
            "Policy scan is task_height_scan."
        )
        scene.height_scanner = None

    terrain = getattr(scene, "terrain", None) if scene is not None else None
    if terrain is not None:
        if getattr(terrain, "terrain_type", None) != "plane":
            print("[WARN] Forcing terrain_type=plane (parent rough generator leftover).")
            terrain.terrain_type = "plane"
        if getattr(terrain, "terrain_generator", None) is not None:
            terrain.terrain_generator = None
        if hasattr(terrain, "debug_vis"):
            terrain.debug_vis = False
        # Parent rough importer carries a Nucleus marble MDL; plane spawn does not need it.
        if getattr(terrain, "visual_material", None) is not None:
            terrain.visual_material = None

    sky = getattr(scene, "sky_light", None) if scene is not None else None
    spawn = getattr(sky, "spawn", None)
    if spawn is not None and getattr(spawn, "texture_file", None):
        print("[WARN] Clearing Nucleus HDR sky so headless gym.make does not block before W&B.")
        spawn.texture_file = None

    _reassert_obs_height_scan(env_cfg)

    commands = getattr(env_cfg, "commands", None)
    base_velocity = getattr(commands, "base_velocity", None)
    if base_velocity is not None:
        if hasattr(base_velocity, "debug_vis"):
            base_velocity.debug_vis = False
        if hasattr(base_velocity, "heading_command"):
            base_velocity.heading_command = False

    sim = getattr(env_cfg, "sim", None)
    mat = getattr(terrain, "physics_material", None) if terrain is not None else None
    if sim is not None and mat is not None:
        sim.physics_material = mat

    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is not None and getattr(curriculum, "terrain_levels", None) is not None:
        # Parent leftover. terrain_levels_vel does terrain_generator.size on reset
        # (RslRlVecEnvWrapper.__init__ → env.reset) — before wandb.init. Plane has none.
        print("[WARN] Clearing leftover curriculum.terrain_levels (plane has no generator).")
        curriculum.terrain_levels = None

    _reassert_unitree_robot(env_cfg)
    _reassert_init_joint_state(env_cfg)
    _reassert_velocity_ranges(env_cfg)
    _reassert_anymal_body_names(env_cfg)
    _reassert_g1_action_joints(env_cfg)
    _reassert_g1_joint_fullmatch(env_cfg)
    _drop_h1_leftover_fingers(env_cfg)
    _reassert_h1_joint_fullmatch(env_cfg)
    _reassert_official_reset_events(env_cfg)
    _reassert_stage2_reset_on_beam(env_cfg)
    _reassert_stage1_timeout_only(env_cfg)
    _reassert_env_spacing(env_cfg)
    _reassert_contact_history(env_cfg)
    _reassert_sim_timing(env_cfg)
    _reassert_infinite_horizon(env_cfg)
    _reassert_physx_floors(env_cfg)


def _none_safe_update_class_from_dict(orig):
    """Leave ``None`` attributes None instead of recursing into a Hydra mapping."""

    def update_class_from_dict(obj, data, _ns=""):
        if obj is None:
            return None
        return orig(obj, data, _ns=_ns)

    return update_class_from_dict


def _patch_hydra_none_from_dict() -> None:
    """Isaac Lab 2.3.2 ``from_dict`` KeyErrors when the instance field is None.

    ``update_class_from_dict(None, {prim_path: ...})`` walks attributes of
    ``None`` and raises. That happens if Hydra compose/CLI supplies a nested
    dict for a field ``apply_stage*`` already nulled (``height_scanner``,
    ``undesired_contacts``, ``rnd_cfg``). Aborting here skips ``gym.make`` and
    W&B. Keeping None is the BeamDojo value; ``reassert_gpu_env_cfg`` still
    clears leftovers that were not None.
    """
    try:
        import isaaclab.utils.configclass as il_cc
        import isaaclab.utils.dict as il_dict
    except ImportError:
        return
    if getattr(il_dict, "_beamdojo_none_safe", False):
        return
    orig = il_dict.update_class_from_dict
    wrapped = _none_safe_update_class_from_dict(orig)
    il_dict.update_class_from_dict = wrapped
    il_dict._beamdojo_none_safe = True
    if getattr(il_cc, "update_class_from_dict", None) is orig:
        il_cc.update_class_from_dict = wrapped


def remaining_learning_iterations(current: int | None, max_iterations: int) -> int:
    """PPO iters so the run *ends* at ``max_iterations`` (paper: 10k / stage).

    rsl-rl 3.0.1 ``learn(n)`` is ``range(current, current + n)``. Passing the
    configured max after a resume at iter 500 would train to 10500 and the
    W&B / Research Lab step axis would not be a 10k Stage 1/2 run.
    """
    start = int(current or 0)
    target = int(max_iterations)
    return max(0, target - start)


def resolve_resume_checkpoint(
    log_root: str | os.PathLike,
    load_run: str | None = None,
    load_checkpoint: str | None = None,
) -> str:
    """Pick a ``model_*.pt`` under ``log_root``.

    Isaac Lab 2.3.2 ``get_checkpoint_path(..., sort_alpha=True)`` sorts run
    folders alphabetically (fine for ``YYYY-MM-DD_HH-MM-SS``) and checkpoints
    with a padded-filename hack. We sort runs by mtime and checkpoints by the
    trailing iteration number so a 5-iter smoke (``model_4.pt``) wins over a
    missing ``model_9999.pt``, and ``model_10.pt`` wins over ``model_9.pt``.

    ``load_run`` / ``load_checkpoint`` are Isaac Lab regexes (``re.match``).
    Defaults match ``RslRlBaseRunnerCfg`` (``.*`` / ``model_.*.pt``).
    """
    root = Path(log_root)
    if not root.is_dir():
        raise ValueError(f"No runs present in the directory: '{root}' match: '{load_run or DEFAULT_LOAD_RUN}'.")
    run_pat = (load_run or "").strip() or DEFAULT_LOAD_RUN
    ckpt_pat = (load_checkpoint or "").strip() or DEFAULT_LOAD_CHECKPOINT
    runs = [path for path in root.iterdir() if path.is_dir() and re.match(run_pat, path.name)]
    if not runs:
        raise ValueError(f"No runs present in the directory: '{root}' match: '{run_pat}'.")
    run_dir = max(runs, key=lambda path: path.stat().st_mtime)
    models = [path for path in run_dir.iterdir() if path.is_file() and re.match(ckpt_pat, path.name)]
    if not models:
        raise ValueError(f"No checkpoints in the directory: '{run_dir}' match '{ckpt_pat}'.")
    chosen = max(models, key=_checkpoint_sort_key)
    return str(chosen.resolve())


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    digits = re.findall(r"\d+", path.name)
    iteration = int(digits[-1]) if digits else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (iteration, mtime)


def pick_play_checkpoint(
    stage: int,
    robot: str,
    *,
    load_run: str | None = None,
    load_checkpoint: str | None = None,
    load_experiment: str | None = None,
) -> str:
    """Play: prefer a Stage 2 run when one exists, else the Stage 1 fine-tune source."""
    load_root = resolve_load_log_root(stage, robot, load_experiment=load_experiment)
    override = (load_experiment or os.environ.get("LOAD_EXPERIMENT") or "").strip()
    if int(stage) >= 2 and not override:
        stage2_root = resolve_log_root(experiment_name(stage, robot))
        try:
            return resolve_resume_checkpoint(stage2_root, load_run, load_checkpoint)
        except ValueError:
            pass
    return resolve_resume_checkpoint(load_root, load_run, load_checkpoint)


def ensure_beamdojo_registered() -> None:
    """Import cfg modules so gym.register side effects run. Requires SimulationApp."""
    global _REGISTERED
    if _REGISTERED:
        return
    import g1_cfg.beamdojo_stage1_cfg  # noqa: F401
    import g1_cfg.beamdojo_stage2_cfg  # noqa: F401
    import h1_cfg.beamdojo_stage1_cfg  # noqa: F401
    import h1_cfg.beamdojo_stage2_cfg  # noqa: F401

    _REGISTERED = True


def ensure_beamdojo_stage1_registered() -> None:
    ensure_beamdojo_registered()


def inject_double_critic() -> None:
    """Put ActorCriticDouble / PPODoubleCritic in rsl-rl's eval() namespace."""
    import rsl_rl.runners.on_policy_runner as opr
    from beamdojo_agents.double_critic import ActorCriticDouble, PPODoubleCritic

    opr.ActorCriticDouble = ActorCriticDouble
    opr.PPODoubleCritic = PPODoubleCritic
    _patch_runner_foot_optimizer(opr.OnPolicyRunner)
    _patch_runner_log_ep_infos(opr.OnPolicyRunner)
    _patch_store_code_state(opr)
    _patch_hydra_none_from_dict()
    sync_wandb_identity_env()
    _patch_wandb_init_retry()
    _patch_wandb_writer()


def _torch_load(path, map_location=None):
    """Load a checkpoint on both old torch (no weights_only) and 2.6+."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _patch_runner_foot_optimizer(runner_cls) -> None:
    """Persist the foothold Adam state next to rsl-rl's loco optimizer.

    rsl-rl 3.0.1 ``save()`` writes the ``.pt`` then calls ``writer.save_model``.
    A W&B 503 after the file is on disk must not abort ``learn()`` at iter 0
    (save_interval hits the first iteration). Same for the extra torch.load
    used to splice in ``foot_optimizer_state_dict``.
    """
    if getattr(runner_cls, "_beamdojo_foot_ckpt", False):
        return
    orig_save = runner_cls.save
    orig_load = runner_cls.load

    def save(self, path, infos=None):
        try:
            orig_save(self, path, infos)
        except Exception as exc:
            if not os.path.isfile(path):
                raise
            print(
                f"[WARN] runner.save logger upload failed ({type(exc).__name__}: {exc}). "
                "Checkpoint is on disk; continuing learn()."
            )
        foot = getattr(getattr(self, "alg", None), "foot_optimizer", None)
        if foot is None:
            return
        try:
            blob = _torch_load(path, map_location="cpu")
            blob["foot_optimizer_state_dict"] = foot.state_dict()
            import torch

            torch.save(blob, path)
        except Exception as exc:
            print(f"[WARN] foothold optimizer not stored ({type(exc).__name__}: {exc})")

    def load(self, path, load_optimizer=True, map_location=None):
        infos = orig_load(self, path, load_optimizer=load_optimizer, map_location=map_location)
        foot = getattr(getattr(self, "alg", None), "foot_optimizer", None)
        if not load_optimizer or foot is None:
            return infos
        try:
            blob = _torch_load(path, map_location=map_location)
            state = blob.get("foot_optimizer_state_dict") if isinstance(blob, dict) else None
            if not state:
                return infos
            foot.load_state_dict(state)
        except Exception as exc:
            print(f"[WARN] Not loading foothold optimizer: {exc}")
        return infos

    runner_cls.save = save
    runner_cls.load = load
    runner_cls._beamdojo_foot_ckpt = True


def _patch_store_code_state(opr) -> None:
    """rsl-rl 3.0.1 dumps git status after the first PPO iter.

    ``store_code_state`` only guards ``git.Repo()``. ``repo.git.status()`` still
    raises on Lambda/Docker 'dubious ownership' and would abort ``learn()``
    after Isaac boot and the first update.
    """
    orig = getattr(opr, "store_code_state", None)
    if not callable(orig) or getattr(opr, "_beamdojo_store_code", False):
        return

    def store_code_state(logdir, repositories):
        try:
            return orig(logdir, repositories)
        except Exception as exc:
            print(f"[WARN] git diff upload skipped ({type(exc).__name__}: {exc})")
            return []

    opr.store_code_state = store_code_state
    opr._beamdojo_store_code = True
    try:
        import rsl_rl.utils as utils

        utils.store_code_state = store_code_state
    except Exception:
        pass


def resolve_log_root(experiment_name: str) -> str:
    """Prefer Lambda NFS so checkpoints survive instance terminate."""
    env_root = os.environ.get("BEAMDOJO_LOG_ROOT", "").strip()
    nfs = Path("/lambda/nfs/beamdojo/logs")
    if env_root:
        base = Path(env_root)
    elif nfs.is_dir():
        base = nfs
    else:
        base = REPO_ROOT / "logs"
    path = (base / "rsl_rl" / experiment_name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_load_log_root(
    stage: int,
    robot: str,
    *,
    load_experiment: str | None = None,
) -> str:
    """Log root used by ``get_checkpoint_path`` when resuming."""
    return resolve_log_root(resolve_load_experiment(stage, robot, load_experiment=load_experiment))


def sync_wandb_identity_env() -> None:
    """Align env vars with rsl-rl 3.0.1 ``WandbSummaryWriter``.

    That writer does ``os.environ["WANDB_USERNAME"]`` (KeyError → entity=None).
    Docs and ``.env.lambda`` set ``WANDB_ENTITY``. If username is missing, copy
    the entity so ``wandb.init`` lands in the same place Kingdom GraphQL queries.
    A blank ``WANDB_USERNAME=`` is set, so it is *not* a KeyError — ``wandb.init(entity="")``
    then aborts ``learn()`` after Isaac boot. Unset blanks.

    ``WANDB_PROJECT`` is optional for the 3.0.1 writer (it reads ``cfg["wandb_project"]``),
    but other wandb helpers KeyError if the env var is missing. Default it.
    """
    entity = (os.environ.get("WANDB_ENTITY") or "").strip()
    username = (os.environ.get("WANDB_USERNAME") or "").strip()
    chosen = entity or username
    if chosen:
        os.environ["WANDB_USERNAME"] = chosen
        if not entity:
            os.environ["WANDB_ENTITY"] = chosen
    elif "WANDB_USERNAME" in os.environ:
        del os.environ["WANDB_USERNAME"]
    project = (os.environ.get("WANDB_PROJECT") or "").strip()
    if project:
        os.environ["WANDB_PROJECT"] = project
    else:
        os.environ["WANDB_PROJECT"] = "beamdojo"


def retry_call(fn, *args, attempts: int = 3, label: str = "call", sleeper=None, **kwargs):
    """Retry transient Lambda / W&B HTTP. ``sleeper`` is injectable for tests."""
    import time

    pause = sleeper if sleeper is not None else time.sleep
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            print(
                f"[WARN] {label} attempt {attempt}/{attempts} failed "
                f"({type(exc).__name__}: {exc})"
            )
            if attempt < attempts:
                pause(min(16.0, float(2**attempt)))
    assert last is not None
    raise last


def _patch_wandb_init_retry() -> None:
    """rsl-rl 3.0.1 ``WandbSummaryWriter`` calls ``wandb.init`` once.

    A single 503/timeout on Lambda falls through to TensorBoard and the 10k
    CUDA run never gets a ``wandb.run.url`` for Research Lab / W&B.
    """
    try:
        import wandb
    except Exception:
        return
    orig = getattr(wandb, "init", None)
    if not callable(orig) or getattr(orig, "_beamdojo_retry", False):
        return

    def init(*args, **kwargs):
        sync_wandb_identity_env()
        if kwargs.get("entity") == "":
            kwargs["entity"] = None

        def attempt():
            try:
                import wandb as wb

                existing = getattr(wb, "run", None)
                if existing is not None:
                    return existing
            except Exception:
                pass
            return orig(*args, **kwargs)

        return retry_call(attempt, attempts=3, label="wandb.init")

    init._beamdojo_retry = True
    wandb.init = init


def _keep_wandb_call(label: str, fn, *args, **kwargs):
    """W&B I/O must not abort PPO after Isaac boot (transient HTTP / JSON)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] wandb {label} skipped ({type(exc).__name__}): {exc}")
        return None


def _as_log_scalar(value):
    """wandb.log rejects GPU tensors from Episode/* means; TensorBoard accepts them."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().float().cpu().item()
    except Exception:
        pass
    return value


def _scalar_ep_info_value(value):
    """Python float for extras['log']. rsl-rl 3.0.1 ``log()`` cats 0-dim / [1] numerics.

    A per-env ``[N]`` tensor in the same dict as Isaac's ``Episode_Reward/*``
    0-dim scalars makes ``torch.cat`` throw. That abort happens *before*
    Loss/* and Train/mean_reward, so the live W&B page stays empty for the iter.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    if isinstance(value, (str, bytes, dict)):
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            count = int(numel())
        except Exception:
            count = -1
        if count <= 0:
            return None
        try:
            tensor = value.float() if callable(getattr(value, "float", None)) else value
            reduced = tensor.mean() if count > 1 else tensor
            item = getattr(reduced, "item", None)
            if callable(item):
                return _scalar_ep_info_value(item())
            return _scalar_ep_info_value(reduced)
        except Exception:
            return None
    try:
        seq = list(value)
    except TypeError:
        try:
            return _scalar_ep_info_value(float(value))
        except (TypeError, ValueError):
            return None
    acc: list[float] = []
    for item in seq:
        number = _scalar_ep_info_value(item)
        if number is not None:
            acc.append(number)
    if not acc:
        return None
    return sum(acc) / len(acc)


def sanitize_ep_infos_for_rsl_log(ep_infos) -> None:
    """Collapse extras['log'] values to scalars in-place before OnPolicyRunner.log."""
    if not isinstance(ep_infos, (list, tuple)):
        return
    for info in ep_infos:
        if not isinstance(info, dict):
            continue
        drop = []
        for key, value in info.items():
            scalar = _scalar_ep_info_value(value)
            if scalar is None:
                drop.append(key)
            else:
                info[key] = scalar
        for key in drop:
            del info[key]


def _patch_runner_log_ep_infos(runner_cls) -> None:
    """Sanitize episode extras before rsl-rl 3.0.1 ``log()`` walks ``ep_infos``."""
    if getattr(runner_cls, "_beamdojo_log_ep", False):
        return
    orig = runner_cls.log

    def log(self, locs, *args, **kwargs):
        if isinstance(locs, dict):
            sanitize_ep_infos_for_rsl_log(locs.get("ep_infos"))
        return orig(self, locs, *args, **kwargs)

    runner_cls.log = log
    runner_cls._beamdojo_log_ep = True


def _patch_wandb_config_update() -> None:
    """rsl-rl 3.0.1 ``WandbSummaryWriter.__init__`` calls ``wandb.config.update``
    *after* ``wandb.init``. A JSON-serializable failure there aborts the writer
    constructor, ``_prepare_logging_writer`` falls back to TensorBoard, and the
    live W&B page never gets PPO scalars.
    """
    try:
        from wandb.sdk.wandb_config import Config
    except Exception:
        return
    orig = getattr(Config, "update", None)
    if not callable(orig) or getattr(orig, "_beamdojo_keep_alive", False):
        return

    def update(self, *args, **kwargs):
        kwargs.setdefault("allow_val_change", True)
        return _keep_wandb_call("config.update", orig, self, *args, **kwargs)

    update._beamdojo_keep_alive = True
    Config.update = update


def _patch_wandb_writer() -> None:
    """Keep a 10k CUDA run alive if W&B JSON/HTTP fails after wandb.init."""
    try:
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter
    except Exception:
        return
    _patch_wandb_init_retry()
    _patch_wandb_config_update()
    if getattr(WandbSummaryWriter, "_beamdojo_keep_alive", False):
        return
    orig_init = WandbSummaryWriter.__init__
    orig_store = WandbSummaryWriter.store_config
    orig_add = WandbSummaryWriter.add_scalar
    orig_log_config = getattr(WandbSummaryWriter, "log_config", None)
    orig_save_file = getattr(WandbSummaryWriter, "save_file", None)
    orig_save_model = getattr(WandbSummaryWriter, "save_model", None)

    def __init__(self, *args, **kwargs):
        sync_wandb_identity_env()
        try:
            orig_init(self, *args, **kwargs)
            return
        except Exception as exc:
            print(
                f"[WARN] WandbSummaryWriter setup recovered ({type(exc).__name__}: {exc}). "
                "Keeping the W&B run if wandb.init already succeeded."
            )
        if not hasattr(self, "name_map"):
            self.name_map = {
                "Train/mean_reward/time": "Train/mean_reward_time",
                "Train/mean_episode_length/time": "Train/mean_episode_length_time",
            }
        try:
            import wandb

            if wandb.run is None:
                raise RuntimeError("wandb.init did not create a run")
        except Exception:
            raise

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        _keep_wandb_call("store_config", orig_store, self, env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def add_scalar(self, tag, scalar_value, *args, **kwargs):
        return _keep_wandb_call(
            "add_scalar",
            orig_add,
            self,
            tag,
            _as_log_scalar(scalar_value),
            *args,
            **kwargs,
        )

    WandbSummaryWriter.__init__ = __init__
    WandbSummaryWriter.store_config = store_config
    WandbSummaryWriter.add_scalar = add_scalar
    if callable(orig_log_config):
        WandbSummaryWriter.log_config = lambda self, *args, **kwargs: _keep_wandb_call(
            "log_config", orig_log_config, self, *args, **kwargs
        )
    if callable(orig_save_file):
        WandbSummaryWriter.save_file = lambda self, path, iter=None: _keep_wandb_call(
            "save_file", orig_save_file, self, path, iter
        )
    if callable(orig_save_model):
        WandbSummaryWriter.save_model = lambda self, model_path, iter: _keep_wandb_call(
            "save_model", orig_save_model, self, model_path, iter
        )
    WandbSummaryWriter._beamdojo_keep_alive = True
    WandbSummaryWriter._beamdojo_store_cfg = True


def _fallback_tensorboard_writer(runner):
    """If wandb.init fails, still log PPO scalars so Research Lab heartbeats run."""
    if getattr(runner, "writer", None) is not None:
        return runner.writer
    log_dir = getattr(runner, "log_dir", None)
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"[WARN] TensorBoard fallback unavailable: {exc}")
        return None
    runner.logger_type = "tensorboard"
    runner.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    return runner.writer


def wandb_project_url(project: str = "beamdojo") -> str:
    entity = (os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME") or "").strip()
    if entity:
        return f"https://wandb.ai/{entity}/{project}"
    return "https://wandb.ai"


def live_wandb_identity(project: str = "beamdojo") -> tuple[str, str | None, str]:
    """Active W&B run URL/entity/project once wandb.init has run; else env fallbacks."""
    entity = (os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME") or "").strip() or None
    proj = project
    url = None
    try:
        import wandb

        run = getattr(wandb, "run", None)
        if run is not None:
            url = getattr(run, "url", None)
            if not url and hasattr(run, "get_url"):
                url = run.get_url()
            ent = getattr(run, "entity", None)
            pr = getattr(run, "project", None)
            if ent:
                entity = str(ent)
            if pr:
                proj = str(pr)
    except Exception:
        pass
    if url:
        return str(url), entity, proj
    return wandb_project_url(proj), entity, proj


def live_wandb_url(project: str = "beamdojo") -> str:
    """Prefer the active W&B run URL once wandb.init has run; else the project page."""
    url, _, _ = live_wandb_identity(project)
    return url


def apply_wandb_defaults(agent_cfg, args_cli) -> None:
    """Use W&B when a key is present unless the user picked another logger."""
    sync_wandb_identity_env()
    logger = getattr(args_cli, "logger", None)
    if logger:
        agent_cfg.logger = logger
    elif os.environ.get("WANDB_API_KEY", "").strip():
        agent_cfg.logger = "wandb"
    project = getattr(args_cli, "log_project_name", None) or os.environ.get("WANDB_PROJECT", "beamdojo")
    if getattr(agent_cfg, "logger", None) in {"wandb", "neptune"}:
        agent_cfg.wandb_project = project
        agent_cfg.neptune_project = project


def _safe_training_status(payload: dict) -> Path | None:
    """Status JSON is for the Research Lab; never abort a live CUDA learn()."""
    try:
        return write_training_status(payload)
    except Exception as exc:
        print(f"[WARN] training-status.json write skipped ({type(exc).__name__}: {exc})")
        return None


def write_training_status(payload: dict) -> Path:
    """Write live-run JSON for Kingdom Research Lab (gitignored)."""
    project = str(payload.get("wandb_project") or os.environ.get("WANDB_PROJECT", "beamdojo"))
    url, entity, project = live_wandb_identity(project)
    body = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "unknown",
        "host": "lambda-a10" if Path("/lambda/nfs/beamdojo").is_dir() else "local",
        "wandb_project": project,
        "wandb_entity": entity,
        "wandb_url": url,
        **payload,
    }
    # A real wandb.run URL always wins over a stale project homepage in payload.
    if url and "/runs/" in url:
        body["wandb_url"] = url
        body["wandb_project"] = project
        if entity:
            body["wandb_entity"] = entity
    raw = json.dumps(body, indent=2) + "\n"
    targets = [REPO_ROOT / "tracking" / "training-status.json"]
    env_root = os.environ.get("BEAMDOJO_LOG_ROOT", "").strip()
    if env_root:
        targets.append(Path(env_root) / "training-status.json")
    nfs = Path("/lambda/nfs/beamdojo/logs/training-status.json")
    if nfs.parent.is_dir():
        targets.append(nfs)
    written = targets[0]
    for path in targets:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw)
            written = path
        except Exception as exc:
            print(f"[WARN] training-status.json write skipped ({path}: {type(exc).__name__}: {exc})")
    return written


def _status_from_runner(runner, payload: dict, *, iteration: int | None = None) -> dict:
    project = payload.get("wandb_project") or os.environ.get("WANDB_PROJECT", "beamdojo")
    url, entity, project = live_wandb_identity(project)
    it = int(iteration if iteration is not None else getattr(runner, "current_learning_iteration", 0) or 0)
    log_dir = payload.get("log_dir")
    ckpt = None
    if log_dir:
        candidate = os.path.join(str(log_dir), f"model_{it}.pt")
        if os.path.isfile(candidate):
            ckpt = candidate
    payload["wandb_url"] = url
    payload["wandb_entity"] = entity
    payload["wandb_project"] = project
    return {
        **payload,
        "status": "running",
        "iteration": it,
        "wandb_url": url,
        "wandb_entity": entity,
        "wandb_project": project,
        "checkpoint": ckpt or payload.get("checkpoint"),
    }


def _apply_live_run_note(payload: dict, runner) -> None:
    """Replace the Isaac-boot note once learn() has opened a logger."""
    url, _, _ = live_wandb_identity(str(payload.get("wandb_project") or "beamdojo"))
    if url and "/runs/" in url:
        payload["note"] = (
            "Live CUDA train. Open the W&B run URL for curves. "
            "Research Lab shows the last PPO snapshot every 10 iters."
        )
        return
    logger = str(payload.get("logger") or getattr(runner, "logger_type", None) or "")
    if logger == "tensorboard":
        payload["note"] = (
            "CUDA train running with TensorBoard after W&B logger setup failed. "
            "No live W&B run URL. ssh -L 6006:localhost:6006 and open NFS logs."
        )
        return
    payload["note"] = (
        "CUDA train running. No wandb.run URL yet — W&B project page or TensorBoard only."
    )


def attach_status_heartbeat(runner, payload: dict, *, every: int = 10) -> None:
    """Rewrite gitignored training-status.json when W&B inits and every N PPO iters.

    OnPolicyRunner._prepare_logging_writer runs wandb.init at the start of learn(),
    which is the first moment a real run URL exists. log() is called once per
    iteration after that. Kingdom syncs this file into the Research Lab.
    """
    orig_prepare = getattr(runner, "_prepare_logging_writer", None)
    if callable(orig_prepare):

        def _prepare(*args, **kwargs):
            try:
                result = orig_prepare(*args, **kwargs)
            except Exception as exc:
                if getattr(runner, "writer", None) is not None:
                    print(
                        f"[WARN] Logger post-init failed ({type(exc).__name__}: {exc}). "
                        "Continuing with the existing writer."
                    )
                    result = runner.writer
                else:
                    print(
                        f"[WARN] Logger setup failed ({type(exc).__name__}: {exc}). "
                        "Falling back to TensorBoard so the 10k run still starts."
                    )
                    result = _fallback_tensorboard_writer(runner)
                    payload["logger"] = "tensorboard"
            _apply_live_run_note(payload, runner)
            _safe_training_status(_status_from_runner(runner, payload))
            return result

        runner._prepare_logging_writer = _prepare

    orig_log = getattr(runner, "log", None)
    if not callable(orig_log):
        return

    def _log(*args, **kwargs):
        locs = args[0] if args else kwargs.get("locs")
        if isinstance(locs, dict):
            sanitize_ep_infos_for_rsl_log(locs.get("ep_infos"))
        try:
            result = orig_log(*args, **kwargs)
        except Exception as exc:
            print(f"[WARN] runner.log skipped ({type(exc).__name__}: {exc})")
            result = None
        it = int(getattr(runner, "current_learning_iteration", 0) or 0)
        if every > 0 and it % every != 0:
            return result
        locs = args[0] if args else kwargs.get("locs")
        try:
            metrics = extract_live_metrics(
                locs if isinstance(locs, dict) else {},
                num_envs=int(payload.get("num_envs") or 0),
                num_steps=int(
                    getattr(runner, "num_steps_per_env", 0) or payload.get("num_steps_per_env") or 0
                ),
            )
            payload.update(metrics)
            append_history(payload, metrics, it)
            _safe_training_status(_status_from_runner(runner, payload, iteration=it))
        except Exception as exc:
            print(f"[WARN] live-status heartbeat skipped ({type(exc).__name__}: {exc})")
        return result

    runner.log = _log


def mark_training_idle(note: str | None = None, **payload) -> Path:
    """Force idle so Kingdom never keeps a dead A10 marked running."""
    return write_training_status(
        {
            **payload,
            "status": "idle",
            "note": note
            or "Idle — GPU instance terminating or no train running. Checkpoints stay on NFS.",
        }
    )


def install_status_signal_hooks(payload: dict) -> None:
    """On SIGINT/SIGTERM, write idle before exit (Lambda terminate / Ctrl-C)."""
    import signal

    def _handle(signum, _frame):
        mark_training_idle(
            f"Caught signal {signum}; marking idle so Research Lab does not show a live train.",
            **payload,
        )
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _write_foothold_extras(extras, contrib) -> None:
    """Per-env term on top-level extras; scalar only under extras['log']."""
    extras["foothold_reward"] = contrib
    extras["foothold_penalty"] = contrib
    log = extras.get("log")
    if not isinstance(log, dict):
        return
    # Isaac already writes Episode_Reward/foothold_penalty on reset. Never
    # replace that 0-dim episode mean with a per-env [N] tensor.
    scalar = _scalar_ep_info_value(contrib)
    if scalar is not None:
        log["foothold_penalty"] = scalar


class FootholdExtrasWrapper:
    """Gym wrapper: put per-step foothold term into extras for the double critic.

    rsl-rl 3.0.1 ``OnPolicyRunner.log`` cats every ``extras['log']`` key. Keep the
    per-env ``[N]`` tensor on the top-level extras dict for ``PPODoubleCritic``.
    """

    def __init__(self, env):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        result = self.env.step(action)
        raw = self.unwrapped
        foot = getattr(raw, "beamdojo_foothold_step", None)
        if foot is None:
            return result
        dt = float(getattr(raw, "step_dt", 0.02) or 0.02)
        contrib = foot * dt
        info = result[-1]
        if isinstance(info, dict):
            _write_foothold_extras(info, contrib)
            raw_extras = getattr(raw, "extras", None)
            if isinstance(raw_extras, dict) and raw_extras is not info:
                _write_foothold_extras(raw_extras, contrib)
        return result

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def close(self):
        return self.env.close()
