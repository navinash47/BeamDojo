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


def leftover_universal_joint_expr(expr) -> bool:
    """``.*`` matches every H1 and G1 joint; do not treat it as the other robot."""
    return str(expr).strip() in {".*", ".+", "*"}


def names_as_list(names) -> list[str]:
    if names is None:
        return []
    items = names if isinstance(names, (list, tuple)) else [names]
    return [str(item) for item in items]


def names_match(left, right) -> bool:
    return tuple(names_as_list(left)) == tuple(names_as_list(right))


# Official H1 regexes that do not ``re.fullmatch`` G1 ``*_joint`` names.
# ``.*_hip_pitch`` / ``.*_knee`` / ``.*_shoulder_*`` miss ``*_joint`` the same way
# ``.*_hip_yaw`` does. ``.*_shoulder_.*`` *does* match G1 shoulders — keep it out.
H1_JOINTS_MISS_G1 = {
    ".*_hip_yaw",
    ".*_hip_roll",
    ".*_hip_pitch",
    ".*_knee",
    ".*_ankle",
    "torso",
    ".*_shoulder_pitch",
    ".*_shoulder_roll",
    ".*_shoulder_yaw",
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


# Actual USD joint names for leftover_joint_fullmatches_robot (Isaac 2.3.2 re.fullmatch).
H1_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "torso",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
)
_G1_SIDE_JOINTS = (
    "hip_pitch_joint",
    "hip_roll_joint",
    "hip_yaw_joint",
    "knee_joint",
    "ankle_pitch_joint",
    "ankle_roll_joint",
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_pitch_joint",
    "elbow_roll_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
    "five_joint",
    "three_joint",
    "six_joint",
    "four_joint",
    "zero_joint",
    "one_joint",
    "two_joint",
)
G1_JOINTS = tuple(f"{side}_{name}" for side in ("left", "right") for name in _G1_SIDE_JOINTS) + (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "torso_joint",
)
G1_FINGER_MARKERS = (
    "five_joint",
    "three_joint",
    "six_joint",
    "four_joint",
    "zero_joint",
    "one_joint",
    "two_joint",
)
_ACTUATOR_MAP_FIELDS = ("stiffness", "damping", "armature", "effort_limit", "velocity_limit")


def leftover_joint_fullmatches_robot(pattern, joints) -> bool:
    try:
        regex = re.compile(str(pattern))
    except re.error:
        return False
    return any(regex.fullmatch(joint) for joint in joints)


def leftover_joint_misses_robot(pattern, joints) -> bool:
    return not leftover_joint_fullmatches_robot(pattern, joints)


def leftover_g1_finger_name(name) -> bool:
    text = str(name).lower()
    return any(marker in text for marker in G1_FINGER_MARKERS)


def leftover_h1_joint_name(name) -> bool:
    """H1 leftover names that miss G1 ``*_joint`` fullmatch (not universal ``.*``)."""
    text = str(name)
    if leftover_universal_joint_expr(text):
        return False
    if leftover_h1_joints_for_g1(text):
        return True
    if leftover_joint_fullmatches_robot(text, G1_JOINTS):
        return False
    return "_joint" not in text


def leftover_g1_joint_name(name) -> bool:
    """G1 leftover ``*_joint`` / finger names that miss official H1 joints."""
    text = str(name)
    if leftover_universal_joint_expr(text):
        return False
    if leftover_g1_joints_for_h1(text) or leftover_g1_finger_name(text):
        return True
    if leftover_joint_fullmatches_robot(text, H1_JOINTS):
        return False
    return "_joint" in text


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
BEAMDOJO_STONE_COUNT = 24


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


def env_cfg_stage(env_cfg) -> int | None:
    """Stage from the env class. Leftover ``catcher`` is not the source of truth.

    Hydra can dump a Stage 2 catcher onto Stage 1 (then timeout-only is skipped and
    leftover ``base_contact`` fires on the plane at the first reset) or drop the
    catcher from Stage 2 (``RigidObject.reset`` / infinite fall before W&B).
    """
    name = type(env_cfg).__name__.lower()
    if re.search(r"stage[_]?2", name):
        return 2
    if re.search(r"stage[_]?1", name):
        return 1
    scene = getattr(env_cfg, "scene", None)
    if _scene_uses_stones(scene):
        return 2
    if getattr(scene, "catcher", None) is not None:
        return 2
    return None


def leftover_rigid_catcher(catcher) -> bool:
    """``InteractiveScene.reset`` cannot index a world ``/World/catcher`` RigidObject."""
    if catcher is None:
        return False
    return "RigidObject" in type(catcher).__name__


def leftover_uncloned_prim_path(path) -> bool:
    """Cloned 1024-env assets must use ``{ENV_REGEX_NS}`` or gym.make dies."""
    if path is None:
        return False
    text = str(path).strip()
    if not text:
        return False
    return "{ENV_REGEX_NS}" not in text and "{ENV_NS}" not in text


def leftover_asset_base_task_beam(beam) -> bool:
    """AssetBase cuboids are not in rigid-object views; Stage 2 falls through at first reset."""
    return leftover_asset_base_rigid(beam)


def leftover_asset_base_rigid(asset) -> bool:
    if asset is None:
        return False
    return "AssetBase" in type(asset).__name__


def _asset_spawn(asset):
    if asset is None:
        return None
    if isinstance(asset, dict):
        return asset.get("spawn")
    return getattr(asset, "spawn", None)


def leftover_disabled_collision_asset(asset) -> bool:
    """Stage 1 visual leftover ``collision_enabled=False`` falls through Stage 2 at reset."""
    spawn = _asset_spawn(asset)
    if spawn is None:
        return False
    props = spawn.get("collision_props") if isinstance(spawn, dict) else getattr(spawn, "collision_props", None)
    flag = None
    if isinstance(props, dict):
        flag = props.get("collision_enabled")
    elif props is not None:
        flag = getattr(props, "collision_enabled", None)
    if flag is None:
        flag = spawn.get("collision_enabled") if isinstance(spawn, dict) else getattr(spawn, "collision_enabled", None)
    return flag is False


def leftover_kinematic_robot(robot) -> bool:
    """Catcher cuboid leftover ``kinematic_enabled=True`` on the robot welds it in place."""
    spawn = _asset_spawn(robot)
    if spawn is None:
        return False
    props = spawn.get("rigid_props") if isinstance(spawn, dict) else getattr(spawn, "rigid_props", None)
    flag = props.get("kinematic_enabled") if isinstance(props, dict) else getattr(props, "kinematic_enabled", None)
    return flag is True


def leftover_disabled_gravity_robot(robot) -> bool:
    """Catcher leftover ``disable_gravity=True`` on the robot floats it off the beam."""
    spawn = _asset_spawn(robot)
    if spawn is None:
        return False
    props = spawn.get("rigid_props") if isinstance(spawn, dict) else getattr(spawn, "rigid_props", None)
    flag = props.get("disable_gravity") if isinstance(props, dict) else getattr(props, "disable_gravity", None)
    return flag is True


def leftover_fixed_root_robot(robot) -> bool:
    """Leftover ``fix_root_link=True`` welds the pelvis; first reset still runs but never walks."""
    spawn = _asset_spawn(robot)
    if spawn is None:
        return False
    props = spawn.get("articulation_props") if isinstance(spawn, dict) else getattr(spawn, "articulation_props", None)
    flag = props.get("fix_root_link") if isinstance(props, dict) else getattr(props, "fix_root_link", None)
    return flag is True


def leftover_self_collisions_robot(robot) -> bool:
    """Leftover ``enabled_self_collisions=True`` can explode PhysX at the first reset."""
    spawn = _asset_spawn(robot)
    if spawn is None:
        return False
    props = spawn.get("articulation_props") if isinstance(spawn, dict) else getattr(spawn, "articulation_props", None)
    flag = (
        props.get("enabled_self_collisions")
        if isinstance(props, dict)
        else getattr(props, "enabled_self_collisions", None)
    )
    return flag is True


def leftover_invalid_root_rot(rot) -> bool:
    """Zero / NaN / non-4-tuple quats NaN PhysX at the first reset — before W&B."""
    if rot is None:
        return False
    try:
        vals = [float(rot[i]) for i in range(4)]
    except (TypeError, ValueError, IndexError, KeyError):
        return True
    if any(value != value for value in vals):
        return True
    return sum(value * value for value in vals) ** 0.5 < 1e-3


def leftover_nucleus_path(value) -> bool:
    if value is None or isinstance(value, (int, float, bool)):
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return (
        text.startswith("omniverse://")
        or "/nvidia/" in text
        or "nucleus" in text
        or text.endswith(".mdl")
    )


def leftover_nucleus_visual_material(mat) -> bool:
    if mat is None:
        return False
    if leftover_nucleus_path(mat):
        return True
    for key in ("mdl_path", "texture_file", "usd_path", "mdl"):
        raw = mat.get(key) if isinstance(mat, dict) else getattr(mat, key, None)
        if leftover_nucleus_path(raw):
            return True
    return False


def leftover_disabled_replicate_physics(scene) -> bool:
    """``replicate_physics=False`` cannot clone 1024 GPU envs before W&B."""
    if scene is None or not hasattr(scene, "replicate_physics"):
        return False
    return scene.replicate_physics is False


SCENE_CAMERA_FIELDS = (
    "tiled_camera",
    "camera",
    "front_camera",
    "left_camera",
    "right_camera",
    "overhead_camera",
)


def leftover_scene_camera_fields(scene) -> list[str]:
    """PLAY leftover cameras × 1024 envs OOM the A10 at gym.make — before W&B."""
    if scene is None:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for key in (*SCENE_CAMERA_FIELDS, *_public_field_names(scene)):
        if key in seen or "camera" not in str(key).lower():
            continue
        seen.add(key)
        if getattr(scene, key, None) is not None:
            names.append(key)
    return names


def leftover_unfiltered_collisions(scene) -> bool:
    """``filter_collisions=False`` lets 1024 clones hit each other at first reset."""
    if scene is None or not hasattr(scene, "filter_collisions"):
        return False
    return scene.filter_collisions is False


def leftover_zero_num_envs(scene) -> bool:
    if scene is None or not hasattr(scene, "num_envs"):
        return False
    try:
        return int(scene.num_envs) < 1
    except (TypeError, ValueError):
        return True


def leftover_excess_num_envs(scene) -> bool:
    """Parent locomotion leftover ``4096`` OOMs A10 24GB at gym.make. Play 64 stays."""
    if scene is None or not hasattr(scene, "num_envs"):
        return False
    try:
        return int(scene.num_envs) > 1024
    except (TypeError, ValueError):
        return False


def leftover_clone_in_fabric(scene) -> bool:
    """Isaac 2.3.2 leftover Fabric clone hides USD prims Stage 2 reset writes before W&B."""
    if scene is None:
        return False
    return getattr(scene, "clone_in_fabric", None) is True


def leftover_stage_in_memory(sim) -> bool:
    """Leftover ``create_stage_in_memory`` cannot pair with cameras / Fabric and dies at gym.make."""
    if sim is None:
        return False
    return getattr(sim, "create_stage_in_memory", None) is True


def leftover_missing_robot(scene) -> bool:
    return scene is not None and getattr(scene, "robot", None) is None


def leftover_missing_terrain(scene) -> bool:
    return scene is not None and getattr(scene, "terrain", None) is None


def leftover_missing_contact_forces(scene) -> bool:
    return scene is not None and getattr(scene, "contact_forces", None) is None


def leftover_invalid_command_frac(value) -> bool:
    """Isaac 2.3.2 ``<= rel_standing_envs`` TypeErrors at first reset if leftover is None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return number != number or number < 0.0 or number > 1.0


def leftover_wrong_scene_asset_name(value) -> bool:
    """``env.scene[asset_name]`` KeyErrors at gym.make unless the name is ``robot``."""
    if value is None:
        return True
    return str(value).strip() != "robot"


def leftover_missing_base_velocity(env_cfg) -> bool:
    commands = getattr(env_cfg, "commands", None)
    if commands is None:
        return True
    if isinstance(commands, dict):
        return commands.get("base_velocity") is None
    return getattr(commands, "base_velocity", None) is None


def leftover_missing_joint_pos_action(env_cfg) -> bool:
    actions = getattr(env_cfg, "actions", None)
    if actions is None:
        return True
    if isinstance(actions, dict):
        return actions.get("joint_pos") is None
    return getattr(actions, "joint_pos", None) is None


def leftover_invalid_rel_frac(cmd, name: str) -> bool:
    """Isaac samples ``rel_standing_envs`` / ``rel_heading_envs`` even when heading is off."""
    if cmd is None:
        return False
    if isinstance(cmd, dict):
        return leftover_invalid_command_frac(cmd.get(name))
    if not hasattr(cmd, name):
        return True
    return leftover_invalid_command_frac(getattr(cmd, name))


def leftover_missing_class_type(cfg) -> bool:
    """Command/Action manager calls ``cfg.class_type(...)`` at gym.make."""
    if cfg is None:
        return False
    if isinstance(cfg, dict):
        return cfg.get("class_type") is None
    return getattr(cfg, "class_type", None) is None


def leftover_wait_for_textures(sim) -> bool:
    """Isaac default True stalls gym.make when leftover Nucleus materials never load."""
    if sim is None:
        return False
    if isinstance(sim, dict):
        if "wait_for_textures" not in sim:
            return False
        return sim.get("wait_for_textures") is not False
    if not hasattr(sim, "wait_for_textures"):
        return False
    return getattr(sim, "wait_for_textures") is not False


def leftover_missing_sim(env_cfg) -> bool:
    """``env_cfg.sim is None`` dies in SimulationContext at gym.make."""
    return env_cfg is not None and getattr(env_cfg, "sim", None) is None


def leftover_missing_policy_obs(env_cfg) -> bool:
    """ObservationManager requires ``observations.policy`` at gym.make."""
    obs = getattr(env_cfg, "observations", None)
    if obs is None:
        return True
    if isinstance(obs, dict):
        return obs.get("policy") is None
    return getattr(obs, "policy", None) is None


def leftover_excess_obs_history(value) -> bool:
    """Any leftover obs history explodes A10 memory at gym.make. BeamDojo uses 0."""
    if value is None:
        return False
    try:
        number = int(value)
    except (TypeError, ValueError):
        return True
    return number != number or number < 0 or number > 0


def leftover_invalid_action_scale(value) -> bool:
    """``scale=None`` TypeErrors JointPositionAction at gym.make. Dict maps stay."""
    if isinstance(value, dict):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return number != number or number <= 0.0


def leftover_invalid_action_offset(value) -> bool:
    """Isaac 2.3.2 JointAction only accepts float/int or dict offset at gym.make."""
    if isinstance(value, dict):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return number != number


def leftover_invalid_action_clip(value) -> bool:
    """Isaac 2.3.2 JointAction only accepts clip=None or dict. Tuple leftover ValueErrors at gym.make."""
    if value is None:
        return False
    if isinstance(value, dict):
        return any(not _range_pair_ok(span) for span in value.values())
    return True


def leftover_missing_term_func(term) -> bool:
    """Manager terms call ``cfg.func`` at gym.make. Hydra leftover ``func=None`` dies."""
    if term is None:
        return False
    if isinstance(term, dict):
        return "func" in term and term.get("func") is None
    return hasattr(term, "func") and getattr(term, "func") is None


def leftover_missing_rewards(env_cfg) -> bool:
    return env_cfg is not None and getattr(env_cfg, "rewards", None) is None


def leftover_missing_events(env_cfg) -> bool:
    return env_cfg is not None and getattr(env_cfg, "events", None) is None


def leftover_missing_terminations(env_cfg) -> bool:
    return env_cfg is not None and getattr(env_cfg, "terminations", None) is None


def leftover_missing_curriculum(env_cfg) -> bool:
    return env_cfg is not None and getattr(env_cfg, "curriculum", None) is None


def leftover_missing_time_out(env_cfg) -> bool:
    """Stage 1 is timeout-only. Missing ``time_out`` never ends an episode."""
    terms = getattr(env_cfg, "terminations", None) if env_cfg is not None else None
    if terms is None:
        return True
    if isinstance(terms, dict):
        return terms.get("time_out") is None
    return getattr(terms, "time_out", None) is None


def leftover_invalid_obs_scale(value) -> bool:
    """``scale=None`` TypeErrors ObservationManager at first reset — before W&B."""
    if isinstance(value, dict):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return number != number or number <= 0.0


def leftover_invalid_obs_clip(value) -> bool:
    """Leftover ``clip=(None, None)`` / bool TypeErrors ``obs.clip`` at first reset."""
    if value is None or value is False:
        return False
    return not _range_pair_ok(value)


def leftover_disabled_fabric(sim) -> bool:
    """Leftover ``use_fabric=False`` hides Stage 2 USD prim writes at first reset."""
    if sim is None:
        return False
    if isinstance(sim, dict):
        return sim.get("use_fabric") is False
    return getattr(sim, "use_fabric", None) is False


def leftover_disabled_contact_processing(sim) -> bool:
    """Leftover True empties ContactSensor at first reset (feet_air_time / Stage 2 fall)."""
    if sim is None:
        return False
    if isinstance(sim, dict):
        return sim.get("disable_contact_processing") is True
    return getattr(sim, "disable_contact_processing", None) is True


def leftover_missing_physx(sim) -> bool:
    """SimulationContext reads ``sim.physx`` at gym.make. Hydra leftover ``physx=None`` dies."""
    if sim is None:
        return False
    if isinstance(sim, dict):
        return sim.get("physx") is None
    return getattr(sim, "physx", None) is None


def leftover_invalid_gravity(sim) -> bool:
    """Leftover ``gravity=None`` / non-triple TypeErrors SimulationContext at gym.make."""
    if sim is None:
        return False
    if isinstance(sim, dict):
        if "gravity" not in sim:
            return False
        gravity = sim.get("gravity")
    elif not hasattr(sim, "gravity"):
        return False
    else:
        gravity = sim.gravity
    try:
        x, y, z = float(gravity[0]), float(gravity[1]), float(gravity[2])
    except (TypeError, ValueError, IndexError, KeyError):
        return True
    return any(value != value for value in (x, y, z))


def leftover_missing_scene(env_cfg) -> bool:
    """InteractiveScene dies at gym.make if Hydra nulled ``env_cfg.scene``."""
    return env_cfg is not None and getattr(env_cfg, "scene", None) is None


def leftover_missing_reset_base(env_cfg) -> bool:
    """No ``reset_base`` leaves the H1 in the plane; PhysX NaNs at first reset."""
    events = getattr(env_cfg, "events", None) if env_cfg is not None else None
    if events is None:
        return True
    if isinstance(events, dict):
        return events.get("reset_base") is None
    return getattr(events, "reset_base", None) is None


def leftover_missing_reset_joints(env_cfg) -> bool:
    events = getattr(env_cfg, "events", None) if env_cfg is not None else None
    if events is None:
        return True
    if isinstance(events, dict):
        return events.get("reset_robot_joints") is None
    return getattr(events, "reset_robot_joints", None) is None


def leftover_invalid_obs_noise(noise) -> bool:
    """Leftover ``noise.n_min=None`` / ``std=None`` TypeErrors at first reset."""
    if noise is None:
        return False
    keys = ("n_min", "n_max", "std", "mean")
    if isinstance(noise, dict):
        return any(k in noise and leftover_invalid_noise_number(noise.get(k)) for k in keys)
    return any(hasattr(noise, k) and leftover_invalid_noise_number(getattr(noise, k)) for k in keys)


def leftover_invalid_noise_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return number != number


VALID_EVENT_MODES = {"startup", "reset", "interval"}


def leftover_event_mode(term):
    if term is None:
        return None
    if isinstance(term, dict):
        return term.get("mode")
    return getattr(term, "mode", None)


def leftover_invalid_event_mode(term) -> bool:
    """EventManager indexes ``term.mode`` at gym.make. Leftover None / Isaac 2.4 names KeyError."""
    if term is None:
        return False
    if isinstance(term, dict):
        if "mode" not in term:
            return False
    elif not hasattr(term, "mode"):
        return False
    mode = leftover_event_mode(term)
    if not isinstance(mode, str) or not mode.strip():
        return True
    return mode.strip().lower() not in VALID_EVENT_MODES


def leftover_invalid_interval_event(term) -> bool:
    """Leftover ``mode=interval`` with ``interval_range_s=None`` TypeErrors at gym.make."""
    if str(leftover_event_mode(term) or "").strip().lower() != "interval":
        return False
    span = term.get("interval_range_s") if isinstance(term, dict) else getattr(term, "interval_range_s", None)
    return not _range_pair_ok(span)


def leftover_invalid_min_step_count(term) -> bool:
    """EventManager does ``min_step_count_between_reset < 0`` for reset terms at gym.make."""
    if term is None:
        return False
    if isinstance(term, dict):
        if "min_step_count_between_reset" not in term:
            return False
        value = term.get("min_step_count_between_reset")
    elif not hasattr(term, "min_step_count_between_reset"):
        return False
    else:
        value = getattr(term, "min_step_count_between_reset")
    try:
        return int(value) < 0
    except (TypeError, ValueError):
        return True


def leftover_invalid_term_params(term) -> bool:
    """ManagerBase does ``term_cfg.params.keys()`` at gym.make. Leftover ``params=None`` dies."""
    if term is None:
        return False
    if isinstance(term, dict):
        if "params" not in term:
            return False
        return not isinstance(term.get("params"), dict)
    if not hasattr(term, "params"):
        return False
    return not isinstance(getattr(term, "params"), dict)


def leftover_invalid_reward_weight(term) -> bool:
    """RewardManager TypeErrors unless ``weight`` is float/int at gym.make — before W&B."""
    if term is None:
        return False
    if isinstance(term, dict):
        if "weight" not in term:
            return False
        value = term.get("weight")
    elif not hasattr(term, "weight"):
        return False
    else:
        value = getattr(term, "weight")
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return True
    return value != value


def leftover_invalid_env_spacing(scene) -> bool:
    """``None`` / parent ``2.5`` overlaps 1024 H1s at gym.make. Play ``6.0`` stays."""
    if scene is None or not hasattr(scene, "env_spacing"):
        return False
    try:
        spacing = float(scene.env_spacing)
    except (TypeError, ValueError):
        return True
    return spacing != spacing or spacing < 6.0


def leftover_contact_filter_prims(contact) -> bool:
    """ANYmal/Cassie leftover filters die at gym.make (PhysX count / one-to-many)."""
    if contact is None:
        return False
    expr = (
        contact.get("filter_prim_paths_expr")
        if isinstance(contact, dict)
        else getattr(contact, "filter_prim_paths_expr", None)
    )
    return any(str(item).strip() for item in names_as_list(expr))


def leftover_wrong_contact_prim(path) -> bool:
    """ANYmal leftover ``.../Robot/base`` / ``LF_FOOT`` finds 0 H1/G1 contact bodies."""
    if path is None:
        return False
    text = str(path).strip()
    if not text or leftover_uncloned_prim_path(text):
        return False
    normalized = text.replace("{ENV_NS}", "{ENV_REGEX_NS}")
    if normalized == "{ENV_REGEX_NS}/Robot/.*":
        return False
    lower = text.lower()
    if leftover_quadruped_joint_key(text):
        return True
    if re.search(r"/(?:base|lf_foot|rf_foot|lh_foot|rh_foot|fl_foot|fr_foot)(?:/|$)", lower):
        return True
    if "terrain" in lower or "*_foot" in lower or "/object" in lower:
        return True
    return "/robot/.*" not in lower


def leftover_track_contact_points(contact) -> bool:
    """Isaac 2.3.2 ValueErrors ``track_contact_points=True`` with an empty filter."""
    if contact is None:
        return False
    flag = (
        contact.get("track_contact_points")
        if isinstance(contact, dict)
        else getattr(contact, "track_contact_points", None)
    )
    return flag is True


def leftover_zero_contact_data_count(contact) -> bool:
    if contact is None:
        return False
    if isinstance(contact, dict):
        if "max_contact_data_count_per_prim" not in contact:
            return False
        raw = contact.get("max_contact_data_count_per_prim")
    elif not hasattr(contact, "max_contact_data_count_per_prim"):
        return False
    else:
        raw = contact.max_contact_data_count_per_prim
    try:
        return int(raw) < 1
    except (TypeError, ValueError):
        return True


def leftover_missing_scene_entity_term(term, present: set[str]) -> bool:
    """True when a manager term still looks up a scene entity Hydra leftover never spawned."""
    if term is None:
        return False
    params = getattr(term, "params", None)
    if params is None:
        return False
    for key in ("sensor_cfg", "asset_cfg"):
        entity = params.get(key) if isinstance(params, dict) else getattr(params, key, None)
        name = _cfg_entity_name(entity)
        if name and name not in present:
            return True
    return False


def _cfg_entity_name(entity) -> str:
    if entity is None:
        return ""
    if isinstance(entity, dict):
        return str(entity.get("name") or "")
    return str(getattr(entity, "name", "") or "")


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
    """ANYmal/Go1/A1 leftover keys miss H1/G1 ``re.fullmatch`` at gym.make."""
    return bool(
        re.search(
            r"("
            r"HAA|HFE|KFE|calf|thigh_joint|F\[L,R\]|R\[L,R\]|"
            r"[LR]_hip_joint|"
            r"(?:^|[._*])(?:LF|RF|LH|RH|FL|FR|RL|RR)(?:[._*]|$)"
            r")",
            str(key),
            flags=re.IGNORECASE,
        )
    )


def leftover_quadruped_actuator(actuator) -> bool:
    """Go1 leftover ``network_file`` hits Nucleus; ANYmal ``.*HAA`` misses H1/G1."""
    if actuator is None:
        return False
    network = (
        actuator.get("network_file") if isinstance(actuator, dict) else getattr(actuator, "network_file", None)
    )
    if network:
        return True
    names = (
        actuator.get("joint_names_expr")
        if isinstance(actuator, dict)
        else getattr(actuator, "joint_names_expr", None)
    )
    if any(leftover_quadruped_joint_key(item) for item in names_as_list(names)):
        return True
    for field in _ACTUATOR_MAP_FIELDS:
        mapping = actuator.get(field) if isinstance(actuator, dict) else getattr(actuator, field, None)
        if isinstance(mapping, dict) and any(leftover_quadruped_joint_key(key) for key in mapping):
            return True
    return False


def leftover_actuator_joint_names(actuator) -> list[str]:
    if actuator is None:
        return []
    names = actuator.get("joint_names_expr") if isinstance(actuator, dict) else getattr(
        actuator, "joint_names_expr", None
    )
    items = names_as_list(names)
    for field in _ACTUATOR_MAP_FIELDS:
        mapping = actuator.get(field) if isinstance(actuator, dict) else getattr(actuator, field, None)
        if isinstance(mapping, dict):
            items.extend(str(key) for key in mapping)
    return items


def leftover_actuator_misses_robot(actuator, joints) -> bool:
    return any(
        leftover_joint_misses_robot(name, joints)
        for name in leftover_actuator_joint_names(actuator)
        if not leftover_universal_joint_expr(name)
    )


def leftover_h1_actuator(actuator) -> bool:
    return any(leftover_h1_joint_name(name) for name in leftover_actuator_joint_names(actuator))


def leftover_g1_actuator(actuator) -> bool:
    return any(leftover_g1_joint_name(name) for name in leftover_actuator_joint_names(actuator))


def leftover_quadruped_actuators(env_cfg) -> bool:
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot is None:
        return False
    actuators = getattr(robot, "actuators", None)
    if actuators is None and isinstance(robot, dict):
        actuators = robot.get("actuators")
    return any(leftover_quadruped_actuator(act) for _, act in _actuator_items(actuators))


def _actuator_items(actuators):
    if actuators is None:
        return []
    if isinstance(actuators, dict):
        return list(actuators.items())
    data = getattr(actuators, "__dict__", None) or {}
    return [(key, value) for key, value in data.items() if not str(key).startswith("_")]


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


def leftover_wrong_robot_joint_key(key, env_cfg) -> bool:
    """Hydra leftover H1 keys on G1 (or reverse) ValueError at gym.make."""
    if leftover_quadruped_joint_key(key):
        return True
    if leftover_universal_joint_expr(key):
        return False
    if _treat_as_g1(env_cfg):
        return leftover_h1_joint_name(key) or leftover_joint_misses_robot(key, G1_JOINTS)
    if _treat_as_h1(env_cfg):
        return leftover_g1_joint_name(key) or leftover_joint_misses_robot(key, H1_JOINTS)
    return False


def leftover_wrong_robot_actuator(actuator, env_cfg) -> bool:
    if leftover_quadruped_actuator(actuator):
        return True
    if _treat_as_g1(env_cfg):
        return leftover_h1_actuator(actuator) or leftover_actuator_misses_robot(actuator, G1_JOINTS)
    if _treat_as_h1(env_cfg):
        return leftover_g1_actuator(actuator) or leftover_actuator_misses_robot(actuator, H1_JOINTS)
    return False


def leftover_wrong_robot_actuators(env_cfg) -> bool:
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot is None:
        return False
    actuators = getattr(robot, "actuators", None)
    if actuators is None and isinstance(robot, dict):
        actuators = robot.get("actuators")
    return any(leftover_wrong_robot_actuator(act, env_cfg) for _, act in _actuator_items(actuators))


def _action_joint_names(env_cfg):
    joint_pos = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if joint_pos is None:
        return None, None
    names = joint_pos.get("joint_names") if isinstance(joint_pos, dict) else getattr(joint_pos, "joint_names", None)
    return joint_pos, names


def leftover_wrong_action_joint_names(env_cfg) -> bool:
    """ANYmal ``.*HAA`` / H1-on-G1 action names resolve 0 joints at gym.make."""
    _, names = _action_joint_names(env_cfg)
    if leftover_all_joints(names) or not names:
        return False
    return any(leftover_wrong_robot_joint_key(item, env_cfg) for item in names_as_list(names))


def leftover_wrong_event_joint_names(term, env_cfg) -> bool:
    """Leftover ``events.actuator_gains`` ``.*HAA`` ValueErrors EventManager at gym.make."""
    names = _entity_joint_names(_term_entity_cfg(term, "asset_cfg"))
    if leftover_all_joints(names) or not names:
        return False
    return any(leftover_wrong_robot_joint_key(item, env_cfg) for item in names_as_list(names))


def _scene_uses_stones(scene) -> bool:
    return scene is not None and getattr(scene, "task_stone_0", None) is not None


def env_cfg_uses_stones(env_cfg) -> bool:
    """Stones from the env class. Leftover ``task_stone_0`` is not the only signal.

    Hydra can drop every stone from ``BeamDojoStage2StonesEnvCfg`` (then a leftover
    beam is restored and the 10k is not stones) or leave AssetBase pads that fall
    through at wrapper reset — before W&B.
    """
    name = type(env_cfg).__name__.lower()
    if re.search(r"stone", name):
        return True
    return _scene_uses_stones(getattr(env_cfg, "scene", None))


def _reassert_physx_floors(env_cfg) -> None:
    """Parent locomotion leftover is ``10 * 2**15`` patches — too small for cloned beams."""
    try:
        from h1_cfg.physx_gpu import PHYSX_A10_UNSAFE_FLOOR, apply_physx_gpu_capacity
    except ImportError as exc:
        print(f"[WARN] PhysX floor reassert skipped ({type(exc).__name__}: {exc})")
        return
    scene = getattr(env_cfg, "scene", None)
    try:
        apply_physx_gpu_capacity(env_cfg, stones=env_cfg_uses_stones(env_cfg))
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
    if env_cfg_stage(env_cfg) == 2:
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
    need = (
        leftover_missing_robot(getattr(env_cfg, "scene", None))
        or leftover_quadruped_robot(env_cfg)
        or leftover_full_unitree_usd(env_cfg)
        or leftover_quadruped_actuators(env_cfg)
        or leftover_wrong_robot_actuators(env_cfg)
    )
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
        if env_cfg_stage(env_cfg) != 2:
            env_cfg.scene.robot.init_state.pos = (0.0, 0.0, spec.pelvis_z)
        return
    except ImportError:
        pass
    from h1_cfg.robot_spec import G1, H1

    spec = G1 if spec_name == "g1" else H1
    _stamp_unitree_usd(env_cfg, spec_name)
    _set_init_pelvis_z(env_cfg, expected_pelvis_z(env_cfg, spec))
    _drop_leftover_actuators(env_cfg)


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
    """Hydra leftover ANYmal / H1-on-G1 / G1-on-H1 keys ValueError at gym.make."""
    state = _robot_init_state(env_cfg)
    if state is None:
        return
    for field in ("joint_pos", "joint_vel"):
        mapping = _joint_map(state, field)
        if not mapping:
            continue
        cleaned = {
            key: mapping[key] for key in mapping if not leftover_wrong_robot_joint_key(key, env_cfg)
        }
        if len(cleaned) == len(mapping):
            continue
        dropped = [key for key in mapping if key not in cleaned]
        print(f"[WARN] Dropping leftover init_state.{field} keys {dropped}.")
        _set_joint_map(state, field, cleaned)


def _robot_attr(env_cfg, name: str):
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot is None:
        return None
    value = getattr(robot, name, None)
    if value is None and isinstance(robot, dict):
        value = robot.get(name)
    return value


def _drop_leftover_actuators(env_cfg) -> None:
    """Isaac-free path: drop leftover Go1 nets / ANYmal / H1↔G1 actuator groups."""
    actuators = _robot_attr(env_cfg, "actuators")
    if actuators is None:
        return
    if isinstance(actuators, dict):
        drop = [
            name
            for name, act in list(actuators.items())
            if leftover_wrong_robot_actuator(act, env_cfg)
        ]
        for name in drop:
            print(f"[WARN] Dropping leftover robot.actuators[{name!r}] (quad / wrong-robot / Nucleus).")
            actuators.pop(name, None)
        return
    for name, act in _actuator_items(actuators):
        if leftover_wrong_robot_actuator(act, env_cfg) and hasattr(actuators, name):
            print(f"[WARN] Dropping leftover robot.actuators.{name} (quad / wrong-robot / Nucleus).")
            setattr(actuators, name, None)


def _reassert_action_joint_maps(env_cfg) -> None:
    """Leftover ``scale={{'.*HAA': 0.5}}`` / H1-on-G1 keys ValueError at gym.make."""
    joint_pos = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if joint_pos is None:
        return
    for field in ("scale", "offset"):
        value = joint_pos.get(field) if isinstance(joint_pos, dict) else getattr(joint_pos, field, None)
        if not isinstance(value, dict):
            continue
        cleaned = {key: value[key] for key in value if not leftover_wrong_robot_joint_key(key, env_cfg)}
        if len(cleaned) == len(value):
            continue
        print(f"[WARN] Dropping leftover actions.joint_pos.{field} keys.")
        restored: dict | float | None = cleaned if cleaned else (0.25 if field == "scale" else None)
        if isinstance(joint_pos, dict):
            joint_pos[field] = restored
        else:
            setattr(joint_pos, field, restored)


def _reassert_contact_sensors(env_cfg) -> None:
    """Leftover ``activate_contact_sensors=False`` empties feet_air_time at first reset."""
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn = getattr(robot, "spawn", None) if robot is not None else None
    if spawn is None and isinstance(robot, dict):
        spawn = robot.get("spawn")
    if spawn is None:
        return
    flag = spawn.get("activate_contact_sensors") if isinstance(spawn, dict) else getattr(
        spawn, "activate_contact_sensors", None
    )
    if flag is False:
        print("[WARN] Enabling leftover activate_contact_sensors (feet_air_time at first reset).")
        if isinstance(spawn, dict):
            spawn["activate_contact_sensors"] = True
        elif hasattr(spawn, "activate_contact_sensors"):
            spawn.activate_contact_sensors = True


def _scene_entity_names(env_cfg) -> set[str]:
    names: set[str] = set()
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return names
    for key in _public_field_names(scene):
        if getattr(scene, key, None) is not None:
            names.add(str(key))
    for known in ("robot", "contact_forces", "terrain", "task_beam", "catcher", "sky_light"):
        if getattr(scene, known, None) is not None:
            names.add(known)
    return names


def leftover_missing_sensor_term(term, present: set[str]) -> bool:
    """True when an obs term still looks up a sensor Hydra leftover never spawned."""
    if term is None or parent_raycast_height_scan(term):
        return False
    params = getattr(term, "params", None)
    sensor = None
    if isinstance(params, dict):
        sensor = params.get("sensor_cfg")
    elif params is not None:
        sensor = getattr(params, "sensor_cfg", None)
    name = _sensor_cfg_name(sensor)
    return bool(name) and name not in present


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
    if cmd is not None and (
        hasattr(cmd, "resampling_time_range") or (isinstance(cmd, dict) and "resampling_time_range" in cmd)
    ):
        raw = cmd.get("resampling_time_range") if isinstance(cmd, dict) else getattr(cmd, "resampling_time_range", None)
        if not _range_pair_ok(raw):
            print(f"[WARN] Restoring leftover commands.base_velocity.resampling_time_range={raw!r}.")
            if isinstance(cmd, dict):
                cmd["resampling_time_range"] = (10.0, 10.0)
            else:
                cmd.resampling_time_range = (10.0, 10.0)
    if ranges is None:
        return
    stage2 = env_cfg_stage(env_cfg) == 2
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


def _manager_get(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _manager_set(obj, name, value) -> None:
    if obj is None:
        return
    if isinstance(obj, dict):
        obj[name] = value
        return
    setattr(obj, name, value)


def _base_velocity_stub(env_cfg):
    stage2 = env_cfg_stage(env_cfg) == 2
    return type(
        "UniformVelocityCommandCfg",
        (),
        {
            "asset_name": "robot",
            "heading_command": False,
            "debug_vis": False,
            "resampling_time_range": (10.0, 10.0),
            "rel_standing_envs": 0.1 if stage2 else 0.5,
            "rel_heading_envs": 0.0,
            "ranges": type(
                "R",
                (),
                {
                    "lin_vel_x": (0.2, 0.8) if stage2 else (-1.0, 1.0),
                    "lin_vel_y": (-0.15, 0.15) if stage2 else (-1.0, 1.0),
                    "ang_vel_z": (-0.4, 0.4) if stage2 else (-1.0, 1.0),
                    "heading": None,
                },
            )(),
        },
    )()


def _joint_pos_action_stub(env_cfg):
    names = [".*"]
    if _treat_as_g1(env_cfg):
        try:
            from h1_cfg.robot_spec import G1

            names = list(G1.action_joints)
        except ImportError:
            names = [
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ]
    return type("JointPositionActionCfg", (), {"asset_name": "robot", "joint_names": names, "scale": 0.25})()


def _reassert_velocity_command(env_cfg) -> None:
    """Leftover None ``rel_*`` / missing ``base_velocity`` dies at first reset before W&B."""
    if leftover_missing_base_velocity(env_cfg):
        print("[WARN] Restoring leftover commands.base_velocity (UniformVelocityCommand at gym.make).")
        commands = getattr(env_cfg, "commands", None)
        if commands is None:
            env_cfg.commands = type("Commands", (), {})()
            commands = env_cfg.commands
        try:
            from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg

            restored = CommandsCfg().base_velocity
            restored.heading_command = False
            restored.debug_vis = False
        except ImportError:
            restored = _base_velocity_stub(env_cfg)
        _manager_set(commands, "base_velocity", restored)

    cmd = _manager_get(getattr(env_cfg, "commands", None), "base_velocity")
    if cmd is None:
        return
    if leftover_wrong_scene_asset_name(_manager_get(cmd, "asset_name")):
        print(f"[WARN] Restoring leftover commands.base_velocity.asset_name={_manager_get(cmd, 'asset_name')!r} to robot.")
        _manager_set(cmd, "asset_name", "robot")
    if leftover_invalid_rel_frac(cmd, "rel_standing_envs"):
        standing = 0.1 if env_cfg_stage(env_cfg) == 2 else 0.5
        print(f"[WARN] Restoring leftover commands.base_velocity.rel_standing_envs to {standing}.")
        _manager_set(cmd, "rel_standing_envs", standing)
    if leftover_invalid_rel_frac(cmd, "rel_heading_envs"):
        print("[WARN] Restoring leftover commands.base_velocity.rel_heading_envs to 0.0.")
        _manager_set(cmd, "rel_heading_envs", 0.0)
    if leftover_missing_class_type(cmd):
        print("[WARN] Restoring leftover commands.base_velocity.class_type (CommandManager at gym.make).")
        try:
            from isaaclab.envs.mdp.commands import UniformVelocityCommand

            _manager_set(cmd, "class_type", UniformVelocityCommand)
        except ImportError:
            _manager_set(cmd, "class_type", type("UniformVelocityCommand", (), {}))
    # Official BeamDojo keeps heading off. Isaac 2.3.2 ValueErrors heading=True + ranges.heading=None.
    _manager_set(cmd, "heading_command", False)
    _manager_set(cmd, "debug_vis", False)


def _reassert_missing_joint_pos_action(env_cfg) -> None:
    """Leftover nulled ``actions.joint_pos`` KeyErrors ActionManager at gym.make."""
    if leftover_missing_joint_pos_action(env_cfg):
        print("[WARN] Restoring leftover actions.joint_pos (ActionManager at gym.make).")
        actions = getattr(env_cfg, "actions", None)
        if actions is None:
            env_cfg.actions = type("Actions", (), {})()
            actions = env_cfg.actions
        try:
            from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg

            restored = ActionsCfg().joint_pos
            if _treat_as_g1(env_cfg):
                from h1_cfg.robot_spec import G1

                restored.joint_names = list(G1.action_joints)
        except ImportError:
            restored = _joint_pos_action_stub(env_cfg)
        _manager_set(actions, "joint_pos", restored)
    joint_pos = _manager_get(getattr(env_cfg, "actions", None), "joint_pos")
    if leftover_wrong_scene_asset_name(_manager_get(joint_pos, "asset_name")):
        print(
            f"[WARN] Restoring leftover actions.joint_pos.asset_name="
            f"{_manager_get(joint_pos, 'asset_name')!r} to robot."
        )
        _manager_set(joint_pos, "asset_name", "robot")
    if leftover_missing_class_type(joint_pos):
        print("[WARN] Restoring leftover actions.joint_pos.class_type (ActionManager at gym.make).")
        try:
            from isaaclab.envs.mdp.actions import JointPositionAction

            _manager_set(joint_pos, "class_type", JointPositionAction)
        except ImportError:
            _manager_set(joint_pos, "class_type", type("JointPositionAction", (), {}))
    if joint_pos is not None and (
        isinstance(joint_pos, dict) or hasattr(joint_pos, "scale")
    ) and leftover_invalid_action_scale(_manager_get(joint_pos, "scale")):
        print("[WARN] Restoring leftover actions.joint_pos.scale to 0.25.")
        _manager_set(joint_pos, "scale", 0.25)
    if joint_pos is not None and (
        isinstance(joint_pos, dict) or hasattr(joint_pos, "offset")
    ) and leftover_invalid_action_offset(_manager_get(joint_pos, "offset")):
        print("[WARN] Restoring leftover actions.joint_pos.offset to 0.0.")
        _manager_set(joint_pos, "offset", 0.0)
    if joint_pos is not None and (
        isinstance(joint_pos, dict) or hasattr(joint_pos, "clip")
    ) and leftover_invalid_action_clip(_manager_get(joint_pos, "clip")):
        print("[WARN] Clearing leftover actions.joint_pos.clip (Isaac JointAction only accepts dict).")
        _manager_set(joint_pos, "clip", None)


def _public_field_names(obj) -> list[str]:
    """Instance + class fields. ``type('X', (), {field: ...})`` stores on the class."""
    names: list[str] = []
    seen: set[str] = set()
    for source in (getattr(obj, "__dict__", None) or {}, getattr(type(obj), "__dict__", None) or {}):
        for key in source:
            if str(key).startswith("_") or key in seen:
                continue
            value = source[key]
            if callable(value) and not isinstance(value, type):
                continue
            seen.add(key)
            names.append(str(key))
    return names


def _iter_obs_groups(obs):
    if obs is None:
        return
    names = _public_field_names(obs)
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
    for key in _public_field_names(group):
        yield key, getattr(group, key, None)


def _reassert_missing_policy_obs(env_cfg) -> None:
    """Leftover nulled ``observations.policy`` dies in ObservationManager at gym.make."""
    if not leftover_missing_policy_obs(env_cfg):
        return
    print("[WARN] Restoring leftover observations.policy (ObservationManager at gym.make).")
    try:
        from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

        policy = ObservationsCfg().policy
    except ImportError:
        policy = type(
            "ObsGroup",
            (),
            {"concatenate_terms": True, "flatten_history_dim": True, "history_length": 0},
        )()
    obs = getattr(env_cfg, "observations", None)
    if obs is None:
        env_cfg.observations = type("Observations", (), {"policy": policy})()
        return
    _manager_set(obs, "policy", policy)


def _restore_manager_cfg(kind: str):
    try:
        from h1_cfg.beamdojo_env_base import (
            BeamDojoCurriculumCfg,
            BeamDojoEventCfg,
            BeamDojoRewardsCfg,
            BeamDojoTerminationsCfg,
        )

        return {
            "rewards": BeamDojoRewardsCfg,
            "events": BeamDojoEventCfg,
            "terminations": BeamDojoTerminationsCfg,
            "curriculum": BeamDojoCurriculumCfg,
        }[kind]()
    except ImportError:
        pass
    try:
        from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
            CurriculumCfg,
            EventCfg,
            RewardsCfg,
            TerminationsCfg,
        )

        return {
            "rewards": RewardsCfg,
            "events": EventCfg,
            "terminations": TerminationsCfg,
            "curriculum": CurriculumCfg,
        }[kind]()
    except ImportError:
        if kind == "terminations":
            return type(
                "Terminations",
                (),
                {"time_out": type("DoneTerm", (), {"func": type("F", (), {"__name__": "time_out"})()})()},
            )()
        return type(kind.title(), (), {})()


def _reassert_missing_managers(env_cfg) -> None:
    """Leftover nulled rewards/events/terminations/curriculum die in managers at gym.make."""
    if leftover_missing_rewards(env_cfg):
        print("[WARN] Restoring leftover rewards (RewardManager at gym.make).")
        env_cfg.rewards = _restore_manager_cfg("rewards")
    if leftover_missing_events(env_cfg):
        print("[WARN] Restoring leftover events (EventManager at gym.make).")
        env_cfg.events = _restore_manager_cfg("events")
    if leftover_missing_terminations(env_cfg):
        print("[WARN] Restoring leftover terminations (TerminationManager at gym.make).")
        env_cfg.terminations = _restore_manager_cfg("terminations")
    if leftover_missing_curriculum(env_cfg):
        print("[WARN] Restoring leftover curriculum (CurriculumManager at gym.make).")
        env_cfg.curriculum = _restore_manager_cfg("curriculum")


def _reassert_missing_time_out(env_cfg) -> None:
    if not leftover_missing_time_out(env_cfg):
        return
    print("[WARN] Restoring leftover terminations.time_out (Stage 1 timeout-only).")
    terms = getattr(env_cfg, "terminations", None)
    if terms is None:
        env_cfg.terminations = type("Terminations", (), {})()
        terms = env_cfg.terminations
    try:
        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import TerminationTermCfg as DoneTerm

        _manager_set(terms, "time_out", DoneTerm(func=mdp.time_out))
    except ImportError:
        _manager_set(
            terms,
            "time_out",
            type("DoneTerm", (), {"func": type("F", (), {"__name__": "time_out"})()})(),
        )


def _scene_stub():
    return type(
        "Scene",
        (),
        {
            "num_envs": 1024,
            "env_spacing": 8.0,
            "replicate_physics": True,
            "filter_collisions": True,
            "clone_in_fabric": False,
            "robot": None,
            "terrain": None,
            "contact_forces": None,
            "height_scanner": None,
            "catcher": None,
        },
    )()


def _reassert_missing_scene(env_cfg) -> None:
    if not leftover_missing_scene(env_cfg):
        return
    print("[WARN] Restoring leftover scene (InteractiveScene at gym.make).")
    try:
        from h1_cfg.beamdojo_env_base import BeamDojoSceneCfg

        env_cfg.scene = BeamDojoSceneCfg(num_envs=1024, env_spacing=8.0)
    except ImportError:
        env_cfg.scene = _scene_stub()


def _reset_base_term(env_cfg):
    stage2 = env_cfg_stage(env_cfg) == 2
    pose = (
        {"x": (-0.2, 0.5), "y": (-0.08, 0.08), "yaw": (-0.3, 0.3)}
        if stage2
        else {"x": (-0.2, 0.2), "y": (-0.08, 0.08), "yaw": (-0.2, 0.2)}
    )
    vel = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    params = {"pose_range": pose, "velocity_range": vel}
    try:
        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import EventTermCfg as EventTerm

        return EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params=params)
    except ImportError:
        return type(
            "EventTerm",
            (),
            {
                "func": type("F", (), {"__name__": "reset_root_state_uniform"})(),
                "mode": "reset",
                "params": params,
            },
        )()


def _reset_joints_term():
    params = {"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}
    try:
        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import EventTermCfg as EventTerm

        return EventTerm(func=mdp.reset_joints_by_scale, mode="reset", params=params)
    except ImportError:
        return type(
            "EventTerm",
            (),
            {
                "func": type("F", (), {"__name__": "reset_joints_by_scale"})(),
                "mode": "reset",
                "params": params,
            },
        )()


def _reassert_missing_reset_events(env_cfg) -> None:
    """Leftover nulled reset_base / reset_robot_joints NaN PhysX at first reset."""
    if leftover_missing_events(env_cfg):
        print("[WARN] Restoring leftover events (EventManager at gym.make).")
        env_cfg.events = _restore_manager_cfg("events")
    events = getattr(env_cfg, "events", None)
    if leftover_missing_reset_base(env_cfg):
        print("[WARN] Restoring leftover events.reset_base (first reset before W&B).")
        _manager_set(events, "reset_base", _reset_base_term(env_cfg))
    if leftover_missing_reset_joints(env_cfg):
        print("[WARN] Restoring leftover events.reset_robot_joints (identity at first reset).")
        _manager_set(events, "reset_robot_joints", _reset_joints_term())


def _reassert_obs_history(env_cfg) -> None:
    """Leftover RNN ``history_length`` OOMs 1024 envs; leftover ``func=None`` dies at gym.make."""
    obs = getattr(env_cfg, "observations", None)
    for group_name, group in _iter_obs_groups(obs):
        if (isinstance(group, dict) or hasattr(group, "history_length")) and leftover_excess_obs_history(
            _manager_get(group, "history_length")
        ):
            print(
                f"[WARN] Restoring leftover observations.{group_name}.history_length="
                f"{_manager_get(group, 'history_length')!r} to 0."
            )
            _manager_set(group, "history_length", 0)
        for term_name, term in _iter_obs_terms(group):
            if leftover_missing_term_func(term):
                print(f"[WARN] Clearing leftover observations.{group_name}.{term_name} (func=None).")
                if isinstance(group, dict):
                    group[term_name] = None
                else:
                    setattr(group, term_name, None)
                continue
            if term is not None and (
                isinstance(term, dict) or hasattr(term, "scale")
            ) and leftover_invalid_obs_scale(_manager_get(term, "scale")):
                restored = {"base_lin_vel": 2.0, "base_ang_vel": 0.25, "projected_gravity": 1.0}.get(
                    term_name, 1.0
                )
                print(
                    f"[WARN] Restoring leftover observations.{group_name}.{term_name}.scale to {restored}."
                )
                _manager_set(term, "scale", restored)
            if term is not None and (
                isinstance(term, dict) or hasattr(term, "clip")
            ) and leftover_invalid_obs_clip(_manager_get(term, "clip")):
                print(f"[WARN] Clearing leftover observations.{group_name}.{term_name}.clip.")
                _manager_set(term, "clip", None)
            if term is not None and (
                isinstance(term, dict) or hasattr(term, "noise")
            ) and leftover_invalid_obs_noise(_manager_get(term, "noise")):
                print(f"[WARN] Clearing leftover observations.{group_name}.{term_name}.noise.")
                _manager_set(term, "noise", None)
            if term is not None and (
                isinstance(term, dict) or hasattr(term, "history_length")
            ) and leftover_excess_obs_history(_manager_get(term, "history_length")):
                print(
                    f"[WARN] Restoring leftover observations.{group_name}.{term_name}.history_length to 0."
                )
                _manager_set(term, "history_length", 0)


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
    present = _scene_entity_names(env_cfg)
    for group_name, group in _iter_obs_groups(obs):
        for term_name, term in _iter_obs_terms(group):
            if not leftover_missing_sensor_term(term, present):
                continue
            print(f"[WARN] Clearing leftover observations.{group_name}.{term_name} (missing sensor).")
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
    """Parent leftover ``env_spacing=2.5`` / ``None`` overlaps 1024 H1s at gym.make."""
    scene = getattr(env_cfg, "scene", None)
    if not leftover_invalid_env_spacing(scene):
        return
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


def _set_contact_field(contact, name: str, value) -> None:
    if contact is None:
        return
    if isinstance(contact, dict):
        contact[name] = value
        return
    setattr(contact, name, value)


def _reassert_contact_filters(env_cfg) -> None:
    """Leftover ANYmal filters / ``Robot/base`` die at ContactSensor init — before W&B."""
    contact = getattr(getattr(env_cfg, "scene", None), "contact_forces", None)
    if contact is None:
        return
    if leftover_contact_filter_prims(contact):
        print("[WARN] Clearing leftover contact_forces.filter_prim_paths_expr (PhysX filter count).")
        _set_contact_field(contact, "filter_prim_paths_expr", [])
    if leftover_track_contact_points(contact):
        print("[WARN] Disabling leftover contact_forces.track_contact_points (needs a leftover filter).")
        _set_contact_field(contact, "track_contact_points", False)
    if leftover_zero_contact_data_count(contact):
        print("[WARN] Restoring leftover contact_forces.max_contact_data_count_per_prim=4.")
        _set_contact_field(contact, "max_contact_data_count_per_prim", 4)
    path = contact.get("prim_path") if isinstance(contact, dict) else getattr(contact, "prim_path", None)
    if leftover_wrong_contact_prim(path):
        print(f"[WARN] Restoring leftover contact_forces.prim_path={path!r} to {{ENV_REGEX_NS}}/Robot/.*")
        _set_contact_field(contact, "prim_path", "{ENV_REGEX_NS}/Robot/.*")


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
    if leftover_missing_sim(env_cfg):
        print("[WARN] Restoring leftover sim (SimulationCfg at gym.make).")
        try:
            from isaaclab.sim import SimulationCfg

            env_cfg.sim = SimulationCfg(dt=0.005, device="cuda:0", render_interval=dec_n or 4)
            env_cfg.sim.wait_for_textures = False
            env_cfg.sim.use_fabric = True
        except ImportError:
            env_cfg.sim = type(
                "SimulationCfg",
                (),
                {
                    "dt": 0.005,
                    "device": "cuda:0",
                    "render_interval": dec_n or 4,
                    "wait_for_textures": False,
                    "use_fabric": True,
                },
            )()
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
    if leftover_wait_for_textures(sim):
        print("[WARN] Disabling leftover sim.wait_for_textures (Nucleus stall at gym.make).")
        if isinstance(sim, dict):
            sim["wait_for_textures"] = False
        else:
            sim.wait_for_textures = False
    if leftover_disabled_fabric(sim):
        print("[WARN] Enabling leftover sim.use_fabric (Stage 2 USD prim writes at first reset).")
        if isinstance(sim, dict):
            sim["use_fabric"] = True
        else:
            sim.use_fabric = True
    if leftover_disabled_contact_processing(sim):
        print("[WARN] Enabling leftover sim contact processing (ContactSensor at first reset).")
        if isinstance(sim, dict):
            sim["disable_contact_processing"] = False
        else:
            sim.disable_contact_processing = False
    if leftover_missing_physx(sim):
        print("[WARN] Restoring leftover sim.physx (SimulationContext at gym.make).")
        try:
            from isaaclab.sim import PhysxCfg

            restored = PhysxCfg()
        except ImportError:
            restored = type(
                "PhysxCfg",
                (),
                {
                    "gpu_max_rigid_contact_count": 2**23,
                    "gpu_max_rigid_patch_count": 16 * 2**15,
                    "gpu_found_lost_pairs_capacity": 2**21,
                },
            )()
        if isinstance(sim, dict):
            sim["physx"] = restored
        else:
            sim.physx = restored
    if leftover_invalid_gravity(sim):
        print("[WARN] Restoring leftover sim.gravity to (0.0, 0.0, -9.81).")
        if isinstance(sim, dict):
            sim["gravity"] = (0.0, 0.0, -9.81)
        else:
            sim.gravity = (0.0, 0.0, -9.81)


def _world_catcher_stub():
    return type("AssetBaseCfg", (), {"prim_path": "/World/catcher", "collision_group": -1})()


def _task_beam_stub():
    return type("RigidObjectCfg", (), {"prim_path": "{ENV_REGEX_NS}/TaskBeam"})()


def _stone_stub(index: int):
    return type("RigidObjectCfg", (), {"prim_path": f"{{ENV_REGEX_NS}}/TaskStone{index}"})()


def leftover_unusable_task_rigid(asset) -> bool:
    return leftover_asset_base_rigid(asset) or leftover_disabled_collision_asset(asset)


def _reassert_stage_task_beam(env_cfg) -> None:
    """Leftover Stage 2 AssetBase / visual-only beam falls through at wrapper reset."""
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    beam = getattr(scene, "task_beam", None)
    if env_cfg_uses_stones(env_cfg):
        if beam is not None:
            print("[WARN] Clearing leftover scene.task_beam (Stage 2 stones has no beam).")
            scene.task_beam = None
        return
    if env_cfg_stage(env_cfg) != 2:
        return
    if beam is not None and not leftover_unusable_task_rigid(beam):
        return
    print("[WARN] Restoring leftover Stage 2 task_beam to cloned colliding RigidObjectCfg.")
    try:
        from h1_cfg.scene_props import task_beam_cfg

        scene.task_beam = task_beam_cfg(collision=True)
    except ImportError:
        scene.task_beam = _task_beam_stub()


def _reassert_stage_stones(env_cfg) -> None:
    """Leftover missing / AssetBase / visual stones fall through at first reset."""
    if not env_cfg_uses_stones(env_cfg):
        return
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    for index in range(BEAMDOJO_STONE_COUNT):
        name = f"task_stone_{index}"
        stone = getattr(scene, name, None)
        if stone is not None and not leftover_unusable_task_rigid(stone):
            continue
        print(f"[WARN] Restoring leftover {name} to cloned colliding RigidObjectCfg.")
        try:
            from h1_cfg.scene_props import stone_cfg

            setattr(scene, name, stone_cfg(index, collision=True))
        except ImportError:
            setattr(scene, name, _stone_stub(index))


def _reassert_init_root_rot(env_cfg) -> None:
    """Leftover ``rot=(0,0,0,0)`` NaNs PhysX at wrapper reset — before W&B."""
    state = _robot_init_state(env_cfg)
    if state is None:
        return
    rot = state.get("rot") if isinstance(state, dict) else getattr(state, "rot", None)
    if not leftover_invalid_root_rot(rot):
        return
    print(f"[WARN] Restoring leftover robot.init_state.rot={rot!r} to identity.")
    if isinstance(state, dict):
        state["rot"] = (1.0, 0.0, 0.0, 0.0)
    elif hasattr(state, "rot"):
        state.rot = (1.0, 0.0, 0.0, 0.0)


def _clear_nucleus_visual_material(asset, label: str) -> None:
    spawn = _asset_spawn(asset)
    target = spawn if spawn is not None else asset
    if target is None:
        return
    mat = target.get("visual_material") if isinstance(target, dict) else getattr(target, "visual_material", None)
    if not leftover_nucleus_visual_material(mat):
        return
    print(f"[WARN] Clearing leftover Nucleus {label} visual_material so gym.make does not hang.")
    if isinstance(target, dict):
        target["visual_material"] = None
    elif hasattr(target, "visual_material"):
        target.visual_material = None


def _reassert_nucleus_visual_materials(env_cfg) -> None:
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    _clear_nucleus_visual_material(getattr(scene, "robot", None), "robot.spawn")
    _clear_nucleus_visual_material(getattr(scene, "task_beam", None), "task_beam.spawn")
    for index in range(BEAMDOJO_STONE_COUNT):
        _clear_nucleus_visual_material(getattr(scene, f"task_stone_{index}", None), f"task_stone_{index}.spawn")


def _reassert_replicate_physics(env_cfg) -> None:
    """Leftover ``replicate_physics=False`` hangs or OOMs 1024-env GPU clone."""
    scene = getattr(env_cfg, "scene", None)
    if not leftover_disabled_replicate_physics(scene):
        return
    print("[WARN] Enabling leftover scene.replicate_physics (cloned 1024-env GPU PhysX).")
    scene.replicate_physics = True


def _reassert_drop_scene_cameras(env_cfg) -> None:
    """PLAY leftover tiled cameras × 1024 envs OOM A10 24GB at gym.make."""
    scene = getattr(env_cfg, "scene", None)
    for name in leftover_scene_camera_fields(scene):
        print(f"[WARN] Clearing leftover scene.{name} (1024-env train camera OOM before W&B).")
        setattr(scene, name, None)


def _reassert_filter_collisions(env_cfg) -> None:
    scene = getattr(env_cfg, "scene", None)
    if not leftover_unfiltered_collisions(scene):
        return
    print("[WARN] Enabling leftover scene.filter_collisions (cloned 1024-env neighbor hits).")
    scene.filter_collisions = True


def _reassert_clone_in_fabric(env_cfg) -> None:
    scene = getattr(env_cfg, "scene", None)
    if not leftover_clone_in_fabric(scene):
        return
    print("[WARN] Disabling leftover scene.clone_in_fabric (USD Stage 2 prim writes at first reset).")
    scene.clone_in_fabric = False


def _reassert_stage_in_memory(env_cfg) -> None:
    sim = getattr(env_cfg, "sim", None)
    if not leftover_stage_in_memory(sim):
        return
    print("[WARN] Disabling leftover sim.create_stage_in_memory (Isaac 2.3.2 gym.make).")
    sim.create_stage_in_memory = False


def _plane_terrain_stub():
    return type("TerrainImporterCfg", (), {"prim_path": "/World/ground", "terrain_type": "plane"})()


def _contact_forces_stub():
    return type(
        "ContactSensorCfg",
        (),
        {
            "prim_path": "{ENV_REGEX_NS}/Robot/.*",
            "history_length": 3,
            "track_air_time": True,
        },
    )()


def _reassert_missing_scene_assets(env_cfg) -> None:
    """Hydra leftover can null terrain / contact_forces; gym.make KeyErrors before W&B."""
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    if leftover_missing_terrain(scene):
        print("[WARN] Restoring leftover scene.terrain to a plane.")
        try:
            from h1_cfg.beamdojo_common import flat_plane_terrain

            flat_plane_terrain(env_cfg)
        except ImportError:
            scene.terrain = _plane_terrain_stub()
    if leftover_missing_contact_forces(scene):
        print("[WARN] Restoring leftover scene.contact_forces (feet_air_time at first reset).")
        try:
            from isaaclab.sensors import ContactSensorCfg

            scene.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=3,
                track_air_time=True,
                update_period=0.0,
            )
        except ImportError:
            scene.contact_forces = _contact_forces_stub()


def _reassert_num_envs(env_cfg) -> None:
    scene = getattr(env_cfg, "scene", None)
    if leftover_zero_num_envs(scene):
        print("[WARN] Restoring leftover scene.num_envs=1024.")
        scene.num_envs = 1024
        return
    if leftover_excess_num_envs(scene):
        print("[WARN] Clamping leftover scene.num_envs to 1024 (parent 4096 OOMs A10 24GB).")
        scene.num_envs = 1024


def _enable_spawn_collision(asset) -> None:
    spawn = _asset_spawn(asset)
    if spawn is None:
        return
    props = spawn.get("collision_props") if isinstance(spawn, dict) else getattr(spawn, "collision_props", None)
    if isinstance(props, dict):
        props["collision_enabled"] = True
    elif props is not None and hasattr(props, "collision_enabled"):
        props.collision_enabled = True
    elif isinstance(spawn, dict):
        spawn["collision_enabled"] = True
    elif hasattr(spawn, "collision_enabled"):
        spawn.collision_enabled = True


def _set_spawn_prop(spawn, props_name: str, field: str, value) -> None:
    if spawn is None:
        return
    props = spawn.get(props_name) if isinstance(spawn, dict) else getattr(spawn, props_name, None)
    if isinstance(props, dict):
        props[field] = value
        return
    if props is not None and hasattr(props, field):
        setattr(props, field, value)


def _reassert_robot_collision(env_cfg) -> None:
    """Leftover visual-only robot (``collision_enabled=False``) falls through at reset."""
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if not leftover_disabled_collision_asset(robot):
        return
    print("[WARN] Enabling leftover robot collision (visual-only robot falls through).")
    _enable_spawn_collision(robot)


def _reassert_robot_dynamics(env_cfg) -> None:
    """Catcher cuboid leftover on the robot welds/floats it; self-collisions die at reset."""
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn = _asset_spawn(robot)
    if leftover_kinematic_robot(robot):
        print("[WARN] Disabling leftover robot kinematic_enabled (catcher cuboid props on H1/G1).")
        _set_spawn_prop(spawn, "rigid_props", "kinematic_enabled", False)
    if leftover_disabled_gravity_robot(robot):
        print("[WARN] Enabling leftover robot gravity (catcher disable_gravity on H1/G1).")
        _set_spawn_prop(spawn, "rigid_props", "disable_gravity", False)
    if leftover_fixed_root_robot(robot):
        print("[WARN] Clearing leftover robot fix_root_link (welded pelvis).")
        _set_spawn_prop(spawn, "articulation_props", "fix_root_link", False)
    if leftover_self_collisions_robot(robot):
        print("[WARN] Disabling leftover robot enabled_self_collisions (PhysX at first reset).")
        _set_spawn_prop(spawn, "articulation_props", "enabled_self_collisions", False)


def _reassert_missing_scene_entity_terms(env_cfg) -> None:
    """Leftover reward/event/done terms still look up height_scanner after we null it."""
    present = _scene_entity_names(env_cfg)
    for group_name in ("rewards", "events", "terminations"):
        group = getattr(env_cfg, group_name, None)
        if group is None:
            continue
        for term_name in _public_field_names(group):
            term = getattr(group, term_name, None)
            if leftover_missing_term_func(term):
                print(f"[WARN] Clearing leftover {group_name}.{term_name} (func=None).")
                setattr(group, term_name, None)
                continue
            if not leftover_missing_scene_entity_term(term, present):
                continue
            print(f"[WARN] Clearing leftover {group_name}.{term_name} (missing scene entity).")
            setattr(group, term_name, None)


def _reassert_stage_catcher(env_cfg) -> None:
    """Leftover Stage 1 catcher / Stage 2 RigidObject catcher dies at first reset."""
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    stage = env_cfg_stage(env_cfg)
    catcher = getattr(scene, "catcher", None)
    if stage == 1 and catcher is not None:
        print("[WARN] Clearing leftover scene.catcher (Stage 1 is plane / timeout-only).")
        scene.catcher = None
        return
    if stage != 2:
        return
    if leftover_rigid_catcher(catcher) or catcher is None:
        print("[WARN] Restoring leftover Stage 2 catcher to world AssetBaseCfg.")
        try:
            from h1_cfg.scene_props import catcher_cfg

            scene.catcher = catcher_cfg()
        except ImportError:
            scene.catcher = _world_catcher_stub()


def _reassert_clone_prim_paths(env_cfg) -> None:
    """Leftover ``/World/Robot`` (no ENV_REGEX_NS) cannot clone 1024 envs."""
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    robot = getattr(scene, "robot", None)
    path = getattr(robot, "prim_path", None) if robot is not None else None
    if robot is not None and leftover_uncloned_prim_path(path):
        print(f"[WARN] Restoring leftover robot.prim_path={path!r} to {{ENV_REGEX_NS}}/Robot.")
        if isinstance(robot, dict):
            robot["prim_path"] = "{ENV_REGEX_NS}/Robot"
        else:
            robot.prim_path = "{ENV_REGEX_NS}/Robot"
    contact = getattr(scene, "contact_forces", None)
    cpath = getattr(contact, "prim_path", None) if contact is not None else None
    if contact is not None and leftover_uncloned_prim_path(cpath):
        print("[WARN] Restoring leftover contact_forces.prim_path to {ENV_REGEX_NS}/Robot/.*")
        if isinstance(contact, dict):
            contact["prim_path"] = "{ENV_REGEX_NS}/Robot/.*"
        else:
            contact.prim_path = "{ENV_REGEX_NS}/Robot/.*"
    beam = getattr(scene, "task_beam", None)
    bpath = getattr(beam, "prim_path", None) if beam is not None else None
    if beam is not None and leftover_uncloned_prim_path(bpath):
        print("[WARN] Restoring leftover task_beam.prim_path to {ENV_REGEX_NS}/TaskBeam.")
        if isinstance(beam, dict):
            beam["prim_path"] = "{ENV_REGEX_NS}/TaskBeam"
        else:
            beam.prim_path = "{ENV_REGEX_NS}/TaskBeam"


def _reassert_stage1_timeout_only(env_cfg) -> None:
    """Stage 1 is timeout-only. Leftover parent base_contact fires on the plane."""
    if env_cfg_stage(env_cfg) == 2:
        return
    terms = getattr(env_cfg, "terminations", None)
    if terms is None:
        return
    for name in ("base_contact", "base_height", "base_orientation"):
        if getattr(terms, name, None) is not None:
            print(f"[WARN] Clearing leftover terminations.{name} (Stage 1 is timeout-only).")
            setattr(terms, name, None)


def _set_action_joint_names(joint_pos, names) -> None:
    if joint_pos is None:
        return
    if isinstance(joint_pos, dict):
        joint_pos["joint_names"] = names
    elif hasattr(joint_pos, "joint_names"):
        joint_pos.joint_names = names


def _reassert_g1_action_joints(env_cfg) -> None:
    """Parent leftover ``joint_names=[".*"]`` / H1 regexes put G1 arms back in the action."""
    if not _treat_as_g1(env_cfg):
        return
    joint_pos, names = _action_joint_names(env_cfg)
    if joint_pos is None:
        return
    if not leftover_all_joints(names) and not leftover_wrong_action_joint_names(env_cfg):
        return
    try:
        from h1_cfg.robot_spec import G1
    except ImportError as exc:
        print(f"[WARN] G1 action-joint reassert skipped ({type(exc).__name__}: {exc})")
        return
    print("[WARN] Restoring leftover G1 action joints off parent '.*' (paper: 12 lower-body).")
    _set_action_joint_names(joint_pos, list(G1.action_joints))


def _reassert_action_joint_names(env_cfg) -> None:
    """H1 leftover ANYmal ``.*HAA`` / G1 ``*_joint`` names resolve 0 actions at gym.make."""
    _reassert_g1_action_joints(env_cfg)
    if _treat_as_g1(env_cfg) or not leftover_wrong_action_joint_names(env_cfg):
        return
    joint_pos, _ = _action_joint_names(env_cfg)
    print("[WARN] Restoring leftover H1 action joints off ANYmal/G1 names.")
    _set_action_joint_names(joint_pos, [".*"])


def _reassert_event_joint_names(env_cfg) -> None:
    """Leftover ``events.actuator_gains`` ANYmal joints ValueError at gym.make."""
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    for term_name in _public_field_names(events):
        term = getattr(events, term_name, None)
        if not leftover_wrong_event_joint_names(term, env_cfg):
            continue
        print(f"[WARN] Restoring leftover events.{term_name} joint_names to '.*'.")
        _set_entity_joint_names(_term_entity_cfg(term, "asset_cfg"), [".*"])


def _default_event_mode(name: str) -> str:
    lower = str(name).lower()
    if lower.startswith("init") or "startup" in lower:
        return "startup"
    return "reset"


def _reassert_event_modes(env_cfg) -> None:
    """Leftover ``mode=None`` / invalid interval / min_step terms die in EventManager at gym.make."""
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    for term_name in _public_field_names(events):
        term = getattr(events, term_name, None)
        if leftover_invalid_interval_event(term):
            print(f"[WARN] Clearing leftover events.{term_name} (interval_range_s invalid at gym.make).")
            setattr(events, term_name, None)
            continue
        if leftover_invalid_event_mode(term):
            mode = _default_event_mode(term_name)
            print(f"[WARN] Restoring leftover events.{term_name}.mode to {mode}.")
            _manager_set(term, "mode", mode)
        if leftover_invalid_min_step_count(term):
            print(f"[WARN] Restoring leftover events.{term_name}.min_step_count_between_reset to 0.")
            _manager_set(term, "min_step_count_between_reset", 0)


def _reassert_term_params(env_cfg) -> None:
    """Leftover ``params=None`` AttributeErrors ``params.keys()`` in ManagerBase at gym.make."""
    for group_name in ("rewards", "events", "terminations", "curriculum"):
        group = getattr(env_cfg, group_name, None)
        if group is None:
            continue
        for term_name in _public_field_names(group):
            term = getattr(group, term_name, None)
            if not leftover_invalid_term_params(term):
                continue
            if group_name == "events" and term_name == "reset_base":
                print("[WARN] Restoring leftover events.reset_base params (first reset before W&B).")
                _manager_set(group, term_name, _reset_base_term(env_cfg))
                continue
            if group_name == "events" and term_name in ("reset_robot_joints", "reset_joints"):
                print("[WARN] Restoring leftover events.reset_robot_joints params (identity at first reset).")
                _manager_set(group, term_name, _reset_joints_term())
                continue
            print(f"[WARN] Clearing leftover {group_name}.{term_name} (params=None at gym.make).")
            setattr(group, term_name, None)


def _default_reward_weight(name: str, env_cfg) -> float:
    g1 = _treat_as_g1(env_cfg)
    weights = {
        "lin_vel_z_l2": 0.0 if g1 else -2.0,
        "ang_vel_xy_l2": -0.05,
        "flat_orientation_l2": -1.0,
        "dof_torques_l2": -1.5e-7 if g1 else 0.0,
        "action_rate_l2": -0.005,
        "dof_acc_l2": -1.25e-7,
        "base_height_penalty": -10.0,
        "termination_penalty": -200.0,
        "track_lin_vel_xy_exp": 1.0,
        "track_ang_vel_z_exp": 2.0 if g1 else 1.0,
        "feet_air_time": 0.25,
        "feet_slide": -0.1 if g1 else -0.25,
        "dof_pos_limits": -1.0,
        "joint_deviation_hip": -0.1 if g1 else -0.2,
        "joint_deviation_arms": -0.1 if g1 else -0.2,
        "joint_deviation_torso": -0.1,
        "foothold_penalty": 1.0,
        "joint_deviation_fingers": -0.05,
    }
    return float(weights.get(name, 0.0))


def _reassert_reward_weights(env_cfg) -> None:
    """Leftover ``weight=None`` TypeErrors RewardManager at gym.make — before W&B."""
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return
    for term_name in _public_field_names(rewards):
        term = getattr(rewards, term_name, None)
        if not leftover_invalid_reward_weight(term):
            continue
        restored = _default_reward_weight(term_name, env_cfg)
        print(f"[WARN] Restoring leftover rewards.{term_name}.weight to {restored}.")
        _manager_set(term, "weight", restored)


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
    if env_cfg_stage(env_cfg) != 2:
        return
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
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
            if env_cfg_stage(env_cfg) == 2:
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
    _reassert_missing_scene(env_cfg)
    scene = getattr(env_cfg, "scene", None)
    if scene is not None and getattr(scene, "height_scanner", None) is not None:
        print(
            "[WARN] Clearing leftover scene.height_scanner (ANYmal RayCaster). "
            "Policy scan is task_height_scan."
        )
        scene.height_scanner = None

    _reassert_drop_scene_cameras(env_cfg)
    _reassert_filter_collisions(env_cfg)
    _reassert_num_envs(env_cfg)
    _reassert_clone_in_fabric(env_cfg)
    _reassert_stage_in_memory(env_cfg)
    _reassert_missing_scene_assets(env_cfg)
    scene = getattr(env_cfg, "scene", None)

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

    _reassert_missing_policy_obs(env_cfg)
    _reassert_obs_height_scan(env_cfg)
    _reassert_obs_history(env_cfg)
    _reassert_velocity_command(env_cfg)
    _reassert_missing_managers(env_cfg)

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

    _reassert_stage_catcher(env_cfg)
    _reassert_stage_task_beam(env_cfg)
    _reassert_stage_stones(env_cfg)
    _reassert_replicate_physics(env_cfg)
    _reassert_clone_prim_paths(env_cfg)
    _reassert_unitree_robot(env_cfg)
    _reassert_robot_collision(env_cfg)
    _reassert_robot_dynamics(env_cfg)
    _drop_leftover_actuators(env_cfg)
    _reassert_init_joint_state(env_cfg)
    _reassert_init_root_rot(env_cfg)
    _reassert_nucleus_visual_materials(env_cfg)
    _reassert_missing_joint_pos_action(env_cfg)
    _reassert_action_joint_maps(env_cfg)
    _reassert_contact_sensors(env_cfg)
    _reassert_velocity_ranges(env_cfg)
    _reassert_anymal_body_names(env_cfg)
    _reassert_missing_scene_entity_terms(env_cfg)
    _reassert_action_joint_names(env_cfg)
    _reassert_g1_joint_fullmatch(env_cfg)
    _drop_h1_leftover_fingers(env_cfg)
    _reassert_h1_joint_fullmatch(env_cfg)
    _reassert_event_modes(env_cfg)
    _reassert_term_params(env_cfg)
    _reassert_reward_weights(env_cfg)
    _reassert_missing_reset_events(env_cfg)
    _reassert_official_reset_events(env_cfg)
    _reassert_event_joint_names(env_cfg)
    _reassert_stage2_reset_on_beam(env_cfg)
    _reassert_stage1_timeout_only(env_cfg)
    _reassert_missing_time_out(env_cfg)
    _reassert_env_spacing(env_cfg)
    _reassert_contact_history(env_cfg)
    _reassert_contact_filters(env_cfg)
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
