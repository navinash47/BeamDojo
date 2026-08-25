"""Shared Stage 1 / Stage 2 env wiring (dual-terrain, rewards, dones)."""

from __future__ import annotations

import math

from isaaclab import sim as sim_utils
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from h1_cfg import mdp as bd_mdp
from h1_cfg.robot_spec import G1_FINGER_JOINTS, H1, RobotSpec
from h1_cfg.scene_props import (
    BEAM_CENTER_Z,
    BEAM_LENGTH,
    BEAM_THICKNESS,
    BEAM_WIDTH_EASY,
    BEAM_WIDTH_HARD,
    STONE_COUNT,
    add_stepping_stones,
    catcher_cfg,
    task_beam_cfg,
)

def spawn_robot(cfg, spec: RobotSpec) -> None:
    """Spawn the official *minimal* USD (fewer collision meshes; A10 1024-env fit)."""
    if spec.name == "g1":
        from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG as ROBOT_CFG
    else:
        try:
            from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG as ROBOT_CFG
        except ImportError:
            from isaaclab_assets.robots.unitree import H1_CFG as ROBOT_CFG

    cfg.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    beam_top = BEAM_CENTER_Z + BEAM_THICKNESS * 0.5
    cfg.scene.robot.init_state.pos = (0.0, 0.0, spec.pelvis_z + beam_top)


def flat_plane_terrain(cfg) -> None:
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # Parent ``LocomotionVelocityRoughEnvCfg.__post_init__`` copies the *generator*
    # material onto ``sim.physics_material`` before this replace.
    cfg.sim.physics_material = cfg.scene.terrain.physics_material


def apply_sensors(cfg, spec: RobotSpec) -> None:
    # Policy scan is the task heightfield (task_height_scan), not rays.
    # Parent ANYmal RayCaster (~256 rays × 1024 envs) is unused and an A10 hitch.
    _ = spec
    cfg.scene.height_scanner = None
    cfg.scene.contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )
    dt = float(getattr(cfg.sim, "dt", 0.005) or 0.005)
    cfg.scene.contact_forces.update_period = dt


def apply_paper_dr(cfg, spec: RobotSpec) -> None:
    """Appendix VI-C / Table IX. Update existing Isaac Lab terms; skip missing names."""
    torso = spec.torso_body
    if getattr(cfg.events, "add_base_mass", None) is not None:
        cfg.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        cfg.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=torso)
    if getattr(cfg.events, "base_com", None) is not None:
        cfg.events.base_com.params["com_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
        }
        cfg.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=torso)
    # Interval pushes are eval-only in the paper, not Table IX training DR.
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    if getattr(cfg.events, "physics_material", None) is not None:
        cfg.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        cfg.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
        cfg.events.physics_material.params["restitution_range"] = (0.0, 1.0)
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.params["stiffness_distribution_params"] = (0.85, 1.15)
        cfg.events.actuator_gains.params["damping_distribution_params"] = (0.85, 1.15)
        cfg.events.actuator_gains.params["operation"] = "scale"

    policy = cfg.observations.policy
    for name, lo, hi in (
        ("base_ang_vel", -0.5, 0.5),
        ("joint_pos", -0.05, 0.05),
        ("joint_vel", -2.0, 2.0),
        ("projected_gravity", -0.05, 0.05),
    ):
        term = getattr(policy, name, None)
        if term is not None:
            term.noise = Unoise(n_min=lo, n_max=hi)


def _zero_root_reset_velocity(cfg) -> None:
    """Official H1/G1 locomotion: zero root twist at reset (parent ANYmal is ±0.5)."""
    reset_base = getattr(cfg.events, "reset_base", None)
    if reset_base is None:
        return
    velocity_range = dict(reset_base.params.get("velocity_range") or {})
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        velocity_range[key] = (0.0, 0.0)
    reset_base.params["velocity_range"] = velocity_range


def apply_shared_locomotion(cfg, spec: RobotSpec, *, stage: int) -> None:
    apply_paper_dr(cfg, spec)

    # Official H1/G1: identity joint scale at reset (parent ANYmal is 0.5–1.5).
    if getattr(cfg.events, "reset_robot_joints", None) is not None:
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    _zero_root_reset_velocity(cfg)

    cfg.rewards.lin_vel_z_l2.weight = -2.0 if spec.name == "h1" else 0.0
    cfg.rewards.ang_vel_xy_l2.weight = -0.05
    cfg.rewards.flat_orientation_l2.weight = -1.0
    cfg.rewards.dof_torques_l2.weight = 0.0
    cfg.rewards.action_rate_l2.weight = -0.005
    cfg.rewards.dof_acc_l2.weight = -1.25e-7
    cfg.rewards.undesired_contacts = None

    pelvis_target = spec.pelvis_z + (BEAM_CENTER_Z + BEAM_THICKNESS * 0.5)
    cfg.rewards.base_height_penalty = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": pelvis_target, "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    cfg.rewards.track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    cfg.rewards.track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    cfg.rewards.feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[spec.feet_body]),
            "threshold": 0.4,
        },
    )
    cfg.rewards.feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[spec.feet_body]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[spec.feet_body]),
        },
    )
    cfg.rewards.dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=spec.ankle_joints)},
    )
    cfg.rewards.joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2 if spec.name == "h1" else -0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[spec.hip_yaw, spec.hip_roll])},
    )
    cfg.rewards.joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2 if spec.name == "h1" else -0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=spec.arm_joints)},
    )
    cfg.rewards.joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=spec.torso_joint)},
    )
    cfg.rewards.foothold_penalty = RewTerm(
        func=bd_mdp.foothold_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[spec.feet_body]),
            "num_samples": 15,
            "depth_threshold": -0.1,
        },
    )
    if spec.name == "g1":
        cfg.rewards.feet_slide.weight = -0.1
        cfg.rewards.track_ang_vel_z_exp.weight = 2.0
        cfg.rewards.dof_torques_l2.weight = -1.5e-7
        acc = cfg.rewards.dof_acc_l2
        if acc is not None:
            if not acc.params:
                acc.params = {}
            acc.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
            )
        torque = cfg.rewards.dof_torques_l2
        if torque is not None:
            if not torque.params:
                torque.params = {}
            torque.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
            )
        cfg.rewards.joint_deviation_fingers = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_FINGER_JOINTS)},
        )

    cfg.observations.policy.base_lin_vel.scale = 2.0
    cfg.observations.policy.base_ang_vel.scale = 0.25
    cfg.observations.policy.projected_gravity.scale = 1.0
    cfg.observations.policy.height_scan = ObsTerm(
        func=bd_mdp.task_height_scan,
        params={"sensor_cfg": SceneEntityCfg("robot"), "offset": 0.5, "grid_n": 15, "extent": 1.4},
        clip=(-1.0, 1.0),
    )

    cfg.episode_length_s = 20.0
    cfg.decimation = 4
    cfg.sim.dt = 0.005
    cfg.curriculum.terrain_levels = None

    if spec.action_joints is not None and hasattr(cfg.actions, "joint_pos"):
        cfg.actions.joint_pos.joint_names = spec.action_joints
    if hasattr(cfg.commands, "base_velocity"):
        # 1024-env command arrows are a common Isaac Lab headless hitch.
        cfg.commands.base_velocity.debug_vis = False
        # Parent ANYmal heading target yaws the robot off a 20 cm imagined/real beam.
        cfg.commands.base_velocity.heading_command = False
        # ActorCritic 3.0.1 asserts each obs group is 2D ([N, dim]).
        if hasattr(cfg.observations, "policy"):
            cfg.observations.policy.concatenate_terms = True
        # Parent sky uses a Nucleus HDR; a miss hangs gym.make before wandb.init.
        sky = getattr(cfg.scene, "sky_light", None)
        spawn = getattr(sky, "spawn", None)
        if spawn is not None and getattr(spawn, "texture_file", None):
            spawn.texture_file = None

    start_w = BEAM_WIDTH_HARD if stage == 1 else BEAM_WIDTH_EASY
    cfg.events.init_beamdojo = EventTerm(
        func=bd_mdp.init_beamdojo_state,
        mode="startup",
        params={"width": start_w, "length": BEAM_LENGTH, "terrain": "beam"},
    )
    cfg.events.reset_beamdojo = EventTerm(
        func=bd_mdp.reset_beamdojo_noise,
        mode="reset",
        params={"vertical_bias_std": 0.03},
    )


def apply_physx_gpu_capacity(cfg, *, stones: bool = False) -> None:
    """Raise dual-terrain PhysX GPU buffers without OOMing A10 24GB.

    Kept as a ``def`` in this file so older ``after_relaunch.sh`` preflights
    still pass after git pull. Floors live in ``h1_cfg/physx_gpu.py``.
    """
    from h1_cfg.physx_gpu import apply_physx_gpu_capacity as _apply

    _apply(cfg, stones=stones)


def apply_stage1(cfg, spec: RobotSpec = H1) -> None:
    cfg.scene.num_envs = 1024
    cfg.scene.env_spacing = 8.0
    spawn_robot(cfg, spec)
    # Stage 1 physics is the plane; robot pelvis uses standing height (beam is visual only).
    cfg.scene.robot.init_state.pos = (0.0, 0.0, spec.pelvis_z)
    flat_plane_terrain(cfg)
    apply_sensors(cfg, spec)
    cfg.scene.task_beam = task_beam_cfg(collision=False, width=BEAM_WIDTH_HARD, center_z=0.02)
    apply_shared_locomotion(cfg, spec, stage=1)
    apply_physx_gpu_capacity(cfg, stones=False)

    cfg.rewards.base_height_penalty.params["target_height"] = spec.pelvis_z

    cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
    for name in ("base_height", "base_orientation", "base_contact"):
        if hasattr(cfg.terminations, name):
            setattr(cfg.terminations, name, None)

    cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.5


def apply_stage2(cfg, spec: RobotSpec = H1, *, stones: bool = False) -> None:
    cfg.scene.num_envs = 1024
    cfg.scene.env_spacing = 8.0
    spawn_robot(cfg, spec)
    flat_plane_terrain(cfg)
    apply_sensors(cfg, spec)
    if stones:
        cfg.scene.task_beam = None
        add_stepping_stones(cfg.scene, count=STONE_COUNT, collision=True)
        terrain = "stones"
        start_w = 0.20
    else:
        # Easy physical width; task-map curriculum tightens dones/rewards to 20 cm.
        cfg.scene.task_beam = task_beam_cfg(collision=True, width=BEAM_WIDTH_EASY)
        terrain = "beam"
        start_w = BEAM_WIDTH_EASY
    cfg.scene.catcher = catcher_cfg()
    apply_shared_locomotion(cfg, spec, stage=2)
    apply_physx_gpu_capacity(cfg, stones=stones)
    cfg.events.init_beamdojo.params["terrain"] = terrain
    cfg.events.init_beamdojo.params["width"] = start_w
    cfg.events.disable_ground = EventTerm(
        func=bd_mdp.disable_ground_collision,
        mode="startup",
        params={"prim_paths": ("/World/ground", "/World/defaultGroundPlane")},
    )
    # Spawn on the beam, not on the collision-disabled plane beside it.
    pose_range = dict(cfg.events.reset_base.params.get("pose_range") or {})
    pose_range["x"] = (-0.2, 0.5)
    pose_range["y"] = (-0.08, 0.08)
    pose_range["yaw"] = (-0.3, 0.3)
    cfg.events.reset_base.params["pose_range"] = pose_range

    cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cfg.terminations.base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.40, "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.terminations.base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(45.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    if hasattr(cfg.terminations, "base_contact") and cfg.terminations.base_contact is not None:
        cfg.terminations.base_contact.params["sensor_cfg"].body_names = spec.torso_body
    cfg.terminations.off_terrain = DoneTerm(
        func=bd_mdp.off_task_terrain,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[spec.feet_body]),
            "min_off_samples": 8.0,
        },
    )

    cfg.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.8)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
    cfg.commands.base_velocity.rel_standing_envs = 0.1

    if not stones:
        cfg.curriculum.beam_width = CurrTerm(
            func=bd_mdp.tighten_beam_width,
            params={
                "start_width": BEAM_WIDTH_EASY,
                "end_width": BEAM_WIDTH_HARD,
                "horizon_steps": 240_000,
            },
        )


def apply_play(cfg) -> None:
    cfg.scene.num_envs = 64
    cfg.scene.env_spacing = 6.0
    cfg.episode_length_s = 40.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    pose_range = dict(cfg.events.reset_base.params.get("pose_range") or {})
    pose_range["x"] = (-0.2, 0.2)
    pose_range["y"] = (-0.08, 0.08)
    pose_range["yaw"] = (-0.2, 0.2)
    cfg.events.reset_base.params["pose_range"] = pose_range
    _zero_root_reset_velocity(cfg)
