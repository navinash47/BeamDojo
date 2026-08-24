"""Isaac Lab MDP terms: dual-terrain scan, foothold (eq. 2), Stage 2 dones, curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from beamdojo_mdp.curriculum import physical_beam_scale_y, should_update_physical_width, task_beam_prim_paths
from beamdojo_mdp.heightfield import (
    BEAM_OFF_Z,
    BEAM_ON_Z,
    beam_surface_z,
    foothold_off_count,
    stone_surface_z,
    yaw_from_quat_wxyz,
    yaw_grid_xy,
)
from beamdojo_mdp.elevation_noise import dilate_on_cells, maybe_repeat_map, tilt_height_grid

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _raw(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def init_beamdojo_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    width: float = 0.20,
    length: float = 8.0,
    terrain: str = "beam",
    stone_size: float = 0.20,
    gap: float = 0.10,
    on_z: float = BEAM_ON_Z,
    off_z: float = BEAM_OFF_Z,
    scan_noise_std: float = 0.03,
) -> None:
    """Allocate per-env task-map buffers (startup)."""
    env = _raw(env)
    n = env.num_envs
    dev = env.device
    env.beamdojo_width = torch.full((n,), float(width), device=dev)
    env.beamdojo_length = float(length)
    env.beamdojo_terrain = terrain
    env.beamdojo_stone_size = float(stone_size)
    env.beamdojo_gap = float(gap)
    env.beamdojo_on_z = float(on_z)
    env.beamdojo_off_z = float(off_z)
    env.beamdojo_scan_noise_std = float(scan_noise_std)
    env.beamdojo_foothold_step = torch.zeros(n, device=dev)
    env.beamdojo_width0 = float(width)
    env.beamdojo_map_yaw = torch.zeros(n, device=dev)
    env.beamdojo_map_hx = torch.zeros(n, device=dev)
    env.beamdojo_map_hy = torch.zeros(n, device=dev)
    env.beamdojo_extend = torch.zeros(n, dtype=torch.bool, device=dev)
    env.beamdojo_last_scan = None


def reset_beamdojo_noise(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    vertical_bias_std: float = 0.03,
) -> None:
    """Paper-style elevation bias resampled on reset."""
    env = _raw(env)
    if not hasattr(env, "beamdojo_width"):
        return
    ids = env_ids if env_ids is not None else slice(None)
    n = env.num_envs if env_ids is None else len(env_ids)
    env.beamdojo_z_bias = getattr(env, "beamdojo_z_bias", torch.zeros(env.num_envs, device=env.device))
    env.beamdojo_z_bias[ids] = vertical_bias_std * torch.randn(n, device=env.device)
    env.beamdojo_map_yaw = getattr(env, "beamdojo_map_yaw", torch.zeros(env.num_envs, device=env.device))
    env.beamdojo_map_hx = getattr(env, "beamdojo_map_hx", torch.zeros(env.num_envs, device=env.device))
    env.beamdojo_map_hy = getattr(env, "beamdojo_map_hy", torch.zeros(env.num_envs, device=env.device))
    env.beamdojo_extend = getattr(
        env, "beamdojo_extend", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    env.beamdojo_map_yaw[ids] = 0.2 * (2.0 * torch.rand(n, device=env.device) - 1.0)
    env.beamdojo_map_hx[ids] = 0.03 * (2.0 * torch.rand(n, device=env.device) - 1.0)
    env.beamdojo_map_hy[ids] = 0.03 * (2.0 * torch.rand(n, device=env.device) - 1.0)
    env.beamdojo_extend[ids] = torch.rand(n, device=env.device) < 0.6


def _origins(env):
    env = _raw(env)
    return env.scene.env_origins[:, 0], env.scene.env_origins[:, 1]


def _widths(env):
    env = _raw(env)
    if not hasattr(env, "beamdojo_width"):
        init_beamdojo_state(env)
    return env.beamdojo_width.view(-1, *([1] * 0))


def task_height_at_xy(env: ManagerBasedRLEnv, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    env = _raw(env)
    ox, oy = _origins(env)
    # broadcast env origins to sample dim
    while ox.ndim < x.ndim:
        ox = ox.unsqueeze(-1)
        oy = oy.unsqueeze(-1)
    w = env.beamdojo_width
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    if env.beamdojo_terrain == "stones":
        z = stone_surface_z(
            x,
            y,
            ox,
            oy,
            env.beamdojo_length,
            env.beamdojo_stone_size,
            env.beamdojo_gap,
            on_z=env.beamdojo_on_z,
            off_z=env.beamdojo_off_z,
        )
    else:
        z = beam_surface_z(
            x,
            y,
            ox,
            oy,
            env.beamdojo_length,
            w,
            on_z=env.beamdojo_on_z,
            off_z=env.beamdojo_off_z,
        )
    bias = getattr(env, "beamdojo_z_bias", None)
    if bias is not None:
        b = bias
        while b.ndim < z.ndim:
            b = b.unsqueeze(-1)
        z = z + b
    return z


def task_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    grid_n: int = 15,
    extent: float = 1.4,
) -> torch.Tensor:
    """15×15 yaw-frame elevation of the *task* map (not the physics plane)."""
    env = _raw(env)
    if not hasattr(env, "beamdojo_width"):
        init_beamdojo_state(env)
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w
    yaw = yaw_from_quat_wxyz(robot.data.root_quat_w)
    corrupt = bool(
        getattr(env, "cfg", None) and getattr(env.cfg.observations.policy, "enable_corruption", True)
    )
    if corrupt and hasattr(env, "beamdojo_map_yaw"):
        yaw = yaw + env.beamdojo_map_yaw
    gx, gy = yaw_grid_xy(pos[:, 0], pos[:, 1], yaw, n=grid_n, extent=extent)
    hz = task_height_at_xy(env, gx, gy)
    if corrupt:
        hx = getattr(env, "beamdojo_map_hx", None)
        hy = getattr(env, "beamdojo_map_hy", None)
        if hx is not None and hy is not None:
            hz = tilt_height_grid(hz, grid_n, hx, hy)
        extend = getattr(env, "beamdojo_extend", None)
        if extend is not None:
            hz = dilate_on_cells(
                hz,
                grid_n,
                on_z=float(getattr(env, "beamdojo_on_z", BEAM_ON_Z)),
                off_z=float(getattr(env, "beamdojo_off_z", BEAM_OFF_Z)),
                extend=extend,
            )
    scan = (pos[:, 2].unsqueeze(-1) - hz) - offset
    # ActorCritic 3.0.1 asserts concatenated policy terms are 2D ([N, dim]).
    scan = scan.reshape(scan.shape[0], -1)
    std = float(getattr(env, "beamdojo_scan_noise_std", 0.0) or 0.0)
    if std > 0 and corrupt:
        scan = scan + std * torch.randn_like(scan)
    last = getattr(env, "beamdojo_last_scan", None)
    if corrupt and last is not None and last.shape == scan.shape:
        repeat = torch.rand(env.num_envs, device=env.device) < 0.2
        scan = maybe_repeat_map(scan, last, repeat)
    env.beamdojo_last_scan = scan.detach()
    return scan


def foothold_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    num_samples: int = 15,
    depth_threshold: float = -0.1,
    foot_length: float = 0.15,
    foot_width: float = 0.08,
) -> torch.Tensor:
    """Sampling-based foothold (paper eq. 2) against the dual-terrain task map."""
    env = _raw(env)
    if not hasattr(env, "beamdojo_width"):
        init_beamdojo_state(env)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]
    foot_sensor_indices, foot_names = contact_sensor.find_bodies(sensor_cfg.body_names, preserve_order=True)
    foot_robot_indices, _ = robot.find_bodies(foot_names, preserve_order=True)
    foot_pos = robot.data.body_pos_w[:, foot_robot_indices, :]
    foot_quat = robot.data.body_quat_w[:, foot_robot_indices, :]
    forces_z = contact_sensor.data.net_forces_w[:, foot_sensor_indices, 2]
    in_contact = torch.abs(forces_z) > 1.0

    n_x, n_y = 5, 3
    xs = torch.linspace(-foot_length / 2, foot_length / 2, n_x, device=env.device)
    ys = torch.linspace(-foot_width / 2, foot_width / 2, n_y, device=env.device)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    local = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.zeros(n_x * n_y, device=env.device)],
        dim=-1,
    )
    n_samples = local.shape[0]
    penalty = torch.zeros(env.num_envs, device=env.device)
    ox, oy = _origins(env)
    w = env.beamdojo_width
    for foot_idx in range(foot_pos.shape[1]):
        quat = foot_quat[:, foot_idx].unsqueeze(1).expand(-1, n_samples, -1)
        pts = quat_apply(quat, local.unsqueeze(0).expand(env.num_envs, -1, -1))
        pts = pts + foot_pos[:, foot_idx].unsqueeze(1)
        n_off = foothold_off_count(
            pts[..., 0],
            pts[..., 1],
            ox.unsqueeze(-1),
            oy.unsqueeze(-1),
            env.beamdojo_length,
            w.unsqueeze(-1),
            depth_threshold=depth_threshold,
            on_z=env.beamdojo_on_z,
            off_z=env.beamdojo_off_z,
            terrain=env.beamdojo_terrain,
            stone_size=env.beamdojo_stone_size,
            gap=env.beamdojo_gap,
        )
        penalty = penalty + in_contact[:, foot_idx].float() * n_off
    rew = -penalty
    env.beamdojo_foothold_step = rew.detach()
    return rew


def off_task_terrain(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_off_samples: float = 8.0,
    foot_length: float = 0.15,
    foot_width: float = 0.08,
    depth_threshold: float = -0.1,
) -> torch.Tensor:
    """Stage 2: terminate when a contacting foot is mostly off the task terrain."""
    env = _raw(env)
    foothold_reward(
        env,
        sensor_cfg,
        depth_threshold=depth_threshold,
        foot_length=foot_length,
        foot_width=foot_width,
    )
    # foothold_step is -count; off if any contacting foot dropped enough samples
    return (-env.beamdojo_foothold_step) >= min_off_samples


def tighten_beam_width(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    start_width: float = 0.40,
    end_width: float = 0.20,
    horizon_steps: int = 240_000,
) -> None:
    """Global curriculum: task map and Stage 2 collision cuboid start → end."""
    env = _raw(env)
    if not hasattr(env, "beamdojo_width"):
        init_beamdojo_state(env, width=start_width)
    step = float(getattr(env, "common_step_counter", 0))
    t = min(max(step / max(horizon_steps, 1), 0.0), 1.0)
    w = start_width + (end_width - start_width) * t
    env.beamdojo_width[:] = w
    prev = getattr(env, "beamdojo_phys_width", None)
    if should_update_physical_width(None if prev is None else float(prev), w):
        env.beamdojo_phys_width = w
        _scale_task_beam_prims(env, physical_beam_scale_y(w, start_width))


def _scale_task_beam_prims(env: ManagerBasedRLEnv, scale_y: float) -> None:
    """Resize cloned TaskBeam cuboids so Stage 2 physics matches the curriculum width."""
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    scene = getattr(env, "scene", None)
    roots = getattr(scene, "env_prim_paths", None)
    n = int(getattr(env, "num_envs", 0) or 0)
    sy = float(scale_y)
    for path in task_beam_prim_paths(roots, n):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim)
        scale_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(1.0, sy, 1.0))


def disable_ground_collision(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    prim_paths: tuple[str, ...] = ("/World/ground", "/World/defaultGroundPlane"),
) -> None:
    """Turn off TerrainImporter plane collision so Stage 2 cannot walk beside the beam."""
    del env, env_ids
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                attr = UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr()
                if attr:
                    attr.Set(False)
            phys = child.GetAttribute("physics:collisionEnabled")
            if phys:
                phys.Set(False)
