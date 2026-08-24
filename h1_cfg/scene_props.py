"""Visual / collision cuboids for the imagined (Stage 1) or real (Stage 2) beam."""

from __future__ import annotations

from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg


BEAM_LENGTH = 8.0
BEAM_WIDTH_HARD = 0.20
BEAM_WIDTH_EASY = 0.40
BEAM_THICKNESS = 0.08
# Raised so walking off is a real drop. The TerrainImporter plane at z=0 is
# collision-disabled in Stage 2; a world catcher below fall-terminate (0.40 m)
# stops infinite fall. H1 pelvis_z=1.05 ⇒ catcher top must be < -0.65 so a
# landing cannot stand above the height done.
BEAM_CENTER_Z = 0.24
CATCHER_Z = -0.90
CATCHER_THICKNESS = 0.10
CATCHER_SIZE_XY = 400.0


def task_beam_cfg(
    *,
    collision: bool,
    width: float = BEAM_WIDTH_HARD,
    length: float = BEAM_LENGTH,
    thickness: float = BEAM_THICKNESS,
    center_z: float = BEAM_CENTER_Z,
    color: tuple[float, float, float] = (0.85, 0.55, 0.12),
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TaskBeam",
        spawn=sim_utils.CuboidCfg(
            size=(length, width, thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(length * 0.5, 0.0, center_z)),
    )


def stone_cfg(
    index: int,
    *,
    collision: bool,
    size: float = 0.20,
    gap: float = 0.10,
    thickness: float = BEAM_THICKNESS,
    center_z: float = BEAM_CENTER_Z,
) -> AssetBaseCfg:
    pitch = size + gap
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/TaskStone{index}",
        spawn=sim_utils.CuboidCfg(
            size=(size, size, thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(index * pitch + size * 0.5, 0.0, center_z)),
    )


def add_stepping_stones(scene, count: int = 24, collision: bool = True) -> None:
    for i in range(count):
        setattr(scene, f"task_stone_{i}", stone_cfg(i, collision=collision))


def catcher_cfg(z: float = CATCHER_Z) -> AssetBaseCfg:
    """Kinematic world pad below Stage 2 fall height. Not cloned per env."""
    return AssetBaseCfg(
        prim_path="/World/catcher",
        spawn=sim_utils.CuboidCfg(
            size=(CATCHER_SIZE_XY, CATCHER_SIZE_XY, CATCHER_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.06, 0.08)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z)),
    )
