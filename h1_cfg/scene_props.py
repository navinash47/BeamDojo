"""Visual / collision cuboids for the imagined (Stage 1) or real (Stage 2) beam."""

from __future__ import annotations

from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg


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
STONE_COUNT = 24

# Same as the locomotion TerrainImporter. Without this, spawn_cuboid adds no
# physics material and Stage 2 feet can ice-skate off the kinematic beam.
_WALK_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
)


def _kinematic_cuboid(
    *,
    size: tuple[float, float, float],
    collision: bool,
    color: tuple[float, float, float],
) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
        physics_material=_WALK_MATERIAL,
    )


def task_beam_cfg(
    *,
    collision: bool,
    width: float = BEAM_WIDTH_HARD,
    length: float = BEAM_LENGTH,
    thickness: float = BEAM_THICKNESS,
    center_z: float = BEAM_CENTER_Z,
    color: tuple[float, float, float] = (0.85, 0.55, 0.12),
) -> RigidObjectCfg:
    """Kinematic beam. RigidObjectCfg so InteractiveScene clones PhysX collision per env.

    AssetBaseCfg is for lights / world pads. A colliding cuboid on that type is not in
    the rigid-object views or env collision filters, so Stage 2 can fall through the beam
    or hit a neighbor env's beam.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TaskBeam",
        spawn=_kinematic_cuboid(size=(length, width, thickness), collision=collision, color=color),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(length * 0.5, 0.0, center_z)),
    )


def stone_cfg(
    index: int,
    *,
    collision: bool,
    size: float = 0.20,
    gap: float = 0.10,
    thickness: float = BEAM_THICKNESS,
    center_z: float = BEAM_CENTER_Z,
) -> RigidObjectCfg:
    pitch = size + gap
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/TaskStone{index}",
        spawn=_kinematic_cuboid(
            size=(size, size, thickness),
            collision=collision,
            color=(0.55, 0.55, 0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(index * pitch + size * 0.5, 0.0, center_z)),
    )


def add_stepping_stones(scene, count: int = STONE_COUNT, collision: bool = True) -> None:
    for i in range(count):
        setattr(scene, f"task_stone_{i}", stone_cfg(i, collision=collision))


def catcher_cfg(z: float = CATCHER_Z) -> AssetBaseCfg:
    """World-level kinematic pad below Stage 2 fall height. Not cloned per env.

    Must stay AssetBaseCfg: InteractiveScene.reset() calls RigidObject.reset(env_ids)
    for every rigid object, and a single ``/World/catcher`` body cannot index 1024 envs.

    ``collision_group=-1`` puts the pad in the global collision filter (same as
    ``/World/ground``) so GPU env-id filtering still lets every robot land on it.
    """
    return AssetBaseCfg(
        prim_path="/World/catcher",
        spawn=_kinematic_cuboid(
            size=(CATCHER_SIZE_XY, CATCHER_SIZE_XY, CATCHER_THICKNESS),
            collision=True,
            color=(0.06, 0.06, 0.08),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z)),
        collision_group=-1,
    )
