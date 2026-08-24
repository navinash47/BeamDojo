"""Per-robot names for BeamDojo H1 / G1 configs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    name: str
    pelvis_z: float
    feet_body: str
    ankle_joints: str | list[str]
    hip_yaw: str
    hip_roll: str
    arm_joints: list[str]
    torso_joint: str
    action_joints: list[str] | None
    scanner_prim: str


H1 = RobotSpec(
    name="h1",
    pelvis_z=1.05,
    feet_body=".*_ankle_link",
    ankle_joints=".*_ankle",
    hip_yaw=".*_hip_yaw",
    hip_roll=".*_hip_roll",
    arm_joints=[".*_shoulder_.*", ".*_elbow"],
    torso_joint="torso",
    action_joints=None,
    scanner_prim="{ENV_REGEX_NS}/Robot/torso_link",
)

G1 = RobotSpec(
    name="g1",
    pelvis_z=0.74,
    feet_body=".*_ankle_roll_link",
    ankle_joints=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    hip_yaw=".*_hip_yaw_joint",
    hip_roll=".*_hip_roll_joint",
    arm_joints=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_pitch_joint",
        ".*_elbow_roll_joint",
    ],
    torso_joint="torso_joint",
    # Paper: 12 lower-body actions (no arms).
    action_joints=[
        ".*_hip_yaw_joint",
        ".*_hip_roll_joint",
        ".*_hip_pitch_joint",
        ".*_knee_joint",
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
    ],
    scanner_prim="{ENV_REGEX_NS}/Robot/torso_link",
)
