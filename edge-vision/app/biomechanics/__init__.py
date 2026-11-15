"""3D clinical biomechanics solver, calibrator, exercise rules, and FSM."""

from app.biomechanics.calibrator import StandingCalibrator
from app.biomechanics.exercises import ALL_26_EXERCISES, ClinicalExerciseRule, resolve_exercise_rule
from app.biomechanics.solver_3d import (
    Biomechanics3DSolver,
    angle_between_vectors_3d,
    compute_pelvic_tilt_deg,
    compute_shoulder_tilt_deg,
    compute_torso_plane,
    joint_angle_3d,
)
from app.biomechanics.state_machine import ClinicalRehabFSM

__all__ = [
    "ALL_26_EXERCISES",
    "Biomechanics3DSolver",
    "ClinicalExerciseRule",
    "ClinicalRehabFSM",
    "StandingCalibrator",
    "angle_between_vectors_3d",
    "compute_pelvic_tilt_deg",
    "compute_shoulder_tilt_deg",
    "compute_torso_plane",
    "joint_angle_3d",
    "resolve_exercise_rule",
]

