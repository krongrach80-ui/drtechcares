import {
  POSE_INDEX,
  checkVisibilityAndBoundaryGate,
  computeJointAngle3D,
  evaluateArmKinematics,
} from './armJointProcessor';
import { evaluateCompensations } from './compensation';
import { JointAngleSmoother } from './oneEuroFilter';
import type {
  BodySide,
  CompensationAlert,
  ExerciseConfig,
  PoseLandmark,
  RepState,
  TrilingualText,
} from './types';

export const REP_DEBOUNCE_MS = 350;

export interface SessionProgressSnapshot {
  exerciseId: string;
  state: RepState;
  currentAngleDeg: number | null;
  lastReliableAngleDeg: number | null;
  elbowAngleDeg: number | null;
  currentRep: number;
  targetReps: number;
  currentSet: number;
  targetSets: number;
  holdElapsedSec: number;
  holdTargetSec: number;
  holdPaused: boolean;
  restRemainingSec: number;
  maxRomDeg: number;
  accuracyPercent: number;
  warningMessage: TrilingualText | null;
  compensations: CompensationAlert[];
  repJustCompleted: boolean;
  setJustCompleted: boolean;
  sessionCompleted: boolean;
}

/**
 * Single Unified Pose Evaluation Pipeline:
 * Combines ArmJointProcessor (Visibility & Boundary Gate + Straight-Arm Guard)
 * + JointAngleSmoother (EMA 0.35 + Spike Clamp 28°)
 * + Compensation Rules + RepStateMachine (IDLE -> MOVING_TO_TARGET -> HOLDING_TARGET -> RETURNING_TO_START -> +1 Rep).
 */
export class RehabExercisePipeline {
  private readonly smoother = new JointAngleSmoother();
  private state: RepState = 'IDLE';
  private currentRep = 0;
  private currentSet = 1;
  private holdAccumMs = 0;
  private restAccumMs = 0;
  private lastTickMs: number | null = null;
  private lastRepCompletedMs = 0;
  private maxRomDeg = 0;
  private goodFrames = 0;
  private totalEvaluatedFrames = 0;

  constructor(
    private exercise: ExerciseConfig,
    private overrideSide?: BodySide,
    private overrideSets?: number,
    private overrideReps?: number,
    private overrideHoldSec?: number,
  ) {
    this.smoother.setExercise(exercise.id);
  }

  reset(newExercise?: ExerciseConfig): void {
    if (newExercise) {
      this.exercise = newExercise;
    }
    this.smoother.setExercise(this.exercise.id);
    this.smoother.reset();
    this.state = 'IDLE';
    this.currentRep = 0;
    this.currentSet = 1;
    this.holdAccumMs = 0;
    this.restAccumMs = 0;
    this.lastTickMs = null;
    this.lastRepCompletedMs = 0;
    this.maxRomDeg = 0;
    this.goodFrames = 0;
    this.totalEvaluatedFrames = 0;
  }

  private resolveActiveSide(): BodySide {
    if (this.overrideSide) return this.overrideSide;
    return this.exercise.detection.side === 'left' ? 'left' : 'right';
  }

  private computeModeRawMetric(landmarks: readonly PoseLandmark[], side: BodySide): {
    angleDeg: number | null;
    elbowAngleDeg: number | null;
    warningMessage: TrilingualText | null;
    elbowGuardFailed: boolean;
    compensations: CompensationAlert[];
  } {
    const det = this.exercise.detection;

    if (det.jointGroup === 'shoulder' || det.jointGroup === 'elbow') {
      const armRes = evaluateArmKinematics(landmarks, side, {
        jointGroup: det.jointGroup,
        requireStraightElbow: det.requireStraightElbow,
        minElbowDeg: det.minElbowDeg,
      });

      const compensations =
        armRes.gate.valid
          ? evaluateCompensations(landmarks, side, {
              elbowAngleDeg: armRes.elbowAngleDeg,
              requireStraightElbow: det.requireStraightElbow,
              minElbowDeg: det.minElbowDeg,
              expectedPlane:
                det.expectedPlane === 'flexion' || det.expectedPlane === 'abduction'
                  ? det.expectedPlane
                  : 'any',
            })
          : [];

      return {
        angleDeg: armRes.angleDeg,
        elbowAngleDeg: armRes.elbowAngleDeg,
        warningMessage: armRes.gate.warningMessage ?? armRes.elbowWarning,
        elbowGuardFailed: armRes.elbowGuardFailed,
        compensations,
      };
    }

    if (det.jointGroup === 'knee' || det.jointGroup === 'hip' || det.jointGroup === 'ankle') {
      const hipIdx = side === 'left' ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP;
      const kneeIdx = side === 'left' ? POSE_INDEX.LEFT_KNEE : POSE_INDEX.RIGHT_KNEE;
      const ankleIdx = side === 'left' ? POSE_INDEX.LEFT_ANKLE : POSE_INDEX.RIGHT_ANKLE;
      const shoulderIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;

      const gate = checkVisibilityAndBoundaryGate(landmarks, [
        { index: hipIdx, side, joint: 'hip' },
        { index: kneeIdx, side, joint: 'knee' },
        { index: ankleIdx, side, joint: 'ankle' },
      ]);
      if (!gate.valid) {
        return {
          angleDeg: null,
          elbowAngleDeg: null,
          warningMessage: gate.warningMessage,
          elbowGuardFailed: false,
          compensations: [],
        };
      }
      const angle =
        det.jointGroup === 'knee'
          ? computeJointAngle3D(landmarks[hipIdx], landmarks[kneeIdx], landmarks[ankleIdx])
          : det.jointGroup === 'hip'
            ? computeJointAngle3D(landmarks[shoulderIdx] ?? landmarks[hipIdx], landmarks[hipIdx], landmarks[kneeIdx])
            : Math.max(0, (landmarks[kneeIdx].y - landmarks[ankleIdx].y) * 180);

      return {
        angleDeg: angle,
        elbowAngleDeg: null,
        warningMessage: null,
        elbowGuardFailed: false,
        compensations: [],
      };
    }

    // Neck tilt or balance mode
    const nose = landmarks[POSE_INDEX.NOSE];
    const lShoulder = landmarks[POSE_INDEX.LEFT_SHOULDER];
    const rShoulder = landmarks[POSE_INDEX.RIGHT_SHOULDER];
    const gate = checkVisibilityAndBoundaryGate(landmarks, [
      { index: POSE_INDEX.LEFT_SHOULDER, side: 'left', joint: 'shoulder' },
      { index: POSE_INDEX.RIGHT_SHOULDER, side: 'right', joint: 'shoulder' },
    ]);
    if (!gate.valid || !nose) {
      return {
        angleDeg: null,
        elbowAngleDeg: null,
        warningMessage: gate.warningMessage,
        elbowGuardFailed: false,
        compensations: [],
      };
    }
    const midShoulderX = (lShoulder.x + rShoulder.x) / 2;
    const midShoulderY = (lShoulder.y + rShoulder.y) / 2;
    const neckAngle = Math.abs(
      (Math.atan2(nose.x - midShoulderX, Math.max(0.05, midShoulderY - nose.y)) * 180) / Math.PI,
    );

    return {
      angleDeg: Number(neckAngle.toFixed(2)),
      elbowAngleDeg: null,
      warningMessage: null,
      elbowGuardFailed: false,
      compensations: [],
    };
  }

  step(landmarks: readonly PoseLandmark[], timestampMs: number): SessionProgressSnapshot {
    const targetReps = this.overrideReps ?? this.exercise.detection.defaultReps;
    const targetSets = this.overrideSets ?? this.exercise.detection.defaultSets;
    const holdTargetSec = this.overrideHoldSec ?? this.exercise.detection.holdSec;
    const side = this.resolveActiveSide();

    const dtMs = this.lastTickMs === null ? 0 : Math.max(0, Math.min(250, timestampMs - this.lastTickMs));
    this.lastTickMs = timestampMs;

    if (this.state === 'COMPLETED') {
      return this.snapshot(null, this.smoother.getLastReliable(side), null, false, null, [], false, false);
    }

    if (this.state === 'RESTING_BETWEEN_SETS') {
      this.restAccumMs += dtMs;
      if (this.restAccumMs >= 5000) {
        this.restAccumMs = 0;
        this.currentSet += 1;
        this.currentRep = 0;
        this.state = 'IDLE';
      }
      return this.snapshot(null, this.smoother.getLastReliable(side), null, false, null, [], false, false);
    }

    const raw = this.computeModeRawMetric(landmarks, side);
    const smoothedOut = this.smoother.update(side, raw.angleDeg, timestampMs);

    // If landmarks failed visibility/boundary gating -> angleDeg is null, pause hold, retain last reliable
    if (smoothedOut.smoothed === null) {
      return this.snapshot(
        null,
        smoothedOut.lastReliable,
        raw.elbowAngleDeg,
        true,
        raw.warningMessage,
        raw.compensations,
        false,
        false,
      );
    }

    const angle = smoothedOut.smoothed;
    if (angle > this.maxRomDeg) this.maxRomDeg = angle;

    const det = this.exercise.detection;
    const inTargetZone = angle >= det.targetMinDeg && angle <= det.targetMaxDeg;
    const inRestZone = angle <= det.startRestMaxDeg;
    const blockedByElbow = raw.elbowGuardFailed;

    this.totalEvaluatedFrames += 1;
    if (!blockedByElbow && raw.compensations.length === 0) {
      this.goodFrames += 1;
    }

    let repJustCompleted = false;
    let setJustCompleted = false;

    if (blockedByElbow) {
      // Do not advance hold or count reps while elbow is bent
      return this.snapshot(
        angle,
        smoothedOut.lastReliable,
        raw.elbowAngleDeg,
        true,
        raw.warningMessage,
        raw.compensations,
        false,
        false,
      );
    }

    // Timer / Balance mode advances hold directly while in target zone
    if (det.mode === 'timer' || det.mode === 'hold_balance') {
      if (inTargetZone) {
        this.state = 'HOLDING_TARGET';
        this.holdAccumMs += dtMs;
        if (this.holdAccumMs >= holdTargetSec * 1000) {
          this.holdAccumMs = 0;
          this.currentRep += 1;
          repJustCompleted = true;
          this.state = 'IDLE';
        }
      } else {
        this.state = 'MOVING_TO_TARGET';
      }
    } else {
      // angle_rep_hold / heel_raise / neck_tilt state machine
      switch (this.state) {
        case 'IDLE': {
          if (angle > det.startRestMaxDeg) {
            this.state = 'MOVING_TO_TARGET';
          }
          break;
        }
        case 'MOVING_TO_TARGET': {
          if (inTargetZone) {
            this.state = 'HOLDING_TARGET';
            this.holdAccumMs = 0;
          } else if (inRestZone) {
            this.state = 'IDLE';
          }
          break;
        }
        case 'HOLDING_TARGET': {
          if (!inTargetZone) {
            // Hysteresis: small dip resets hold progress back to moving
            if (angle < det.targetMinDeg - 8) {
              this.state = 'MOVING_TO_TARGET';
              this.holdAccumMs = 0;
            }
          } else {
            this.holdAccumMs += dtMs;
            if (this.holdAccumMs >= holdTargetSec * 1000) {
              this.state = 'RETURNING_TO_START';
            }
          }
          break;
        }
        case 'RETURNING_TO_START': {
          if (inRestZone && timestampMs - this.lastRepCompletedMs >= REP_DEBOUNCE_MS) {
            this.lastRepCompletedMs = timestampMs;
            this.currentRep += 1;
            this.holdAccumMs = 0;
            repJustCompleted = true;
            this.state = 'IDLE';
          }
          break;
        }
      }
    }

    if (repJustCompleted && this.currentRep >= targetReps) {
      setJustCompleted = true;
      if (this.currentSet >= targetSets) {
        this.state = 'COMPLETED';
      } else {
        this.state = 'RESTING_BETWEEN_SETS';
        this.restAccumMs = 0;
      }
    }

    return this.snapshot(
      angle,
      smoothedOut.lastReliable,
      raw.elbowAngleDeg,
      false,
      raw.warningMessage,
      raw.compensations,
      repJustCompleted,
      setJustCompleted,
    );
  }

  private snapshot(
    currentAngleDeg: number | null,
    lastReliableAngleDeg: number | null,
    elbowAngleDeg: number | null,
    holdPaused: boolean,
    warningMessage: TrilingualText | null,
    compensations: CompensationAlert[],
    repJustCompleted: boolean,
    setJustCompleted: boolean,
  ): SessionProgressSnapshot {
    const targetReps = this.overrideReps ?? this.exercise.detection.defaultReps;
    const targetSets = this.overrideSets ?? this.exercise.detection.defaultSets;
    const holdTargetSec = this.overrideHoldSec ?? this.exercise.detection.holdSec;
    const accuracyPercent =
      this.totalEvaluatedFrames === 0
        ? 100
        : Math.round((this.goodFrames / this.totalEvaluatedFrames) * 100);

    return {
      exerciseId: this.exercise.id,
      state: this.state,
      currentAngleDeg,
      lastReliableAngleDeg,
      elbowAngleDeg,
      currentRep: this.currentRep,
      targetReps,
      currentSet: this.currentSet,
      targetSets,
      holdElapsedSec: Number((this.holdAccumMs / 1000).toFixed(1)),
      holdTargetSec,
      holdPaused,
      restRemainingSec: this.state === 'RESTING_BETWEEN_SETS' ? Math.max(0, Math.ceil((5000 - this.restAccumMs) / 1000)) : 0,
      maxRomDeg: Number(this.maxRomDeg.toFixed(1)),
      accuracyPercent,
      warningMessage,
      compensations,
      repJustCompleted,
      setJustCompleted,
      sessionCompleted: this.state === 'COMPLETED',
    };
  }
}

