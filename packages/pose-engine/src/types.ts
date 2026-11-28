export interface PoseLandmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

export type BodySide = 'left' | 'right';

export type ExerciseEvaluationMode =
  | 'angle_rep_hold'
  | 'timer'
  | 'hold_balance'
  | 'heel_raise'
  | 'neck_tilt'
  | 'wrist_finger'
  | 'shrug_roll'
  | 'bilateral_raise';

export interface TrilingualText {
  th: string;
  en: string;
  cn: string;
}

export interface ExerciseDetectionConfig {
  mode: ExerciseEvaluationMode;
  jointGroup: 'shoulder' | 'elbow' | 'wrist' | 'knee' | 'hip' | 'ankle' | 'neck' | 'trunk' | 'balance';
  side: BodySide | 'both';
  targetMinDeg: number;
  targetMaxDeg: number;
  targetRomDeg?: number;
  startRestMaxDeg: number;
  holdSec: number;
  defaultSets: number;
  defaultReps: number;
  requireStraightElbow?: boolean;
  minElbowDeg?: number;
  expectedPlane?:
    | 'flexion'
    | 'abduction'
    | 'rotation'
    | 'cross_body'
    | 'scap_protract'
    | 'wrist_finger'
    | 'forearm_turn'
    | 'thumb_pinch'
    | 'hand_to_head'
    | 'shrug_roll'
    | 'neck_turn'
    | 'bilateral'
    | 'bilateral_push'
    | 'any';
}

export interface ExerciseConfig {
  id: string;
  code: string;
  category: 'upper_limb' | 'lower_limb' | 'neck_trunk' | 'balance';
  clinicalFocus?: 'elderly_adl' | 'moph_elderly' | 'upper_body' | 'general';
  name: TrilingualText;
  instruction: TrilingualText;
  detection: ExerciseDetectionConfig;
}

export interface CompensationAlert {
  type: 'trunk_lean' | 'scapular_hiking' | 'elbow_flexion_cheat' | 'elevation_plane';
  message: TrilingualText;
  severity: 'warning' | 'blocking';
}

export type RepState =
  | 'IDLE'
  | 'MOVING_TO_TARGET'
  | 'HOLDING_TARGET'
  | 'RETURNING_TO_START'
  | 'RESTING_BETWEEN_SETS'
  | 'COMPLETED';

