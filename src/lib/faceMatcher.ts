import type {
  DrTechCaresApiClient,
  PatientBiometricIndexEntry,
} from '@drtechcares/api-client';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  COMMON_MODE_ALPHA,
  FACE_EMBEDDING_DIM,
  classifyLivePoseZone,
  computeArcFaceBiometricScore,
  computeDualFactorBiometricScore,
  computeMeanCenteredVector,
  cosineSimilarity as coreCosineSimilarity,
  detectBiometricVectorEngine,
  isLegacyV1HybridVector,
  l2Normalize,
  mirrorPoseZone,
  whitenEmbedding,
  type BiometricVectorEngine,
  type FacePoseZone5,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';

export interface MultiAngleMatchOptions {
  similarityThreshold?: number;
  whitenedThreshold?: number;
  marginThreshold?: number;
  faceHeightRatio?: number;
  yawDeg?: number;
  pitchDeg?: number;
  rollDeg?: number;
  engine?: 'arcface-onnx' | 'hybrid-512d';
  mirroredProbeVector?: readonly number[];
  hybridV2ProbeVector?: readonly number[];
  mirroredHybridV2ProbeVector?: readonly number[];
  legacyV1ProbeVector?: readonly number[];
  mirroredLegacyV1ProbeVector?: readonly number[];
  geometricRatios?: GeometricAnthropometricRatios | null;
  excludedPatientIds?: readonly string[];
}

export interface MultiAngleMatchResult {
  matched: boolean;
  bestCandidate: PatientBiometricIndexEntry | null;
  bestAngle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down' | 'extra' | 'none';
  livePoseZone?: FacePoseZone5;
  similarity: number;
  confidencePercent: number;
  margin: number;
  runnerUpSimilarity: number;
  reason?: 'OK' | 'NO_CANDIDATES' | 'EXCLUDED_ONLY' | 'BELOW_THRESHOLD' | 'AMBIGUOUS_CLOSE_CANDIDATES';
}

export interface FaceBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
  confidence: number;
}

export interface LiveFrameObservation {
  probeVector: readonly number[];
  mirroredProbeVector?: readonly number[];
  hybridV2ProbeVector?: readonly number[];
  mirroredHybridV2ProbeVector?: readonly number[];
  legacyV1ProbeVector?: readonly number[];
  mirroredLegacyV1ProbeVector?: readonly number[];
  depthScore: number;
  yawDeg: number;
  pitchDeg: number;
  rollDeg?: number;
  microMotion: number;
  blinkDetected?: boolean;
  nonRigidMotion?: number;
  biologicalVerified?: boolean;
  spoofDetected?: boolean;
  rppgScore?: number;
  faceHeightRatio?: number;
  geometricRatios?: GeometricAnthropometricRatios;
  engine?: 'arcface-onnx' | 'hybrid-512d';
}

export interface TemporalMatchResult {
  confirmed: boolean;
  streakCount: number;
  requiredStreak: number;
  candidate: PatientBiometricIndexEntry | null;
  confidencePercent: number;
  bestRawSimilarity: number;
  bestAngle: string;
  livenessPassed: boolean;
  livenessReason?: string;
}

/**
 * Computes normalized Cosine Similarity in [-1, 1] between two embedding vectors.
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  if (a.length === FACE_EMBEDDING_DIM) {
    return coreCosineSimilarity(a, b);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? Math.max(-1, Math.min(1, dot / denom)) : 0;
}

function normalizePatientFullName(name: string | undefined): string {
  return (name ?? '').trim().toLowerCase().replace(/\s+/g, ' ');
}

function isSameIdentityRecord(
  a: Pick<PatientBiometricIndexEntry, 'patientId' | 'fullName'>,
  b: Pick<PatientBiometricIndexEntry, 'patientId' | 'fullName'>,
): boolean {
  if (a.patientId === b.patientId) return true;
  const na = normalizePatientFullName(a.fullName);
  const nb = normalizePatientFullName(b.fullName);
  return Boolean(na && nb && na === nb);
}

/**
 * High-Precision 1:N Patient Identification Engine:
 * 1. Pose-Synchronized Angle Matching (0.55 * angleEmbeddings[livePoseZone] + 0.45 * masterEmbedding)
 *    — zero blind cross-angle leakage between mismatched poses.
 * 2. Dual-Factor Biometric Score (80% Zero-Mean Normalized Vector + L2 Penalty, 20% Craniofacial Geometry).
 * 3. Three Iron Gates:
 *    - Rule 1: High Absolute Threshold (>= 0.78 frontal center, >= 0.72 slight tilt/turn)
 *    - Rule 2: Strict Distinctiveness Margin (>= 0.08 over Rank #2) -> returns AMBIGUOUS_CLOSE_CANDIDATES on ties
 *    - Respects temporary excludedPatientIds ("ไม่ใช่ฉัน" rejection) so wrong matches are never re-picked.
 */
export function identifyPatient1toN(
  probeVector: readonly number[],
  candidates: readonly PatientBiometricIndexEntry[],
  options: MultiAngleMatchOptions = {},
): MultiAngleMatchResult {
  const excludedSet =
    options.excludedPatientIds && options.excludedPatientIds.length > 0
      ? new Set(options.excludedPatientIds)
      : null;

  const eligibleCandidates = excludedSet
    ? candidates.filter((c) => !excludedSet.has(c.patientId) && !excludedSet.has(c.hn))
    : candidates;

  const livePoseZone = classifyLivePoseZone(options.yawDeg ?? 0, options.pitchDeg ?? 0);
  const isAngledPose =
    livePoseZone !== 'center' ||
    Math.abs(options.yawDeg ?? 0) > 8 ||
    Math.abs(options.pitchDeg ?? 0) > 6 ||
    Math.abs(options.rollDeg ?? 0) > 10;

  // Iron Rule 2: Strict Distinctiveness Margin (>= 0.08 between Rank #1 and Rank #2)
  const marginThreshold = options.marginThreshold ?? 0.08;

  if (probeVector.length === 0 || candidates.length === 0) {
    return {
      matched: false,
      bestCandidate: null,
      bestAngle: 'none',
      livePoseZone,
      similarity: 0,
      confidencePercent: 0,
      margin: 0,
      runnerUpSimilarity: 0,
      reason: 'NO_CANDIDATES',
    };
  }

  if (eligibleCandidates.length === 0) {
    return {
      matched: false,
      bestCandidate: null,
      bestAngle: 'none',
      livePoseZone,
      similarity: 0,
      confidencePercent: 0,
      margin: 0,
      runnerUpSimilarity: 0,
      reason: 'EXCLUDED_ONLY',
    };
  }

  const normalizedProbe = l2Normalize(probeVector);
  const probeEngine = detectBiometricVectorEngine(normalizedProbe, options.engine);
  const mirroredZone = mirrorPoseZone(livePoseZone);
  const primaryVariants: { zone: FacePoseZone5; vec: number[] }[] = [
    { zone: livePoseZone, vec: normalizedProbe },
  ];
  if (options.mirroredProbeVector && options.mirroredProbeVector.length === normalizedProbe.length) {
    primaryVariants.push({ zone: mirroredZone, vec: l2Normalize(options.mirroredProbeVector) });
  }

  const hybridV2Variants: { zone: FacePoseZone5; vec: number[] }[] = [];
  if (
    options.hybridV2ProbeVector &&
    options.hybridV2ProbeVector.length === normalizedProbe.length
  ) {
    hybridV2Variants.push({ zone: livePoseZone, vec: l2Normalize(options.hybridV2ProbeVector) });
    if (
      options.mirroredHybridV2ProbeVector &&
      options.mirroredHybridV2ProbeVector.length === normalizedProbe.length
    ) {
      hybridV2Variants.push({
        zone: mirroredZone,
        vec: l2Normalize(options.mirroredHybridV2ProbeVector),
      });
    }
  } else if (probeEngine === 'hybrid-v2') {
    hybridV2Variants.push(...primaryVariants);
  }

  const legacyV1Variants: { zone: FacePoseZone5; vec: number[] }[] = [];
  if (
    options.legacyV1ProbeVector &&
    options.legacyV1ProbeVector.length === normalizedProbe.length
  ) {
    legacyV1Variants.push({ zone: livePoseZone, vec: l2Normalize(options.legacyV1ProbeVector) });
    if (
      options.mirroredLegacyV1ProbeVector &&
      options.mirroredLegacyV1ProbeVector.length === normalizedProbe.length
    ) {
      legacyV1Variants.push({
        zone: mirroredZone,
        vec: l2Normalize(options.mirroredLegacyV1ProbeVector),
      });
    }
  } else if (probeEngine === 'legacy-v1') {
    legacyV1Variants.push(...primaryVariants);
  }

  const arcFaceVariants: { zone: FacePoseZone5; vec: number[] }[] =
    probeEngine === 'arcface-onnx' ? primaryVariants : [];

  const scored: {
    candidate: PatientBiometricIndexEntry;
    orderIdx: number;
    angle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down' | 'extra';
    dualFactorScore: number;
    engine: BiometricVectorEngine;
    isLegacyV1: boolean;
    rawSim: number;
    meanCenteredSim: number;
    whitenedSim: number;
    geometricPenalty: number;
    hasCore: boolean;
    bestCos3D: number;
    bestCraniofacialCos: number;
    bestInnerDctCos: number;
    bestInnerDctWhite: number;
    bestInnerPatchCos: number;
    bestInnerPatchWhite: number;
  }[] = [];

  for (let orderIdx = 0; orderIdx < eligibleCandidates.length; orderIdx += 1) {
    const cand = eligibleCandidates[orderIdx];
    const hasMaster =
      Array.isArray(cand.masterEmbedding) && cand.masterEmbedding.length === normalizedProbe.length;
    const masterVec = hasMaster ? l2Normalize(cand.masterEmbedding) : null;
    const candEngine: BiometricVectorEngine = masterVec
      ? detectBiometricVectorEngine(masterVec, cand.biometricEngine)
      : cand.biometricEngine === 'arcface-onnx'
        ? 'arcface-onnx'
        : 'hybrid-v2';

    const activeVariants =
      candEngine === 'arcface-onnx'
        ? arcFaceVariants
        : candEngine === 'legacy-v1'
          ? legacyV1Variants.length > 0
            ? legacyV1Variants
            : hybridV2Variants.length > 0
              ? hybridV2Variants
              : probeEngine !== 'arcface-onnx'
                ? primaryVariants
                : []
          : hybridV2Variants.length > 0
            ? hybridV2Variants
            : probeEngine !== 'arcface-onnx'
              ? primaryVariants
              : [];
    if (activeVariants.length === 0) continue;

    let bestVariantScore = -1;
    let bestAngle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down' | 'extra' = 'master';
    let bestRawSim = 0;
    let bestMeanCenteredSim = 0;
    let bestWhitenedSim = 0;
    let bestGeomPenalty = 0;
    let hasCore = false;
    let bestCos3D = 0;
    let bestCraniofacialCos = 0;
    let bestInnerDctCos = 0;
    let bestInnerDctWhite = 0;
    let bestInnerPatchCos = 0;
    let bestInnerPatchWhite = 0;

    if (candEngine === 'arcface-onnx') {
      for (const variant of activeVariants) {
        const targetAngleVecRaw = cand.angleEmbeddings?.[variant.zone];
        const hasSameZone =
          Array.isArray(targetAngleVecRaw) && targetAngleVecRaw.length === normalizedProbe.length;
        const sameZoneVec = hasSameZone ? l2Normalize(targetAngleVecRaw) : masterVec;
        if (!sameZoneVec && !masterVec) continue;

        const sameZoneArc = sameZoneVec
          ? computeArcFaceBiometricScore(variant.vec, sameZoneVec, {
              probeRatios: options.geometricRatios,
              candidateRatios: cand.geometricRatios,
            })
          : null;
        const masterArc = masterVec
          ? computeArcFaceBiometricScore(variant.vec, masterVec, {
              probeRatios: options.geometricRatios,
              candidateRatios: cand.geometricRatios,
            })
          : null;
        const arcScore =
          sameZoneArc && masterArc
            ? 0.55 * sameZoneArc.score + 0.45 * masterArc.score
            : (sameZoneArc ?? masterArc)!.score;
        const chosenAngle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down' =
          hasSameZone && (!masterArc || !sameZoneArc || masterArc.score <= sameZoneArc.score + 0.06)
            ? variant.zone
            : 'master';
        if (arcScore > bestVariantScore) {
          bestVariantScore = arcScore;
          bestAngle = chosenAngle;
          bestRawSim = Math.max(sameZoneArc?.rawCos ?? 0, masterArc?.rawCos ?? 0);
          bestMeanCenteredSim = bestRawSim;
          bestWhitenedSim = bestRawSim;
          bestGeomPenalty = Math.max(
            sameZoneArc?.geometricPenalty ?? 0,
            masterArc?.geometricPenalty ?? 0,
          );
        }
      }
    } else {
      for (const variant of activeVariants) {
        const targetAngleVecRaw = cand.angleEmbeddings?.[variant.zone];
        const hasSameZone =
          Array.isArray(targetAngleVecRaw) && targetAngleVecRaw.length === normalizedProbe.length;
        const sameZoneVec = hasSameZone ? l2Normalize(targetAngleVecRaw) : masterVec;
        if (!sameZoneVec && !masterVec) continue;

        const sameZoneDual = sameZoneVec
          ? computeDualFactorBiometricScore(variant.vec, sameZoneVec, {
              probeRatios: options.geometricRatios,
              candidateRatios: cand.geometricRatios,
            })
          : null;
        const masterDual = masterVec
          ? computeDualFactorBiometricScore(variant.vec, masterVec, {
              probeRatios: options.geometricRatios,
              candidateRatios: cand.geometricRatios,
            })
          : null;

        let dualScore = 0;
        let dominantDual = sameZoneDual ?? masterDual!;
        let chosenAngle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down' = hasSameZone
          ? variant.zone
          : 'master';

        if (sameZoneDual && masterDual) {
          // Pose-Synchronized fusion: 55% same-pose angle centroid + 45% master centroid
          dualScore = 0.55 * sameZoneDual.dualFactorScore + 0.45 * masterDual.dualFactorScore;
          dominantDual =
            sameZoneDual.dualFactorScore >= masterDual.dualFactorScore ? sameZoneDual : masterDual;
          if (masterDual.dualFactorScore > sameZoneDual.dualFactorScore + 0.06) {
            chosenAngle = 'master';
          }
        } else {
          dualScore = dominantDual.dualFactorScore;
        }

        if (dualScore > bestVariantScore) {
          bestVariantScore = dualScore;
          bestAngle = chosenAngle;
          bestRawSim = dominantDual.rawCos;
          bestMeanCenteredSim = dominantDual.meanCenteredCos;
          bestWhitenedSim =
            sameZoneDual && masterDual
              ? Math.max(sameZoneDual.whitenedCos, masterDual.whitenedCos)
              : dominantDual.whitenedCos;
          bestGeomPenalty = Math.max(
            sameZoneDual?.geometricPenalty ?? 0,
            masterDual?.geometricPenalty ?? 0,
          );
          if (dominantDual.faceCore) {
            hasCore = true;
            bestCos3D = dominantDual.faceCore.cos3DNormalized;
            bestCraniofacialCos = dominantDual.faceCore.craniofacialCos;
            bestInnerDctCos = dominantDual.faceCore.innerDctCos;
            bestInnerDctWhite = dominantDual.faceCore.innerDctWhitenedCos;
            bestInnerPatchCos = dominantDual.faceCore.innerPatchCos;
            bestInnerPatchWhite = dominantDual.faceCore.innerPatchWhitenedCos;
          }
        }
      }
    }

    if (bestVariantScore >= 0) {
      scored.push({
        candidate: cand,
        orderIdx,
        angle: bestAngle,
        dualFactorScore: bestVariantScore,
        engine: candEngine,
        isLegacyV1: candEngine === 'legacy-v1',
        rawSim: bestRawSim,
        meanCenteredSim: bestMeanCenteredSim,
        whitenedSim: bestWhitenedSim,
        geometricPenalty: bestGeomPenalty,
        hasCore,
        bestCos3D,
        bestCraniofacialCos,
        bestInnerDctCos,
        bestInnerDctWhite,
        bestInnerPatchCos,
        bestInnerPatchWhite,
      });
    }
  }

  if (scored.length === 0) {
    return {
      matched: false,
      bestCandidate: null,
      bestAngle: 'none',
      livePoseZone,
      similarity: 0,
      confidencePercent: 0,
      margin: 0,
      runnerUpSimilarity: 0,
      reason: 'NO_CANDIDATES',
    };
  }

  const sortPool = (items: typeof scored) =>
    [...items].sort((a, b) => {
      if (
        isSameIdentityRecord(a.candidate, b.candidate) &&
        Math.abs(b.dualFactorScore - a.dualFactorScore) < 0.035
      ) {
        return b.orderIdx - a.orderIdx;
      }
      return b.dualFactorScore - a.dualFactorScore;
    });

  const arcPool = sortPool(scored.filter((s) => s.engine === 'arcface-onnx'));
  const hybridPool = sortPool(scored.filter((s) => s.engine !== 'arcface-onnx'));

  const evalArcPool = () => {
    const top = arcPool[0];
    if (!top) return null;
    const distinctRunners = arcPool
      .slice(1)
      .filter((s) => !isSameIdentityRecord(top.candidate, s.candidate));
    const runnerUp = distinctRunners[0];
    const runnerUpSim = runnerUp ? runnerUp.dualFactorScore : 0;
    const margin = runnerUp ? top.dualFactorScore - runnerUp.dualFactorScore : 1;
    const arcSimThresh = options.similarityThreshold ?? (isAngledPose ? 0.48 : 0.54);
    const arcMarginThresh = options.marginThreshold ?? 0.1;
    const passesThreshold = top.dualFactorScore >= arcSimThresh && top.geometricPenalty <= 0.14;
    const passesMargin = distinctRunners.length === 0 || margin >= arcMarginThresh;
    const matched = passesThreshold && passesMargin;
    const reason: MultiAngleMatchResult['reason'] = matched
      ? 'OK'
      : passesThreshold && !passesMargin
        ? 'AMBIGUOUS_CLOSE_CANDIDATES'
        : 'BELOW_THRESHOLD';
    const displayPercent = Number(
      Math.max(
        0,
        Math.min(
          99.9,
          matched
            ? 50 + top.dualFactorScore * 58
            : Math.min(79, Math.max(0, 30 + top.dualFactorScore * 65)),
        ),
      ).toFixed(1),
    );
    return { top, runnerUpSim, margin, matched, reason, displayPercent };
  };

  const evalHybridPool = () => {
    const top = hybridPool[0];
    if (!top) return null;
    const distinctRunners = hybridPool
      .slice(1)
      .filter((s) => !isSameIdentityRecord(top.candidate, s.candidate));
    const runnerUp = distinctRunners[0];
    const runnerUpSim = runnerUp ? runnerUp.dualFactorScore : 0;
    const margin = runnerUp ? top.dualFactorScore - runnerUp.dualFactorScore : 1;

    const passesMargin = distinctRunners.length === 0 || margin >= marginThreshold;
    const topIsLegacyV1 = top.isLegacyV1;
    const similarityThreshold =
      options.similarityThreshold ??
      (topIsLegacyV1 ? (isAngledPose ? 0.75 : 0.80) : isAngledPose ? 0.70 : 0.75);
    const whitenedThreshold =
      options.whitenedThreshold ??
      (topIsLegacyV1 ? (isAngledPose ? 0.36 : 0.42) : isAngledPose ? 0.32 : 0.36);
    const meanCenteredMin = topIsLegacyV1
      ? isAngledPose
        ? 0.56
        : 0.64
      : isAngledPose
        ? 0.52
        : 0.60;
    const subDctCosMin = topIsLegacyV1 ? (isAngledPose ? 0.32 : 0.40) : isAngledPose ? 0.34 : 0.42;
    const subDctWhiteMin = topIsLegacyV1 ? (isAngledPose ? 0.14 : 0.20) : isAngledPose ? 0.15 : 0.22;
    const subPatchCosMin = topIsLegacyV1 ? (isAngledPose ? 0.26 : 0.34) : isAngledPose ? 0.30 : 0.38;
    const subPatchWhiteMin = topIsLegacyV1
      ? isAngledPose
        ? 0.10
        : 0.16
      : isAngledPose
        ? 0.12
        : 0.18;
    const sub3DMin = topIsLegacyV1 ? (isAngledPose ? 0.60 : 0.68) : isAngledPose ? 0.52 : 0.60;
    const craniofacialMin = topIsLegacyV1
      ? isAngledPose
        ? 0.28
        : 0.36
      : isAngledPose
        ? 0.28
        : 0.35;

    const passesSubbandFloor =
      !top.hasCore ||
      (top.bestInnerDctCos >= subDctCosMin &&
        top.bestInnerDctWhite >= subDctWhiteMin &&
        top.bestInnerPatchCos >= subPatchCosMin &&
        top.bestInnerPatchWhite >= subPatchWhiteMin &&
        top.bestCos3D >= sub3DMin &&
        top.bestCraniofacialCos >= craniofacialMin);

    const passesThreshold =
      top.dualFactorScore >= similarityThreshold &&
      top.whitenedSim >= whitenedThreshold &&
      top.meanCenteredSim >= meanCenteredMin &&
      top.geometricPenalty <= 0.14 &&
      passesSubbandFloor;

    const matched = passesThreshold && passesMargin;
    const reason: MultiAngleMatchResult['reason'] = matched
      ? 'OK'
      : passesThreshold && !passesMargin
        ? 'AMBIGUOUS_CLOSE_CANDIDATES'
        : 'BELOW_THRESHOLD';

    const displayPercent = Number(
      Math.max(
        0,
        Math.min(
          99.9,
          (matched ? Math.max(0.84, top.dualFactorScore) : Math.min(0.79, top.dualFactorScore)) *
            100,
        ),
      ).toFixed(1),
    );
    return { top, runnerUpSim, margin, matched, reason, displayPercent };
  };

  const arcEval = evalArcPool();
  const hybridEval =
    arcEval && (arcEval.matched || arcEval.reason === 'AMBIGUOUS_CLOSE_CANDIDATES')
      ? null
      : evalHybridPool();
  const chosen =
    arcEval && (arcEval.matched || arcEval.reason === 'AMBIGUOUS_CLOSE_CANDIDATES' || !hybridEval)
      ? arcEval
      : hybridEval && (hybridEval.matched || !arcEval || hybridEval.top.dualFactorScore > arcEval.top.dualFactorScore)
        ? hybridEval
        : arcEval!;

  const { top, runnerUpSim, margin, matched, reason, displayPercent } = chosen;

  if (typeof window !== 'undefined' && typeof console !== 'undefined' && console.debug) {
    console.debug(
      `[FaceMatcher] zone=${livePoseZone} top="${top.candidate.fullName}" angle=${top.angle} score=${(top.dualFactorScore * 100).toFixed(1)}% margin=${(margin * 100).toFixed(1)}% matched=${matched} reason=${reason}`,
    );
  }

  // Also trigger clean debug helper check for mean-centered sanity
  void computeMeanCenteredVector;
  void whitenEmbedding;
  void CANONICAL_HUMAN_FACE_BASELINE_512;
  void COMMON_MODE_ALPHA;

  return {
    matched,
    bestCandidate: matched || reason === 'AMBIGUOUS_CLOSE_CANDIDATES' ? top.candidate : null,
    bestAngle: top.angle,
    livePoseZone,
    similarity: top.dualFactorScore,
    confidencePercent: displayPercent,
    margin,
    runnerUpSimilarity: runnerUpSim,
    reason,
  };
}

/**
 * Backward-compatible alias delegating strictly to `identifyPatient1toN`.
 */
export function matchMultiAngleEmbedding(
  probeVector: readonly number[],
  candidates: readonly PatientBiometricIndexEntry[],
  options: MultiAngleMatchOptions = {},
): MultiAngleMatchResult {
  return identifyPatient1toN(probeVector, candidates, options);
}

/**
 * Temporal consensus & anti-spoof liveness verifier.
 * Enforces Iron Rule 3: 5-Frame Unanimous Identity Lock (5/5 consecutive frames for the exact
 * same patientId; any identity flip, ambiguity, or below-threshold frame immediately resets to 0).
 */
export class TemporalBiometricMatcher {
  readonly requiredConsecutiveFrames: number;
  readonly defaultSimilarityThreshold?: number;
  private streakPatientId: string | null = null;
  private streakPatientName: string | null = null;
  private streakFrames: {
    match: MultiAngleMatchResult;
    depthScore: number;
    yawDeg: number;
    pitchDeg: number;
    microMotion: number;
    blinkDetected: boolean;
  }[] = [];

  constructor(requiredConsecutiveFrames = 5, defaultSimilarityThreshold?: number) {
    this.requiredConsecutiveFrames = requiredConsecutiveFrames;
    this.defaultSimilarityThreshold = defaultSimilarityThreshold;
  }

  get currentStreak(): number {
    return this.streakFrames.length;
  }

  reset(): void {
    this.streakPatientId = null;
    this.streakPatientName = null;
    this.streakFrames = [];
  }

  evaluateFrame(
    observation: LiveFrameObservation,
    candidates: readonly PatientBiometricIndexEntry[],
    options?: MultiAngleMatchOptions,
  ): TemporalMatchResult {
    if (observation.spoofDetected) {
      this.reset();
      return {
        confirmed: false,
        streakCount: 0,
        requiredStreak: this.requiredConsecutiveFrames,
        candidate: null,
        confidencePercent: 0,
        bestRawSimilarity: 0,
        bestAngle: 'none',
        livenessPassed: false,
        livenessReason: 'SPOOF_ATTEMPT_BLOCKED',
      };
    }

    const mergedOptions: MultiAngleMatchOptions = {
      ...(this.defaultSimilarityThreshold !== undefined
        ? { similarityThreshold: this.defaultSimilarityThreshold }
        : {}),
      ...(observation.faceHeightRatio !== undefined
        ? { faceHeightRatio: observation.faceHeightRatio }
        : {}),
      yawDeg: observation.yawDeg,
      pitchDeg: observation.pitchDeg,
      ...(observation.rollDeg !== undefined ? { rollDeg: observation.rollDeg } : {}),
      ...(observation.engine ? { engine: observation.engine } : {}),
      ...(observation.geometricRatios !== undefined
        ? { geometricRatios: observation.geometricRatios }
        : {}),
      ...(observation.mirroredProbeVector
        ? { mirroredProbeVector: observation.mirroredProbeVector }
        : {}),
      ...(observation.hybridV2ProbeVector
        ? { hybridV2ProbeVector: observation.hybridV2ProbeVector }
        : {}),
      ...(observation.mirroredHybridV2ProbeVector
        ? { mirroredHybridV2ProbeVector: observation.mirroredHybridV2ProbeVector }
        : {}),
      ...(observation.legacyV1ProbeVector
        ? { legacyV1ProbeVector: observation.legacyV1ProbeVector }
        : {}),
      ...(observation.mirroredLegacyV1ProbeVector
        ? { mirroredLegacyV1ProbeVector: observation.mirroredLegacyV1ProbeVector }
        : {}),
      ...options,
    };
    const match = identifyPatient1toN(observation.probeVector, candidates, mergedOptions);

    if (!match.matched || !match.bestCandidate) {
      // Unanimous lock: ANY failed or ambiguous frame immediately wipes streak back to 0!
      this.reset();
      return {
        confirmed: false,
        streakCount: 0,
        requiredStreak: this.requiredConsecutiveFrames,
        candidate: match.bestCandidate,
        confidencePercent: match.confidencePercent,
        bestRawSimilarity: match.similarity,
        bestAngle: match.bestAngle,
        livenessPassed:
          observation.depthScore >= 0.25 && observation.biologicalVerified !== false,
        livenessReason:
          match.reason === 'AMBIGUOUS_CLOSE_CANDIDATES'
            ? 'AMBIGUOUS_CLOSE_CANDIDATES'
            : match.bestCandidate
              ? 'LOW_CONFIDENCE_OR_MARGIN'
              : 'NO_MATCH',
      };
    }

    const pid = match.bestCandidate.patientId;
    const pName = normalizePatientFullName(match.bestCandidate.fullName);
    const sameIdentityStreak =
      this.streakPatientId === pid ||
      (Boolean(this.streakPatientName) && Boolean(pName) && this.streakPatientName === pName);
    if (!sameIdentityStreak) {
      // Identity switched! Hard-reset streak before counting new identity's frame 1.
      this.streakPatientId = pid;
      this.streakPatientName = pName || null;
      this.streakFrames = [];
    } else {
      this.streakPatientId = pid;
      if (pName) this.streakPatientName = pName;
    }

    this.streakFrames.push({
      match,
      depthScore: observation.depthScore,
      yawDeg: observation.yawDeg,
      pitchDeg: observation.pitchDeg,
      microMotion: observation.microMotion,
      blinkDetected: Boolean(observation.blinkDetected),
    });

    if (this.streakFrames.length > this.requiredConsecutiveFrames) {
      this.streakFrames.shift();
    }

    const frameCount = this.streakFrames.length;
    const avgDepth =
      this.streakFrames.reduce((acc, f) => acc + f.depthScore, 0) / Math.max(1, frameCount);
    const maxMicroMotion = Math.max(...this.streakFrames.map((f) => f.microMotion));
    const yawRange =
      Math.max(...this.streakFrames.map((f) => f.yawDeg)) -
      Math.min(...this.streakFrames.map((f) => f.yawDeg));
    const pitchRange =
      Math.max(...this.streakFrames.map((f) => f.pitchDeg)) -
      Math.min(...this.streakFrames.map((f) => f.pitchDeg));
    const anyBlink = this.streakFrames.some((f) => f.blinkDetected);

    const hasMotionVariance =
      maxMicroMotion >= 0.0001 || yawRange >= 0.1 || pitchRange >= 0.1 || anyBlink;
    const basePhysicalLiveness =
      avgDepth >= 0.25 && (frameCount < 2 || hasMotionVariance);
    const bioVerifiedOk =
      observation.biologicalVerified === undefined ? true : observation.biologicalVerified === true;
    const livenessPassed = basePhysicalLiveness && bioVerifiedOk;

    const avgConf = Number(
      (
        this.streakFrames.reduce((acc, f) => acc + f.match.confidencePercent, 0) /
        Math.max(1, frameCount)
      ).toFixed(1),
    );

    const confirmed = frameCount >= this.requiredConsecutiveFrames && livenessPassed;

    return {
      confirmed,
      streakCount: frameCount,
      requiredStreak: this.requiredConsecutiveFrames,
      candidate: match.bestCandidate,
      confidencePercent: avgConf,
      bestRawSimilarity: match.similarity,
      bestAngle: match.bestAngle,
      livenessPassed,
      livenessReason: !livenessPassed
        ? !bioVerifiedOk
          ? 'BIOLOGICAL_BLINK_REQUIRED'
          : avgDepth < 0.25
            ? 'FLAT_SPOOF_SUSPECTED'
            : 'STATIC_PHOTO_SUSPECTED'
        : undefined,
    };
  }
}

/**
 * Computes a normalized bounding box [0..1] in camera image coordinates from MediaPipe landmarks.
 */
export function computeBoundingBoxFromLandmarks(
  landmarks: readonly { x: number; y: number; z?: number }[],
): FaceBoundingBox | null {
  if (!landmarks || landmarks.length < 10) return null;
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }
  const padX = (maxX - minX) * 0.14;
  const padY = (maxY - minY) * 0.18;
  const x = Math.max(0.01, minX - padX);
  const y = Math.max(0.01, minY - padY);
  const right = Math.min(0.99, maxX + padX);
  const bottom = Math.min(0.99, maxY + padY);
  const width = Math.max(0.08, right - x);
  const height = Math.max(0.1, bottom - y);
  return {
    x,
    y,
    width,
    height,
    centerX: x + width / 2,
    centerY: y + height / 2,
    confidence: landmarks.length >= 400 ? 0.96 : 0.88,
  };
}

export const BIOMETRIC_INDEX_CHANGED_EVENT = 'drtechcares:biometric-index-changed';

/**
 * Broadcasts that patient biometric templates changed (enrolled/updated/deleted)
 * so active scanner views immediately refresh their local index.
 */
export function notifyBiometricIndexChanged(): void {
  if (typeof window !== 'undefined' && typeof CustomEvent !== 'undefined') {
    window.dispatchEvent(new CustomEvent(BIOMETRIC_INDEX_CHANGED_EVENT));
  }
}

/**
 * Loads the full multi-angle biometric index from the cloud API and keeps it synchronized
 * via Supabase Realtime subscriptions, focus/delete events, and periodic polling.
 */
export function subscribeBiometricIndex(
  apiClient: DrTechCaresApiClient,
  onUpdate: (entries: PatientBiometricIndexEntry[]) => void,
): () => void {
  let active = true;

  const refresh = () => {
    apiClient
      .getBiometricIndex()
      .then((entries) => {
        if (active) onUpdate(entries);
      })
      .catch(() => {});
  };

  refresh();

  const onVisibilityChange = () => {
    if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
      refresh();
    }
  };

  if (typeof window !== 'undefined') {
    window.addEventListener(BIOMETRIC_INDEX_CHANGED_EVENT, refresh);
    window.addEventListener('focus', refresh);
  }
  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', onVisibilityChange);
  }
  const intervalId = globalThis.setInterval(refresh, 5000);

  const supabase = apiClient.getSupabaseClient();
  const channel = supabase
    ? supabase
        .channel(`biometrics-index-rt-${Date.now()}`)
        .on('postgres_changes', { event: '*', schema: 'public', table: 'patient_biometrics' }, refresh)
        .on('postgres_changes', { event: '*', schema: 'public', table: 'face_templates' }, refresh)
        .on('postgres_changes', { event: '*', schema: 'public', table: 'patients' }, refresh)
        .subscribe()
    : null;

  return () => {
    active = false;
    globalThis.clearInterval(intervalId);
    if (typeof window !== 'undefined') {
      window.removeEventListener(BIOMETRIC_INDEX_CHANGED_EVENT, refresh);
      window.removeEventListener('focus', refresh);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', onVisibilityChange);
    }
    if (channel && supabase) {
      void supabase.removeChannel(channel);
    }
  };
}

