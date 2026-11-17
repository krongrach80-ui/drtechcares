import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DrTechCaresApiClient, PatientBiometricIndexEntry } from '@drtechcares/api-client';
import {
  BIOMETRIC_ENGINE_ID,
  BIOMETRIC_EXPECTED_DIM,
  BIOMETRIC_HUMAN_SIM_THRESHOLD,
  BIOMETRIC_MATCH_THRESHOLD,
  BIOMETRIC_REQUIRED_STREAK,
  BIOMETRIC_VAULT_STORAGE_KEY,
  LiveIdentityStreakTracker,
  buildBiometricProfileFromSamples,
  computeHumanDescriptorSimilarity,
  computePatientMatchScore,
  cosineSimilarityNormalized,
  evaluateLiveMatch1toN,
  isValidNeuralEmbedding,
  loadEnrolledPatientsWithBiometrics,
  normalizeVector,
  readLocalBiometricVault,
  removeProfileFromLocalBiometricVault,
  saveProfileToLocalBiometricVault,
  type EnrolledBiometricCandidate,
  type NeuralSampleInput,
} from '../lib/biometricEngine';

function makeDeterministicUnitVector(seed: number, dim: number = BIOMETRIC_EXPECTED_DIM): number[] {
  const vec = new Array<number>(dim);
  for (let i = 0; i < dim; i += 1) {
    const theta = (i + 1) * (seed * 1.37 + 0.19);
    vec[i] = Math.sin(theta) + 0.5 * Math.cos((i + 3) * (seed + 0.71));
  }
  return normalizeVector(vec);
}

function blendWithNoise(base: readonly number[], orthogonalSeed: number, noiseWeight: number): number[] {
  const noise = makeDeterministicUnitVector(orthogonalSeed, base.length);
  const out = base.map((v, i) => v + noiseWeight * noise[i]);
  return normalizeVector(out);
}

function makeCandidate(
  patientId: string,
  hn: string,
  fullName: string,
  seed: number,
  dim: number = BIOMETRIC_EXPECTED_DIM,
): EnrolledBiometricCandidate {
  const master = makeDeterministicUnitVector(seed, dim);
  return {
    patientId,
    hn,
    fullName,
    diagnosis: 'Stroke Rehab',
    assignedPtId: 'pt-01',
    nextAppointmentDate: '2026-12-01',
    framesCaptured: 120,
    qualityScore: 0.96,
    biometricEngine: dim === BIOMETRIC_EXPECTED_DIM ? BIOMETRIC_ENGINE_ID : undefined,
    dimension: dim,
    master_embedding: master,
    angle_embeddings: {
      center: master,
      left: blendWithNoise(master, seed + 100, 0.14),
      right: blendWithNoise(master, seed + 200, 0.14),
      up: blendWithNoise(master, seed + 300, 0.14),
      down: blendWithNoise(master, seed + 400, 0.14),
    },
    exemplar_embeddings: [master, blendWithNoise(master, seed + 50, 0.08)],
    sourceEntry: {
      patientId,
      hn,
      fullName,
      diagnosis: 'Stroke Rehab',
      assignedPtId: 'pt-01',
      nextAppointmentDate: '2026-12-01',
      framesCaptured: 120,
      qualityScore: 0.96,
      masterEmbedding: master,
      angleEmbeddings: {
        center: master,
        left: master,
        right: master,
        up: master,
        down: master,
      },
      exemplarEmbeddings: [master],
      biometricEngine: dim === BIOMETRIC_EXPECTED_DIM ? BIOMETRIC_ENGINE_ID : undefined,
    } as PatientBiometricIndexEntry,
  };
}

describe('Unified @vladmandic/human (1024-D) Biometric Engine', () => {
  const originalLocalStorage = globalThis.localStorage;
  const storageMap = new Map<string, string>();

  beforeEach(() => {
    storageMap.clear();
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: {
        getItem: (key: string) => storageMap.get(key) ?? null,
        setItem: (key: string, val: string) => {
          storageMap.set(key, val);
        },
        removeItem: (key: string) => {
          storageMap.delete(key);
        },
        clear: () => {
          storageMap.clear();
        },
        key: (idx: number) => Array.from(storageMap.keys())[idx] ?? null,
        get length() {
          return storageMap.size;
        },
      },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: originalLocalStorage,
    });
  });

  it('computes pure L2 normalization and calibrated Human descriptor similarity', () => {
    const raw = [3, 4, ...new Array(1022).fill(0)];
    const norm = normalizeVector(raw);
    expect(norm[0]).toBeCloseTo(0.6, 5);
    expect(norm[1]).toBeCloseTo(0.8, 5);
    expect(isValidNeuralEmbedding(norm)).toBe(true);
    expect(isValidNeuralEmbedding(new Array(1024).fill(0))).toBe(false);
    expect(isValidNeuralEmbedding([1, 2, 3])).toBe(false);

    const a = makeDeterministicUnitVector(11);
    const same = cosineSimilarityNormalized(a, a);
    expect(same).toBeCloseTo(1, 5);
    expect(computeHumanDescriptorSimilarity(a, a)).toBeGreaterThanOrEqual(0.99);

    // Construct an exact orthogonal basis vector b so cos(a, probe) is controllable
    const rawB = makeDeterministicUnitVector(77);
    const dotAB = cosineSimilarityNormalized(a, rawB);
    const orthoB = normalizeVector(rawB.map((v, i) => v - dotAB * a[i]));
    const probeAt062 = normalizeVector(a.map((v, i) => 0.62 * v + Math.sqrt(1 - 0.62 * 0.62) * orthoB[i]));
    expect(cosineSimilarityNormalized(a, probeAt062)).toBeCloseTo(0.62, 3);
    expect(computeHumanDescriptorSimilarity(a, probeAt062)).toBeGreaterThanOrEqual(
      BIOMETRIC_HUMAN_SIM_THRESHOLD,
    );

    const unrelated = orthoB;
    expect(cosineSimilarityNormalized(a, unrelated)).toBeCloseTo(0, 4);
    expect(computeHumanDescriptorSimilarity(a, unrelated)).toBeLessThan(0.5);
  });

  it('builds 5-angle profile and Top-10 sharp exemplars cleanly without fallbacks', () => {
    const base = makeDeterministicUnitVector(5);
    const buckets: NeuralSampleInput['bucket'][] = ['center', 'left', 'right', 'up', 'down'];
    const samples: NeuralSampleInput[] = [];

    for (let i = 0; i < 25; i += 1) {
      const bucket = buckets[i % buckets.length];
      samples.push({
        bucket,
        sharpness: 60 + i * 2,
        yawDeg: bucket === 'left' ? -15 : bucket === 'right' ? 15 : 0,
        pitchDeg: bucket === 'up' ? -10 : bucket === 'down' ? 10 : 0,
        embedding: blendWithNoise(base, 500 + i, 0.05),
      });
    }

    const profile = buildBiometricProfileFromSamples(samples, {
      patientId: 'pat-build-1',
      hn: 'HN-99001',
      fullName: 'Somchai K.',
      qualityScore: 0.97,
    });

    expect(profile).not.toBeNull();
    expect(profile?.engine).toBe(BIOMETRIC_ENGINE_ID);
    expect(profile?.dimension).toBe(1024);
    expect(profile?.exemplar_embeddings.length).toBeGreaterThanOrEqual(5);
    expect(profile?.exemplar_embeddings.length).toBeLessThanOrEqual(10);
    expect(cosineSimilarityNormalized(profile!.master_embedding, base)).toBeGreaterThan(0.97);
    expect(profile?.angle_embeddings.left).toHaveLength(1024);
    expect(profile?.angle_embeddings.right).toHaveLength(1024);
  });

  it('enforces dimension filtering, 1:N exemplar scoring, stranger rejection, and ambiguity blocking', () => {
    const patA = makeCandidate('pat-a', 'HN-101', 'Patient Alpha', 1);
    const patB = makeCandidate('pat-b', 'HN-102', 'Patient Beta', 2);
    const legacy512 = makeCandidate('pat-legacy', 'HN-099', 'Legacy Patient', 3, 512);
    const pool = [patA, patB, legacy512];

    // Live frame of Patient Alpha under mild lighting/pose drift -> matches patA cleanly
    const liveAlpha = blendWithNoise(patA.exemplar_embeddings[1], 999, 0.25);
    const resAlpha = evaluateLiveMatch1toN(liveAlpha, pool);
    expect(resAlpha.matched).toBe(true);
    expect(resAlpha.topCandidate?.patientId).toBe('pat-a');
    expect(resAlpha.topSimilarity).toBeGreaterThanOrEqual(BIOMETRIC_MATCH_THRESHOLD);
    expect(resAlpha.validAccounts).toBe(2); // ignores 512-D candidate automatically
    expect(resAlpha.totalAccounts).toBe(3);

    // Stranger -> zero false acceptance
    const stranger = makeDeterministicUnitVector(444);
    const resStranger = evaluateLiveMatch1toN(stranger, pool);
    expect(resStranger.matched).toBe(false);
    expect(resStranger.reason).toBe('BELOW_THRESHOLD');

    // Ambiguous mid-point probe close to both distinct patients -> blocked!
    const tieProbe = normalizeVector(
      patA.master_embedding.map((v, i) => 0.72 * v + 0.7 * patB.master_embedding[i]),
    );
    const scoreA = computePatientMatchScore(tieProbe, patA);
    const scoreB = computePatientMatchScore(tieProbe, patB);
    expect(Math.abs(scoreA.bestSim - scoreB.bestSim)).toBeLessThan(0.06);
    const resTie = evaluateLiveMatch1toN(tieProbe, [patA, patB]);
    expect(resTie.matched).toBe(false);
    expect(resTie.reason).toBe('AMBIGUOUS_CLOSE_CANDIDATES');
  });

  it('requires 3/3 consecutive matching live frames and resets on spoof or flip', () => {
    const patA = makeCandidate('pat-a', 'HN-101', 'Patient Alpha', 1);
    const patB = makeCandidate('pat-b', 'HN-102', 'Patient Beta', 2);
    const tracker = new LiveIdentityStreakTracker(BIOMETRIC_REQUIRED_STREAK);

    const evalA = evaluateLiveMatch1toN(blendWithNoise(patA.master_embedding, 10, 0.1), [patA, patB]);
    const evalB = evaluateLiveMatch1toN(blendWithNoise(patB.master_embedding, 20, 0.1), [patA, patB]);

    const s1 = tracker.step(evalA, { biologicalVerified: true });
    expect(s1.confirmed).toBe(false);
    expect(s1.streak).toBe(1);

    // Identity flip resets streak immediately
    const flip = tracker.step(evalB, { biologicalVerified: true });
    expect(flip.confirmed).toBe(false);
    expect(flip.streak).toBe(1);
    expect(flip.candidate?.patientId).toBe('pat-b');

    // Spoof clears streak to 0
    const spoofed = tracker.step(evalA, { biologicalVerified: true, spoofDetected: true });
    expect(spoofed.confirmed).toBe(false);
    expect(spoofed.streak).toBe(0);

    // 3/3 clean frames -> confirmed
    expect(tracker.step(evalA, { biologicalVerified: true }).confirmed).toBe(false);
    expect(tracker.step(evalA, { biologicalVerified: true }).confirmed).toBe(false);
    const third = tracker.step(evalA, { biologicalVerified: true });
    expect(third.confirmed).toBe(true);
    expect(third.streak).toBe(3);
    expect(third.candidate?.patientId).toBe('pat-a');
  });

  it('persists, merges, and filters legacy vs 1024-D accounts in drtech_biometric_vault_v3', async () => {
    const profile1024 = buildBiometricProfileFromSamples(
      [
        {
          bucket: 'center',
          sharpness: 110,
          yawDeg: 0,
          pitchDeg: 0,
          embedding: makeDeterministicUnitVector(7),
        },
      ],
      {
        patientId: 'pat-new-1024',
        hn: 'HN-77701',
        fullName: 'New User 1024',
      },
    )!;

    saveProfileToLocalBiometricVault(profile1024);
    expect(storageMap.has(BIOMETRIC_VAULT_STORAGE_KEY)).toBe(true);
    expect(readLocalBiometricVault()).toHaveLength(1);

    const legacyCloudEntry = makeCandidate('pat-old-512', 'HN-11111', 'Old User 512', 99, 512)
      .sourceEntry;
    const mockApiClient = {
      getBiometricIndex: vi.fn(async () => [legacyCloudEntry]),
      getSupabaseClient: vi.fn(() => null),
    } as unknown as DrTechCaresApiClient;

    const loaded = await loadEnrolledPatientsWithBiometrics(mockApiClient);
    expect(loaded.totalAccounts).toBe(2);
    expect(loaded.validAccounts).toBe(1);
    expect(loaded.legacyAccountsCount).toBe(1);
    expect(loaded.compatibleCandidates[0]?.patientId).toBe('pat-new-1024');

    removeProfileFromLocalBiometricVault('pat-new-1024');
    expect(readLocalBiometricVault()).toHaveLength(0);
  });
});

