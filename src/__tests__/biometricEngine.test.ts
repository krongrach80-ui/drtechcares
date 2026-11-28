import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DrTechCaresApiClient, PatientBiometricIndexEntry } from '@drtechcares/api-client';
import {
  IMPOSTER_COHORT_VECTORS_1024,
  computeCraniofacialRatios12,
  computePatchZnccScore,
  extractFacePatchSignature384FromRgba,
} from '@drtechcares/smart-biometric-core';
import {
  BIOMETRIC_ENGINE_ID,
  BIOMETRIC_EXPECTED_DIM,
  BIOMETRIC_FINAL_SCORE_THRESHOLD,
  BIOMETRIC_HUMAN_SIM_THRESHOLD,
  BIOMETRIC_MARGIN_THRESHOLD,
  BIOMETRIC_MATCH_THRESHOLD,
  BIOMETRIC_REQUIRED_STREAK,
  BIOMETRIC_VAULT_STORAGE_KEY,
  HUMAN_BIOMETRIC_CONFIG,
  LiveIdentityStreakTracker,
  buildBiometricProfileFromSamples,
  computeHumanDescriptorSimilarity,
  computePatientMatchScore,
  cosineSimilarityNormalized,
  evaluateLiveMatch1toN,
  fetchLiveBiometricsFromSupabase,
  isValidNeuralEmbedding,
  loadEnrolledPatientsWithBiometrics,
  normalizeImageDataLuminance,
  normalizeVector,
  readLocalBiometricVault,
  removeProfileFromLocalBiometricVault,
  saveProfileToLocalBiometricVault,
  syncPatientBiometricEverywhere,
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
    diagnosis: 'Elderly Rehab',
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
      diagnosis: 'Elderly Rehab',
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

    // Raw cosine at 0.62 stays strictly unboosted (~0.62) and MUST NOT clear >= 0.76 threshold
    const rawB = makeDeterministicUnitVector(77);
    const dotAB = cosineSimilarityNormalized(a, rawB);
    const orthoB = normalizeVector(rawB.map((v, i) => v - dotAB * a[i]));
    const probeAt062 = normalizeVector(a.map((v, i) => 0.62 * v + Math.sqrt(1 - 0.62 * 0.62) * orthoB[i]));
    expect(cosineSimilarityNormalized(a, probeAt062)).toBeCloseTo(0.62, 3);
    expect(computeHumanDescriptorSimilarity(a, probeAt062)).toBeCloseTo(0.62, 3);
    expect(computeHumanDescriptorSimilarity(a, probeAt062)).toBeLessThan(BIOMETRIC_HUMAN_SIM_THRESHOLD);

    // Probe at 0.78 is BELOW strict >= 0.84 threshold and MUST be rejected
    const probeAt078 = normalizeVector(a.map((v, i) => 0.78 * v + Math.sqrt(1 - 0.78 * 0.78) * orthoB[i]));
    expect(computeHumanDescriptorSimilarity(a, probeAt078)).toBeCloseTo(0.78, 3);
    expect(computeHumanDescriptorSimilarity(a, probeAt078)).toBeLessThan(
      BIOMETRIC_HUMAN_SIM_THRESHOLD,
    );

    // Genuine probe at 0.90 passes strict >= 0.84 threshold cleanly
    const probeAt090 = normalizeVector(a.map((v, i) => 0.9 * v + Math.sqrt(1 - 0.9 * 0.9) * orthoB[i]));
    expect(computeHumanDescriptorSimilarity(a, probeAt090)).toBeCloseTo(0.9, 3);
    expect(computeHumanDescriptorSimilarity(a, probeAt090)).toBeGreaterThanOrEqual(
      BIOMETRIC_HUMAN_SIM_THRESHOLD,
    );

    const unrelated = orthoB;
    expect(cosineSimilarityNormalized(a, unrelated)).toBeCloseTo(0, 4);
    expect(computeHumanDescriptorSimilarity(a, unrelated)).toBeLessThan(0.1);
  });

  it('builds 5-angle profile and 15 sharp exemplars distributed across all 5 angles cleanly without fallbacks', () => {
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
    expect(profile?.exemplar_embeddings.length).toBe(15);
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

    // Stranger -> zero false acceptance and zero candidate name leak
    const stranger = makeDeterministicUnitVector(444);
    const resStranger = evaluateLiveMatch1toN(stranger, pool);
    expect(resStranger.matched).toBe(false);
    expect(resStranger.topCandidate).toBeNull();
    expect(resStranger.reason).toBe('BELOW_THRESHOLD');

    // Open-set sub-threshold stranger (~0.52 cosine) -> MUST be rejected cleanly with zero name leak
    const rawOrthoBase = makeDeterministicUnitVector(555);
    const dBase = cosineSimilarityNormalized(patA.master_embedding, rawOrthoBase);
    const orthoUnit = normalizeVector(rawOrthoBase.map((v, i) => v - dBase * patA.master_embedding[i]));
    const mediumStranger052 = normalizeVector(
      patA.master_embedding.map((v, i) => 0.52 * v + Math.sqrt(1 - 0.52 * 0.52) * orthoUnit[i]),
    );
    const resMedium = evaluateLiveMatch1toN(mediumStranger052, [patA, patB]);
    expect(resMedium.matched).toBe(false);
    expect(resMedium.topCandidate).toBeNull();
    expect(resMedium.confidencePercent).toBeLessThan(60);
    expect(resMedium.reason).toBe('BELOW_THRESHOLD');

    // Sub-threshold stranger at ~0.68 -> MUST be rejected (strict floor = 0.84 / centered 0.80 / L2 <= 0.56)
    const strangerAt068 = normalizeVector(
      patA.master_embedding.map((v, i) => 0.68 * v + Math.sqrt(1 - 0.68 * 0.68) * orthoUnit[i]),
    );
    const resRejected068 = evaluateLiveMatch1toN(strangerAt068, [patA, patB]);
    expect(resRejected068.matched).toBe(false);
    expect(resRejected068.topCandidate).toBeNull();

    // Genuine probe at ~0.92 matching master & center clears all Gate 1+3 thresholds cleanly
    const genuineAt092 = normalizeVector(
      patA.master_embedding.map((v, i) => 0.92 * v + Math.sqrt(1 - 0.92 * 0.92) * orthoUnit[i]),
    );
    const resCalibrated092 = evaluateLiveMatch1toN(genuineAt092, [patA, patB]);
    expect(resCalibrated092.matched).toBe(true);
    expect(resCalibrated092.topCandidate?.patientId).toBe('pat-a');

    // Ambiguous mid-point probe clearing >= 0.76 for TWO correlated candidates within margin -> blocked!
    const corrPatBVec = normalizeVector(
      patA.master_embedding.map((v, i) => 0.6 * v + 0.8 * orthoUnit[i]),
    );
    const corrPatB = {
      ...patB,
      master_embedding: corrPatBVec,
      angle_embeddings: { center: corrPatBVec, left: corrPatBVec, right: corrPatBVec, up: corrPatBVec, down: corrPatBVec },
      exemplar_embeddings: [corrPatBVec, corrPatBVec, corrPatBVec],
    };
    const tieProbe = normalizeVector(
      patA.master_embedding.map((v, i) => 0.8944 * v + 0.4472 * orthoUnit[i]),
    );
    const scoreA = computePatientMatchScore(tieProbe, patA);
    const scoreB = computePatientMatchScore(tieProbe, corrPatB);
    expect(scoreA.bestSim).toBeGreaterThanOrEqual(0.76);
    expect(scoreB.bestSim).toBeGreaterThanOrEqual(0.76);
    expect(Math.abs(scoreA.finalScore - scoreB.finalScore)).toBeLessThan(BIOMETRIC_MARGIN_THRESHOLD);
    const resTie = evaluateLiveMatch1toN(tieProbe, [patA, corrPatB]);
    expect(resTie.matched).toBe(false);
    expect(resTie.topCandidate).toBeNull();
    expect(resTie.reason).toBe('AMBIGUOUS_CLOSE_CANDIDATES');
  });

  it('requires 6/6 consecutive matching live frames with low variance and resets on spoof, flip, or below-0.84 dip', () => {
    const patA = makeCandidate('pat-a', 'HN-101', 'Patient Alpha', 1);
    const patB = makeCandidate('pat-b', 'HN-102', 'Patient Beta', 2);
    const tracker = new LiveIdentityStreakTracker(BIOMETRIC_REQUIRED_STREAK);
    expect(BIOMETRIC_REQUIRED_STREAK).toBe(6);

    const evalA = evaluateLiveMatch1toN(blendWithNoise(patA.master_embedding, 10, 0.05), [patA, patB]);
    const evalAHigh = evaluateLiveMatch1toN(patA.master_embedding, [patA, patB]);
    const evalB = evaluateLiveMatch1toN(blendWithNoise(patB.master_embedding, 20, 0.05), [patA, patB]);

    const s1 = tracker.step(evalA, { biologicalVerified: true });
    expect(s1.confirmed).toBe(false);
    expect(s1.streak).toBe(1);
    // Strict zero-leak rule: candidate MUST stay null until 6/6 completes!
    expect(s1.candidate).toBeNull();

    // Identity flip resets streak back to 1 (still hiding candidate until 6/6)
    const flip = tracker.step(evalB, { biologicalVerified: true });
    expect(flip.confirmed).toBe(false);
    expect(flip.streak).toBe(1);
    expect(flip.candidate).toBeNull();

    // Spoof clears streak to 0
    const spoofed = tracker.step(evalA, { biologicalVerified: true, spoofDetected: true });
    expect(spoofed.confirmed).toBe(false);
    expect(spoofed.streak).toBe(0);
    expect(spoofed.candidate).toBeNull();

    // Large score variance (>= 0.08) between frame 1 and frame 2 resets streak to 0/6
    const rawDiffVec = makeDeterministicUnitVector(99);
    const dotD = cosineSimilarityNormalized(patA.master_embedding, rawDiffVec);
    const orthoD = normalizeVector(rawDiffVec.map((v, i) => v - dotD * patA.master_embedding[i]));
    const frameAt088 = evaluateLiveMatch1toN(
      normalizeVector(patA.master_embedding.map((v, i) => 0.88 * v + Math.sqrt(1 - 0.88 * 0.88) * orthoD[i])),
      [patA, patB],
    );
    expect(tracker.step(frameAt088, { biologicalVerified: true }).streak).toBe(1);
    const highVarFlip = tracker.step(evalAHigh, { biologicalVerified: true });
    expect(highVarFlip.confirmed).toBe(false);
    expect(highVarFlip.streak).toBe(0);

    // 6/6 clean low-variance frames -> confirmed!
    for (let i = 1; i <= 5; i += 1) {
      const pending = tracker.step(evalA, { biologicalVerified: true });
      expect(pending.confirmed).toBe(false);
      expect(pending.streak).toBe(i);
      expect(pending.candidate).toBeNull();
    }
    const sixth = tracker.step(evalA, { biologicalVerified: true });
    expect(sixth.confirmed).toBe(true);
    expect(sixth.streak).toBe(6);
    expect(sixth.candidate?.patientId).toBe('pat-a');
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

  it('enables high-speed facial descriptor config without Iris or filter overhead', () => {
    expect(HUMAN_BIOMETRIC_CONFIG.filter?.enabled).toBe(false);
    expect(HUMAN_BIOMETRIC_CONFIG.face?.detector?.maxDetected).toBe(1);
    expect(HUMAN_BIOMETRIC_CONFIG.face?.detector?.rotation).toBe(true);
    expect(HUMAN_BIOMETRIC_CONFIG.face?.mesh?.enabled).toBe(true);
    expect(HUMAN_BIOMETRIC_CONFIG.face?.iris?.enabled).toBe(false);
    expect(HUMAN_BIOMETRIC_CONFIG.face?.description?.enabled).toBe(true);
  });

  it('normalizes dim vs backlit images to target Mean Luminance = 128', () => {
    const mkSyntheticImage = (offset: number, scale: number): ImageData => {
      const w = 16;
      const h = 16;
      const data = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i += 1) {
        const base = ((i % w) / w) * 50 + (Math.floor(i / w) / h) * 30;
        const v = Math.max(0, Math.min(255, Math.round(offset + base * scale)));
        const idx = i * 4;
        data[idx] = v;
        data[idx + 1] = v;
        data[idx + 2] = v;
        data[idx + 3] = 255;
      }
      return { data, width: w, height: h, colorSpace: 'srgb' } as ImageData;
    };

    const dimImg = normalizeImageDataLuminance(mkSyntheticImage(18, 0.7), 128);
    const brightImg = normalizeImageDataLuminance(mkSyntheticImage(120, 1.3), 128);

    const meanLuma = (img: ImageData) => {
      let sum = 0;
      const n = img.width * img.height;
      for (let i = 0; i < img.data.length; i += 4) {
        sum += 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
      }
      return sum / n;
    };

    expect(Math.abs(meanLuma(dimImg) - 128)).toBeLessThan(3);
    expect(Math.abs(meanLuma(brightImg) - 128)).toBeLessThan(3);
  });

  it('enforces Read-After-Write verification on Supabase persistence and loads cross-day when localStorage is empty', async () => {
    const dbPatients = new Map<string, Record<string, unknown>>();
    const dbBiometrics = new Map<string, Record<string, unknown>>();

    const makeMockSupabase = (corruptReadAfterWrite = false) => ({
      from: (table: string) => {
        if (table === 'patients') {
          return {
            select: () => ({
              eq: (_col: string, val: string) => ({
                maybeSingle: async () => ({
                  data: corruptReadAfterWrite ? null : dbPatients.get(val) ?? null,
                  error: null,
                }),
              }),
              then: (res: (v: { data: Record<string, unknown>[]; error: null }) => void) =>
                Promise.resolve({
                  data: Array.from(dbPatients.values()).map((r) => ({
                    ...r,
                    patient_biometrics: dbBiometrics.get(String(r.id)) ?? null,
                  })),
                  error: null,
                }).then(res),
            }),
            upsert: (payload: Record<string, unknown>) => ({
              select: () => {
                dbPatients.set(String(payload.id), payload);
                return Promise.resolve({ data: [{ id: payload.id }], error: null });
              },
            }),
            update: (payload: Record<string, unknown>) => ({
              eq: (_c: string, id: string) => ({
                select: () => {
                  const prev = dbPatients.get(id) ?? { id };
                  dbPatients.set(id, { ...prev, ...payload });
                  return Promise.resolve({ data: [{ id }], error: null });
                },
              }),
            }),
          };
        }
        return {
          select: () => ({
            eq: (_col: string, val: string) => ({
              maybeSingle: async () => ({
                data: corruptReadAfterWrite
                  ? { patient_id: val, master_embedding: [], embedding_dim: 0 }
                  : dbBiometrics.get(val) ?? null,
                error: null,
              }),
            }),
          }),
          upsert: async (payload: Record<string, unknown>) => {
            dbBiometrics.set(String(payload.patient_id), payload);
            return { data: payload, error: null };
          },
        };
      },
    });

    const masterVec = makeDeterministicUnitVector(101);
    const patientStub = {
      id: 'pat-sb-persist-1',
      hn: 'HN-88001',
      fullName: 'คุณวิชัย สุขภาพดี',
      age: 62,
      phone: '0811112222',
      diagnosis: 'Elderly Rehab',
      registrationStatus: 'active' as const,
      assignedPtId: 'pt-01',
      delegatedToPtId: null,
      delegationNote: null,
      nextAppointmentDate: '2026-12-01',
      enrolledAt: '2026-12-01T00:00:00.000Z',
    };
    const angles = {
      center: masterVec,
      left: blendWithNoise(masterVec, 201, 0.08),
      right: blendWithNoise(masterVec, 202, 0.08),
      up: blendWithNoise(masterVec, 203, 0.08),
      down: blendWithNoise(masterVec, 204, 0.08),
    };

    // 1. Failing Read-After-Write MUST throw instead of silently succeeding
    const failApiClient = {
      getSupabaseClient: () => makeMockSupabase(true),
      getBiometricIndex: async () => [],
    } as unknown as DrTechCaresApiClient;
    await expect(
      syncPatientBiometricEverywhere({
        apiClient: failApiClient,
        patient: patientStub,
        masterEmbedding: masterVec,
        angleEmbeddings: angles,
        exemplarEmbeddings: [masterVec],
        requireSupabase: true,
      }),
    ).rejects.toThrow(/Read-After-Write/i);

    // 2. Successful Supabase enrollment writes & verifies permanently
    const okApiClient = {
      getSupabaseClient: () => makeMockSupabase(false),
      getBiometricIndex: async () => [],
    } as unknown as DrTechCaresApiClient;
    const saved = await syncPatientBiometricEverywhere({
      apiClient: okApiClient,
      patient: patientStub,
      masterEmbedding: masterVec,
      angleEmbeddings: angles,
      exemplarEmbeddings: [masterVec, angles.left, angles.right],
      requireSupabase: true,
    });
    expect(saved?.dimension).toBe(1024);

    // 3. Simulate next-day reload / new Kiosk device: wipe localStorage completely
    storageMap.clear();
    expect(readLocalBiometricVault()).toHaveLength(0);

    const liveSynced = await fetchLiveBiometricsFromSupabase(okApiClient);
    expect(liveSynced.syncedFromSupabase).toBe(true);
    expect(liveSynced.validAccounts).toBe(1);
    expect(liveSynced.compatibleCandidates[0]?.patientId).toBe('pat-sb-persist-1');
    const matchRes = evaluateLiveMatch1toN(
      blendWithNoise(masterVec, 555, 0.12),
      liveSynced.compatibleCandidates,
    );
    expect(matchRes.matched).toBe(true);
    expect(matchRes.topFinalScore).toBeGreaterThanOrEqual(BIOMETRIC_FINAL_SCORE_THRESHOLD);
  });

  it('purges deleted zombie vectors from all localStorage keys and ignores orphaned patient_biometrics', async () => {
    const zombieVec = makeDeterministicUnitVector(707);
    const activeVec = makeDeterministicUnitVector(808);
    const zombieProfile = buildBiometricProfileFromSamples(
      [{ bucket: 'center', sharpness: 120, yawDeg: 0, pitchDeg: 0, embedding: zombieVec }],
      { patientId: 'pat-zombie-1', hn: 'HN-ZOMBIE-1', fullName: 'Deleted Somchai' },
    )!;
    const activeProfile = buildBiometricProfileFromSamples(
      [{ bucket: 'center', sharpness: 120, yawDeg: 0, pitchDeg: 0, embedding: activeVec }],
      { patientId: 'pat-active-1', hn: 'HN-ACTIVE-1', fullName: 'Active Waranya' },
    )!;

    saveProfileToLocalBiometricVault(zombieProfile);
    saveProfileToLocalBiometricVault(activeProfile);
    storageMap.set(
      'drtech_patients',
      JSON.stringify([
        { id: 'pat-zombie-1', hn: 'HN-ZOMBIE-1', fullName: 'Deleted Somchai' },
        { id: 'pat-active-1', hn: 'HN-ACTIVE-1', fullName: 'Active Waranya' },
      ]),
    );
    storageMap.set(
      'drtech_face_templates',
      JSON.stringify([
        { patientId: 'pat-zombie-1', hn: 'HN-ZOMBIE-1', masterEmbedding: zombieVec },
        { patientId: 'pat-active-1', hn: 'HN-ACTIVE-1', masterEmbedding: activeVec },
      ]),
    );

    const mockSupabase = {
      from: (table: string) => ({
        select: async (cols = '*') => {
          if (cols.includes('patient_biometrics')) {
            return { data: null, error: { message: 'Join skipped in mock to exercise separate query' } };
          }
          if (table === 'patients') {
            return {
              data: [{ id: 'pat-active-1', hn: 'HN-ACTIVE-1', full_name: 'Active Waranya', registration_status: 'active' }],
              error: null,
            };
          }
          return {
            data: [
              // Orphaned deleted vector still lingering in DB table — MUST NOT resurrect!
              {
                patient_id: 'pat-zombie-1',
                patient_hn: 'HN-ZOMBIE-1',
                full_name: 'Deleted Somchai',
                biometric_engine: BIOMETRIC_ENGINE_ID,
                dimension: 1024,
                master_embedding: zombieVec,
                angle_embeddings: { center: zombieVec },
                exemplar_embeddings: [zombieVec],
                quality_score: 0.99,
              },
              {
                patient_id: 'pat-active-1',
                patient_hn: 'HN-ACTIVE-1',
                full_name: 'Active Waranya',
                biometric_engine: BIOMETRIC_ENGINE_ID,
                dimension: 1024,
                master_embedding: activeVec,
                angle_embeddings: { center: activeVec },
                exemplar_embeddings: [activeVec],
                quality_score: 0.99,
              },
            ],
            error: null,
          };
        },
      }),
    };

    const strictApiClient = {
      getSupabaseClient: () => mockSupabase,
      getBiometricIndexWithRoster: async () => ({
        index: [],
        activePatients: [{ id: 'pat-active-1', hn: 'HN-ACTIVE-1', fullName: 'Active Waranya' }],
        deletedIds: ['pat-zombie-1', 'HN-ZOMBIE-1'],
      }),
      getBiometricIndex: async () => [],
    } as unknown as DrTechCaresApiClient;

    const synced = await loadEnrolledPatientsWithBiometrics(strictApiClient);
    expect(synced.compatibleCandidates.map((c) => c.patientId)).toEqual(['pat-active-1']);

    // Deleted zombie record MUST be purged from every localStorage key immediately
    expect(readLocalBiometricVault().map((v) => v.patientId)).toEqual(['pat-active-1']);
    const remainingPatients = JSON.parse(storageMap.get('drtech_patients') ?? '[]') as { id: string }[];
    expect(remainingPatients.map((p) => p.id)).toEqual(['pat-active-1']);
    const remainingTemplates = JSON.parse(storageMap.get('drtech_face_templates') ?? '[]') as { patientId: string }[];
    expect(remainingTemplates.map((t) => t.patientId)).toEqual(['pat-active-1']);
  });

  it('enforces Triple-Gate zero false-accept: rejects 0.70 strangers in 1-patient & 3-patient DBs while passing genuine +/-2% probes via ZNCC + Craniofacial + Cohort', () => {
    const mkRgbaFace = (seed: number, brightnessBias = 0, contrastScale = 1, invert = false) => {
      const w = 32;
      const h = 32;
      const data = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          const idx = (y * w + x) * 4;
          const wave =
            Math.sin((x + 1) * (0.25 + seed * 0.13)) * 55 +
            Math.cos((y + 1) * (0.31 + seed * 0.07)) * 45 +
            128;
          const val = invert ? 255 - wave : wave * contrastScale + brightnessBias;
          const clamped = Math.max(0, Math.min(255, Math.round(val)));
          data[idx] = clamped;
          data[idx + 1] = Math.max(0, Math.min(255, clamped + (seed % 9) - 4));
          data[idx + 2] = Math.max(0, Math.min(255, clamped - (seed % 7) + 3));
          data[idx + 3] = 255;
        }
      }
      return { width: w, height: h, data };
    };

    const mkLms = (jitter = 0) => {
      const lms = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
      lms[33] = { x: 0.42 + jitter, y: 0.42, z: 0 };
      lms[133] = { x: 0.46, y: 0.42, z: 0 };
      lms[263] = { x: 0.58 - jitter, y: 0.42, z: 0 };
      lms[362] = { x: 0.54, y: 0.42, z: 0 };
      lms[168] = { x: 0.5, y: 0.41, z: -0.02 };
      lms[10] = { x: 0.5, y: 0.28, z: 0 };
      lms[1] = { x: 0.5, y: 0.51, z: -0.04 };
      lms[61] = { x: 0.45, y: 0.62, z: 0 };
      lms[291] = { x: 0.55, y: 0.62, z: 0 };
      lms[152] = { x: 0.5, y: 0.74, z: 0 };
      lms[234] = { x: 0.34, y: 0.48, z: 0 };
      lms[454] = { x: 0.66, y: 0.48, z: 0 };
      lms[172] = { x: 0.38, y: 0.63, z: 0 };
      lms[397] = { x: 0.62, y: 0.63, z: 0 };
      return lms;
    };

    const enrolledPatch = extractFacePatchSignature384FromRgba(mkRgbaFace(3, 0, 1));
    const genuineShiftedLightPatch = extractFacePatchSignature384FromRgba(mkRgbaFace(3, 38, 1.35));
    const strangerPatch = extractFacePatchSignature384FromRgba(mkRgbaFace(19, 0, 1, true));
    expect(enrolledPatch).not.toBeNull();
    expect(genuineShiftedLightPatch).not.toBeNull();
    expect(strangerPatch).not.toBeNull();
    expect(computePatchZnccScore(enrolledPatch!, genuineShiftedLightPatch!)).toBeGreaterThanOrEqual(0.76);
    expect(computePatchZnccScore(enrolledPatch!, strangerPatch!)).toBeLessThan(0.76);

    const enrolledCranio = computeCraniofacialRatios12(mkLms(0));
    const genuineCranio = computeCraniofacialRatios12(mkLms(0.002));
    expect(enrolledCranio).toBeDefined();
    expect(genuineCranio).toBeDefined();

    const makeTriplePatient = (id: string, hn: string, name: string, seed: number, patch: number[], cranio: number[]) => ({
      ...makeCandidate(id, hn, name, seed),
      geometricRatios: {
        eyeToNoseRatio: 1.1,
        faceWidthToUpperHeight: 1.2,
        mouthToEyeRatio: 0.85,
        noseToChinRatio: 0.9,
        appearancePatch384: patch,
        craniofacialRatios12: cranio,
      },
    });

    const pat1 = makeTriplePatient('pat-only-1', 'HN-ONLY-1', 'Somchai Only', 7, enrolledPatch!, enrolledCranio!);
    const pat2 = makeTriplePatient('pat-two-2', 'HN-TWO-2', 'Waranya Two', 17, strangerPatch!, enrolledCranio!);
    const pat3 = makeTriplePatient('pat-three-3', 'HN-THREE-3', 'Nida Three', 29, strangerPatch!, enrolledCranio!);

    // 1) 1-patient DB vs 0.70 similarity stranger -> MUST return matched: false, topCandidate: null
    const rawO = makeDeterministicUnitVector(101);
    const dotO = cosineSimilarityNormalized(pat1.master_embedding, rawO);
    const orthoO = normalizeVector(rawO.map((v, i) => v - dotO * pat1.master_embedding[i]));
    const stranger070 = normalizeVector(
      pat1.master_embedding.map((v, i) => 0.7 * v + Math.sqrt(1 - 0.7 * 0.7) * orthoO[i]),
    );
    const singleStrangerRes = evaluateLiveMatch1toN(stranger070, [pat1], [], undefined, 'center', {
      eyeToNoseRatio: 1.1,
      faceWidthToUpperHeight: 1.2,
      mouthToEyeRatio: 0.85,
      noseToChinRatio: 0.9,
      appearancePatch384: strangerPatch!,
      craniofacialRatios12: genuineCranio!,
    });
    expect(singleStrangerRes.matched).toBe(false);
    expect(singleStrangerRes.topCandidate).toBeNull();

    // 2) 3-patient DB vs unregistered stranger & imposter cohort member -> matched: false
    const cohortProbe = IMPOSTER_COHORT_VECTORS_1024[0];
    const multiStrangerRes = evaluateLiveMatch1toN(cohortProbe, [pat1, pat2, pat3]);
    expect(multiStrangerRes.matched).toBe(false);
    expect(multiStrangerRes.topCandidate).toBeNull();

    // 3) Stranger wearing fake identical vector (0.94) but wrong Gate-2 facial patch ZNCC (< 0.76) -> REJECTED!
    const fakePatchProbe = blendWithNoise(pat1.master_embedding, 88, 0.02);
    const failedZnccRes = evaluateLiveMatch1toN(fakePatchProbe, [pat1], [], undefined, 'center', {
      eyeToNoseRatio: 1.1,
      faceWidthToUpperHeight: 1.2,
      mouthToEyeRatio: 0.85,
      noseToChinRatio: 0.9,
      appearancePatch384: strangerPatch!,
      craniofacialRatios12: genuineCranio!,
    });
    expect(failedZnccRes.matched).toBe(false);
    expect(failedZnccRes.topCandidate).toBeNull();

    // 4) Genuine user (+/-2% realistic noise + brightness-shifted patch) passes ALL 3 Gates + 6/6 consecutive streak
    const genuineProbe = blendWithNoise(pat1.master_embedding, 77, 0.02);
    const genuineRes = evaluateLiveMatch1toN(genuineProbe, [pat1, pat2, pat3], [], undefined, 'center', {
      eyeToNoseRatio: 1.1,
      faceWidthToUpperHeight: 1.2,
      mouthToEyeRatio: 0.85,
      noseToChinRatio: 0.9,
      appearancePatch384: genuineShiftedLightPatch!,
      craniofacialRatios12: genuineCranio!,
    });
    expect(genuineRes.matched).toBe(true);
    expect(genuineRes.topCandidate?.patientId).toBe('pat-only-1');

    const strictTracker = new LiveIdentityStreakTracker(BIOMETRIC_REQUIRED_STREAK);
    for (let f = 1; f < 6; f += 1) {
      const pending = strictTracker.step(genuineRes, { biologicalVerified: true });
      expect(pending.confirmed).toBe(false);
      expect(pending.streak).toBe(f);
    }
    const locked = strictTracker.step(genuineRes, { biologicalVerified: true });
    expect(locked.confirmed).toBe(true);
    expect(locked.streak).toBe(6);
    expect(locked.candidate?.patientId).toBe('pat-only-1');
  });
});

