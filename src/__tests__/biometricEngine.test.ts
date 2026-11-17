import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DrTechCaresApiClient, PatientBiometricIndexEntry } from '@drtechcares/api-client';
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
    expect(Math.abs(scoreA.finalScore - scoreB.finalScore)).toBeLessThan(BIOMETRIC_MARGIN_THRESHOLD);
    const resTie = evaluateLiveMatch1toN(tieProbe, [patA, patB]);
    expect(resTie.matched).toBe(false);
    expect(resTie.reason).toBe('AMBIGUOUS_CLOSE_CANDIDATES');
  });

  it('requires 4/4 consecutive matching live frames and resets on spoof or flip', () => {
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

    // 4/4 clean frames -> confirmed
    expect(tracker.step(evalA, { biologicalVerified: true }).confirmed).toBe(false);
    expect(tracker.step(evalA, { biologicalVerified: true }).confirmed).toBe(false);
    expect(tracker.step(evalA, { biologicalVerified: true }).confirmed).toBe(false);
    const fourth = tracker.step(evalA, { biologicalVerified: true });
    expect(fourth.confirmed).toBe(true);
    expect(fourth.streak).toBe(4);
    expect(fourth.candidate?.patientId).toBe('pat-a');
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
      diagnosis: 'Stroke',
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
});

