import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';
import type { DrTechCaresApiClient } from '@drtechcares/api-client';
import {
  fetchLiveBiometricsFromSupabase,
  warmupBiometricEngine,
  type LoadedBiometricVaultResult,
} from './biometricEngine';

export type VisionDelegate = 'GPU' | 'CPU';

export interface CachedVisionTaskHandle<T> {
  instance: T;
  delegate: VisionDelegate;
}

const WASM_PATHS = [
  '/mediapipe/wasm',
  '/models/wasm',
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
] as const;

let cachedVisionPromise: Promise<Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>> | null = null;
let cachedPrimaryLandmarkerPromise: Promise<FaceLandmarker> | null = null;
let cachedRoiLandmarkerPromise: Promise<FaceLandmarker | null> | null = null;
let cachedPoseLandmarkerPromises: Partial<Record<1 | 2, Promise<CachedVisionTaskHandle<PoseLandmarker>>>> = {};
let cachedHandLandmarkerPromises: Partial<Record<string, Promise<CachedVisionTaskHandle<HandLandmarker>>>> = {};
let cachedRoiHandLandmarkerPromises: Partial<Record<string, Promise<CachedVisionTaskHandle<HandLandmarker>>>> = {};
let cachedRehabFaceLandmarkerPromise: Promise<CachedVisionTaskHandle<FaceLandmarker> | null> | null = null;
let cachedVaultResult: {
  apiClient: DrTechCaresApiClient;
  vault: LoadedBiometricVaultResult;
  fetchedAt: number;
} | null = null;
let inflightPreloadPromise: Promise<void> | null = null;
let inflightPosePreloadPromise: Promise<void> | null = null;
let lastCachedVideoDetectTs = 0;
let lastCachedPoseDetectTs = 0;
let lastCachedHandDetectTs = 0;
let lastCachedRehabFaceDetectTs = 0;

let primaryFaceDelegate: VisionDelegate | 'unknown' = 'unknown';
let primaryFaceWarmupMs = 0;
let primaryFaceShaderCompiled = false;
let prewarmCompleted = false;

export interface VisionWarmupTelemetry {
  primaryFaceDelegate: VisionDelegate | 'unknown';
  primaryFaceWarmupMs: number;
  primaryFaceShaderCompiled: boolean;
  prewarmCompleted: boolean;
}

export function getVisionWarmupTelemetry(): VisionWarmupTelemetry {
  return {
    primaryFaceDelegate,
    primaryFaceWarmupMs,
    primaryFaceShaderCompiled,
    prewarmCompleted,
  };
}

function createBlankWarmupCanvas(width: number, height: number): HTMLCanvasElement | null {
  if (typeof document === 'undefined') return null;
  try {
    const cvs = document.createElement('canvas');
    cvs.width = width;
    cvs.height = height;
    const ctx = cvs.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#808080';
      ctx.fillRect(0, 0, width, height);
    }
    return cvs;
  } catch {
    return null;
  }
}

/**
 * Ensures strictly monotonically increasing `detectForVideo` timestamps even when
 * `KioskBiometricLoginView` unmounts and re-mounts against a shared cached `FaceLandmarker`.
 */
export function getNextMonotonicVideoDetectTs(candidateMs: number): number {
  const rounded = Number.isFinite(candidateMs) ? Math.round(candidateMs) : Date.now();
  const next = Math.max(lastCachedVideoDetectTs + 1, rounded);
  lastCachedVideoDetectTs = next;
  return next;
}

export function getNextMonotonicPoseDetectTs(candidateMs: number): number {
  const rounded = Number.isFinite(candidateMs) ? Math.round(candidateMs) : Date.now();
  const next = Math.max(lastCachedPoseDetectTs + 1, rounded);
  lastCachedPoseDetectTs = next;
  return next;
}

export function getNextMonotonicHandDetectTs(candidateMs: number): number {
  const rounded = Number.isFinite(candidateMs) ? Math.round(candidateMs) : Date.now();
  const next = Math.max(lastCachedHandDetectTs + 1, rounded);
  lastCachedHandDetectTs = next;
  return next;
}

export function getNextMonotonicRehabFaceDetectTs(candidateMs: number): number {
  const rounded = Number.isFinite(candidateMs) ? Math.round(candidateMs) : Date.now();
  const next = Math.max(lastCachedRehabFaceDetectTs + 1, rounded);
  lastCachedRehabFaceDetectTs = next;
  return next;
}

export async function getCachedVisionResolver(): Promise<
  Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>
> {
  if (!cachedVisionPromise) {
    cachedVisionPromise = (async () => {
      for (const wasmBase of WASM_PATHS) {
        try {
          return await FilesetResolver.forVisionTasks(wasmBase);
        } catch {
          // Try next WASM base
        }
      }
      throw new Error('Unable to initialize MediaPipe WASM Vision FilesetResolver');
    })().catch((err) => {
      cachedVisionPromise = null;
      throw err;
    });
  }
  return cachedVisionPromise;
}

export async function getCachedPrimaryFaceLandmarker(): Promise<FaceLandmarker> {
  if (!cachedPrimaryLandmarkerPromise) {
    const startMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    cachedPrimaryLandmarkerPromise = (async () => {
      const vision = await getCachedVisionResolver();
      for (const delegate of ['GPU', 'CPU'] as const) {
        try {
          const instance = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
            runningMode: 'VIDEO',
            numFaces: 2,
            outputFaceBlendshapes: true,
            minFaceDetectionConfidence: 0.32,
            minFacePresenceConfidence: 0.32,
            minTrackingConfidence: 0.32,
          });
          primaryFaceDelegate = delegate;
          // Pre-compile WebGL/WASM shaders once on mount using a lightweight 64x64 dummy canvas
          // so the first real camera frame does not stall or freeze the screen.
          const blank = createBlankWarmupCanvas(64, 64);
          if (blank) {
            try {
              const warmTs = getNextMonotonicVideoDetectTs(
                typeof performance !== 'undefined' ? performance.now() : Date.now(),
              );
              instance.detectForVideo(blank, warmTs);
              primaryFaceShaderCompiled = true;
            } catch {
              // Shader warmup best-effort; ignore in headless/mock contexts
            }
          }
          const endMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
          primaryFaceWarmupMs = Number(Math.max(0, endMs - startMs).toFixed(1));
          return instance;
        } catch {
          // Fall back to CPU delegate
        }
      }
      throw new Error('Unable to create primary FaceLandmarker');
    })().catch((err) => {
      cachedPrimaryLandmarkerPromise = null;
      throw err;
    });
  }
  return cachedPrimaryLandmarkerPromise;
}

export async function getCachedRoiFaceLandmarker(): Promise<FaceLandmarker | null> {
  if (!cachedRoiLandmarkerPromise) {
    cachedRoiLandmarkerPromise = (async () => {
      const vision = await getCachedVisionResolver();
      for (const delegate of ['GPU', 'CPU'] as const) {
        try {
          const instance = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
            runningMode: 'IMAGE',
            numFaces: 1,
            outputFaceBlendshapes: true,
            minFaceDetectionConfidence: 0.28,
            minFacePresenceConfidence: 0.28,
          });
          const blank = createBlankWarmupCanvas(128, 128);
          if (blank) {
            try {
              instance.detect(blank);
            } catch {
              // Best-effort shader warmup
            }
          }
          return instance;
        } catch {
          // Try next delegate
        }
      }
      return null;
    })().catch(() => {
      cachedRoiLandmarkerPromise = null;
      return null;
    });
  }
  return cachedRoiLandmarkerPromise;
}

export async function getCachedPoseLandmarker(options?: {
  numPoses?: 1 | 2;
}): Promise<CachedVisionTaskHandle<PoseLandmarker>> {
  const numPoses: 1 | 2 = options?.numPoses === 2 ? 2 : 1;
  if (!cachedPoseLandmarkerPromises[numPoses]) {
    cachedPoseLandmarkerPromises[numPoses] = (async () => {
      const vision = await getCachedVisionResolver();
      const modelPaths = ['/models/pose_landmarker_full.task', '/models/pose_landmarker.task'] as const;
      for (const modelAssetPath of modelPaths) {
        for (const delegate of ['GPU', 'CPU'] as const) {
          try {
            const instance = await PoseLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath, delegate },
              runningMode: 'VIDEO',
              numPoses,
              minPoseDetectionConfidence: 0.5,
              minPosePresenceConfidence: 0.5,
              minTrackingConfidence: 0.5,
            });
            return { instance, delegate };
          } catch {
            // Try next delegate or fallback model
          }
        }
      }
      throw new Error('Unable to create shared PoseLandmarker');
    })().catch((err) => {
      delete cachedPoseLandmarkerPromises[numPoses];
      throw err;
    });
  }
  return cachedPoseLandmarkerPromises[numPoses]!;
}

export async function getCachedHandLandmarker(options?: {
  numHands?: 2 | 4;
  minConfidence?: number;
  forceCpu?: boolean;
}): Promise<CachedVisionTaskHandle<HandLandmarker>> {
  const numHands: 2 | 4 = options?.numHands === 4 ? 4 : 2;
  const forceCpu = Boolean(options?.forceCpu);
  const cacheKey = `${numHands}_${forceCpu ? 'cpu' : 'auto'}`;
  const minConfidence = options?.minConfidence ?? 0.36;
  const minTrackingConfidence = Math.min(minConfidence, 0.32);
  if (!cachedHandLandmarkerPromises[cacheKey]) {
    cachedHandLandmarkerPromises[cacheKey] = (async () => {
      const vision = await getCachedVisionResolver();
      const modelPaths = [
        '/models/hand_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      ] as const;
      const delegates: readonly VisionDelegate[] = forceCpu ? ['CPU'] : ['GPU', 'CPU'];
      for (const modelAssetPath of modelPaths) {
        for (const delegate of delegates) {
          let instance: HandLandmarker | null = null;
          try {
            instance = await HandLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath, delegate },
              runningMode: 'VIDEO',
              numHands,
              minHandDetectionConfidence: minConfidence,
              minHandPresenceConfidence: minConfidence,
              minTrackingConfidence,
            });
            // Verify live VIDEO inference succeeds on this delegate (catches Windows WebGL context contention on GPU)
            const blank = createBlankWarmupCanvas(64, 64);
            if (blank) {
              const warmTs = getNextMonotonicHandDetectTs(
                typeof performance !== 'undefined' ? performance.now() : Date.now(),
              );
              instance.detectForVideo(blank, warmTs);
            }
            return { instance, delegate };
          } catch {
            try {
              instance?.close();
            } catch {
              // ignore close error
            }
            // Try next delegate or model path
          }
        }
      }
      throw new Error('Unable to create shared HandLandmarker');
    })().catch((err) => {
      delete cachedHandLandmarkerPromises[cacheKey];
      throw err;
    });
  }
  return cachedHandLandmarkerPromises[cacheKey]!;
}

/**
 * High-precision IMAGE-mode HandLandmarker dedicated to zoomed 256x256 per-hand ROI crops
 * anchored on PoseLandmarker wrists. Gives ~6x-8x higher effective pixel resolution on distant
 * hands (1.5m-2.5m), closed fists, and fingers gripping objects (e.g., mouse/cane) on BOTH sides.
 */
export async function getCachedRoiHandLandmarker(options?: {
  minConfidence?: number;
  forceCpu?: boolean;
}): Promise<CachedVisionTaskHandle<HandLandmarker>> {
  const forceCpu = Boolean(options?.forceCpu);
  const cacheKey = forceCpu ? 'roi_cpu' : 'roi_auto';
  const minConfidence = options?.minConfidence ?? 0.28;
  if (!cachedRoiHandLandmarkerPromises[cacheKey]) {
    cachedRoiHandLandmarkerPromises[cacheKey] = (async () => {
      const vision = await getCachedVisionResolver();
      const modelPaths = [
        '/models/hand_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      ] as const;
      const delegates: readonly VisionDelegate[] = forceCpu ? ['CPU'] : ['GPU', 'CPU'];
      for (const modelAssetPath of modelPaths) {
        for (const delegate of delegates) {
          let instance: HandLandmarker | null = null;
          try {
            instance = await HandLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath, delegate },
              runningMode: 'IMAGE',
              numHands: 1,
              minHandDetectionConfidence: minConfidence,
              minHandPresenceConfidence: minConfidence,
            });
            const blank = createBlankWarmupCanvas(96, 96);
            if (blank) {
              instance.detect(blank);
            }
            return { instance, delegate };
          } catch {
            try {
              instance?.close();
            } catch {
              // ignore close error
            }
          }
        }
      }
      throw new Error('Unable to create ROI HandLandmarker');
    })().catch((err) => {
      delete cachedRoiHandLandmarkerPromises[cacheKey];
      throw err;
    });
  }
  return cachedRoiHandLandmarkerPromises[cacheKey]!;
}

export async function getCachedRehabFaceLandmarker(): Promise<
  CachedVisionTaskHandle<FaceLandmarker> | null
> {
  if (!cachedRehabFaceLandmarkerPromise) {
    cachedRehabFaceLandmarkerPromise = (async () => {
      const vision = await getCachedVisionResolver();
      for (const delegate of ['GPU', 'CPU'] as const) {
        try {
          const instance = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
            runningMode: 'VIDEO',
            numFaces: 2,
            outputFaceBlendshapes: false,
            minFaceDetectionConfidence: 0.45,
            minFacePresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
          });
          return { instance, delegate };
        } catch {
          // Try next delegate
        }
      }
      return null;
    })().catch(() => {
      cachedRehabFaceLandmarkerPromise = null;
      return null;
    });
  }
  return cachedRehabFaceLandmarkerPromise;
}

export async function preloadRehabPosePipeline(): Promise<void> {
  if (typeof window === 'undefined') return;
  if (inflightPosePreloadPromise) return inflightPosePreloadPromise;

  inflightPosePreloadPromise = getCachedPoseLandmarker({ numPoses: 1 })
    .then(() => undefined)
    .catch(() => undefined)
    .finally(() => {
      inflightPosePreloadPromise = null;
    });

  return inflightPosePreloadPromise;
}

export function getPreloadedBiometricVault(
  apiClient: DrTechCaresApiClient,
  maxAgeMs = 20000,
): { vault: LoadedBiometricVaultResult; fetchedAt: number } | null {
  if (!cachedVaultResult) return null;
  if (cachedVaultResult.apiClient !== apiClient) return null;
  if (Date.now() - cachedVaultResult.fetchedAt > maxAgeMs) return null;
  return {
    vault: cachedVaultResult.vault,
    fetchedAt: cachedVaultResult.fetchedAt,
  };
}

export function setPreloadedBiometricVault(
  apiClient: DrTechCaresApiClient,
  vault: LoadedBiometricVaultResult,
): void {
  cachedVaultResult = {
    apiClient,
    vault,
    fetchedAt: Date.now(),
  };
}

/**
 * Eagerly pre-warms @vladmandic/human faceres (1024-D), MediaPipe 478-pt FaceLandmarker
 * (including GPU shader compilation on a blank canvas), Cloudflare Worker/Supabase connection,
 * and count-only library stats during Standby idle WITHOUT opening the camera and WITHOUT
 * downloading any other patients' biometric vectors to the browser.
 */
export async function preloadKioskBiometricPipeline(
  apiClient: DrTechCaresApiClient,
): Promise<void> {
  if (typeof window === 'undefined') return;
  if (inflightPreloadPromise) return inflightPreloadPromise;

  inflightPreloadPromise = (async () => {
    const healthPing =
      typeof apiClient.checkHealth === 'function'
        ? apiClient.checkHealth().catch(() => undefined)
        : Promise.resolve();
    const statsPing =
      typeof apiClient.getBiometricLibraryStats === 'function'
        ? apiClient.getBiometricLibraryStats(false, true).catch(() => undefined)
        : Promise.resolve();
    await Promise.allSettled([
      warmupBiometricEngine(),
      getCachedPrimaryFaceLandmarker().then(() => {
        void getCachedRoiFaceLandmarker();
      }),
      healthPing,
      statsPing,
    ]);
    prewarmCompleted = true;
  })().finally(() => {
    inflightPreloadPromise = null;
  });

  return inflightPreloadPromise;
}

export function resetBiometricVisionCacheForTests(): void {
  cachedVisionPromise = null;
  cachedPrimaryLandmarkerPromise = null;
  cachedRoiLandmarkerPromise = null;
  cachedPoseLandmarkerPromises = {};
  cachedHandLandmarkerPromises = {};
  cachedRoiHandLandmarkerPromises = {};
  cachedRehabFaceLandmarkerPromise = null;
  cachedVaultResult = null;
  inflightPreloadPromise = null;
  inflightPosePreloadPromise = null;
  lastCachedVideoDetectTs = 0;
  lastCachedPoseDetectTs = 0;
  lastCachedHandDetectTs = 0;
  lastCachedRehabFaceDetectTs = 0;
  primaryFaceDelegate = 'unknown';
  primaryFaceWarmupMs = 0;
  primaryFaceShaderCompiled = false;
  prewarmCompleted = false;
}

