import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { DrTechCaresApiClient } from '@drtechcares/api-client';
import {
  fetchLiveBiometricsFromSupabase,
  warmupBiometricEngine,
  type LoadedBiometricVaultResult,
} from './biometricEngine';

const WASM_PATHS = [
  '/mediapipe/wasm',
  '/models/wasm',
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
] as const;

let cachedVisionPromise: Promise<Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>> | null = null;
let cachedPrimaryLandmarkerPromise: Promise<FaceLandmarker> | null = null;
let cachedRoiLandmarkerPromise: Promise<FaceLandmarker | null> | null = null;
let cachedVaultResult: {
  apiClient: DrTechCaresApiClient;
  vault: LoadedBiometricVaultResult;
  fetchedAt: number;
} | null = null;
let inflightPreloadPromise: Promise<void> | null = null;
let lastCachedVideoDetectTs = 0;

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
    cachedPrimaryLandmarkerPromise = (async () => {
      const vision = await getCachedVisionResolver();
      for (const delegate of ['GPU', 'CPU'] as const) {
        try {
          return await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
            runningMode: 'VIDEO',
            numFaces: 2,
            outputFaceBlendshapes: true,
            minFaceDetectionConfidence: 0.32,
            minFacePresenceConfidence: 0.32,
            minTrackingConfidence: 0.32,
          });
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
          return await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
            runningMode: 'IMAGE',
            numFaces: 1,
            outputFaceBlendshapes: true,
            minFaceDetectionConfidence: 0.28,
            minFacePresenceConfidence: 0.28,
          });
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
 * Eagerly pre-warms @vladmandic/human faceres (1024-D), MediaPipe 478-pt FaceLandmarker,
 * and the enrolled patient vault in the background while on the Standby screen so Kiosk
 * begins matching on Frame 0 with zero waiting.
 */
export async function preloadKioskBiometricPipeline(
  apiClient: DrTechCaresApiClient,
): Promise<void> {
  if (typeof window === 'undefined') return;
  if (inflightPreloadPromise) return inflightPreloadPromise;

  inflightPreloadPromise = (async () => {
    await Promise.allSettled([
      warmupBiometricEngine(),
      getCachedPrimaryFaceLandmarker().then(() => {
        void getCachedRoiFaceLandmarker();
      }),
      fetchLiveBiometricsFromSupabase(apiClient).then((vault) => {
        setPreloadedBiometricVault(apiClient, vault);
      }),
    ]);
  })().finally(() => {
    inflightPreloadPromise = null;
  });

  return inflightPreloadPromise;
}

export function resetBiometricVisionCacheForTests(): void {
  cachedVisionPromise = null;
  cachedPrimaryLandmarkerPromise = null;
  cachedRoiLandmarkerPromise = null;
  cachedVaultResult = null;
  inflightPreloadPromise = null;
  lastCachedVideoDetectTs = 0;
}
