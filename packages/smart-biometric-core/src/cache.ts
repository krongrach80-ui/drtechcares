/**
 * Zero Ghost Local Cache Policy:
 * Patient biometric vectors and identities must NEVER be cached in localStorage or IndexedDB.
 * Single Source of Truth is Supabase Cloud PostgreSQL + pgvector.
 */

const FORBIDDEN_BIOMETRIC_STORAGE_PATTERNS = [
  /face/i,
  /biometric/i,
  /embedding/i,
  /template/i,
  /patient_cache/i,
  /centroid/i,
];

const EXEMPT_ACTIVE_STORAGE_KEYS = new Set<string>([
  'drtech_biometric_vault_v3',
  'drtech_face_templates',
]);

function isForbiddenLegacyKey(key: string | null): key is string {
  if (!key || EXEMPT_ACTIVE_STORAGE_KEYS.has(key)) return false;
  return FORBIDDEN_BIOMETRIC_STORAGE_PATTERNS.some((re) => re.test(key));
}

export function purgeLegacyBiometricCache(): string[] {
  const purgedKeys: string[] = [];
  if (typeof globalThis.localStorage !== 'undefined') {
    const keysToRemove: string[] = [];
    for (let i = 0; i < globalThis.localStorage.length; i += 1) {
      const key = globalThis.localStorage.key(i);
      if (isForbiddenLegacyKey(key)) {
        keysToRemove.push(key);
      }
    }
    for (const key of keysToRemove) {
      globalThis.localStorage.removeItem(key);
      purgedKeys.push(key);
    }
  }
  if (typeof globalThis.sessionStorage !== 'undefined') {
    const keysToRemove: string[] = [];
    for (let i = 0; i < globalThis.sessionStorage.length; i += 1) {
      const key = globalThis.sessionStorage.key(i);
      if (isForbiddenLegacyKey(key)) {
        keysToRemove.push(key);
      }
    }
    for (const key of keysToRemove) {
      globalThis.sessionStorage.removeItem(key);
      purgedKeys.push(key);
    }
  }
  return purgedKeys;
}

export function assertZeroGhostLocalCache(): { clean: boolean; leakedKeys: string[] } {
  const leakedKeys: string[] = [];
  if (typeof globalThis.localStorage !== 'undefined') {
    for (let i = 0; i < globalThis.localStorage.length; i += 1) {
      const key = globalThis.localStorage.key(i);
      if (isForbiddenLegacyKey(key)) {
        leakedKeys.push(key);
      }
    }
  }
  return {
    clean: leakedKeys.length === 0,
    leakedKeys,
  };
}

