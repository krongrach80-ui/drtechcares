import fs from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { createCloudRouter } from '../../packages/api-client/src/cloudRouter';
import {
  resetSupabaseCircuitForTests,
  resetUnsupportedColumnsForTests,
} from '../../packages/api-client/src/supabasePersistence';
import {
  reconcileStaffProfileOverridesFromServer,
  resolveUserProfile,
} from '../components/ClinicalStaffPortalShell';
import type { UserAccount } from '@drtechcares/api-client';

const UNICODE_EMOJI_RE = /[\p{Extended_Pictographic}\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u;
const MOCK_SUPABASE_URL = 'https://mock-sync-project.supabase.co';
const MOCK_SERVICE_KEY = 'sb_secret_mock_service_role_key_for_cross_device_sync_test_2026';

/**
 * Intercepts `@supabase/supabase-js` HTTP calls (PostgREST `/rest/v1/*` + Storage `/storage/v1/object/*`)
 * to simulate the real production Supabase environment where:
 * - `public.staff_users` has base columns (no `avatar_url`/`bio` columns yet -> returns HTTP 400 PGRST204)
 * - `patient-faces` Storage bucket stores uploaded staff photos (`staff-avatars/...`) & `staff-profiles/manifest.json`
 * Two completely separate Worker router instances (Device 1 & Device 2) share this cloud state.
 */
function installMockSupabaseFetch(options?: { withAvatarColumns?: boolean }) {
  const withAvatarColumns = options?.withAvatarColumns ?? false;
  const staffRows: Array<Record<string, unknown>> = [
    {
      id: '00000000-0000-4000-8000-000000000001',
      username: 'admin',
      password_hash: 'plain:Admin@2026',
      role: 'super_admin',
      full_name: 'นพ.สมชาย ผู้อำนวยการโรงพยาบาล (Super Admin)',
      phone: '0914443232',
      specialty: 'Hospital Director',
      license_no: 'MD-10001',
      patient_id: null,
      active: true,
      force_logged_out_at: null,
      created_at: '2026-01-01T00:00:00.000Z',
      ...(withAvatarColumns ? { avatar_url: null, bio: null } : {}),
    },
  ];
  const storageObjects = new Map<string, { bytes: Uint8Array; contentType: string }>();

  const readRequestBytes = async (init?: RequestInit): Promise<Uint8Array> => {
    const body = init?.body;
    if (!body) return new Uint8Array();
    if (body instanceof Uint8Array) return new Uint8Array(body);
    if (body instanceof ArrayBuffer) return new Uint8Array(body);
    if (typeof body === 'string') return new TextEncoder().encode(body);
    if (typeof FormData !== 'undefined' && body instanceof FormData) {
      for (const [, val] of body.entries()) {
        if (typeof val !== 'string' && typeof val.arrayBuffer === 'function') {
          return new Uint8Array(await val.arrayBuffer());
        }
        if (typeof val === 'string' && val.startsWith('{')) {
          return new TextEncoder().encode(val);
        }
      }
      return new Uint8Array();
    }
    if (typeof Blob !== 'undefined' && body instanceof Blob) {
      return new Uint8Array(await body.arrayBuffer());
    }
    return new Uint8Array();
  };

  const origFetch = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const urlStr = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    const url = new URL(urlStr);
    const method = (init?.method || (typeof input !== 'string' && !(input instanceof URL) ? input.method : 'GET') || 'GET').toUpperCase();

    // 1. Supabase Storage Object Upload & Download
    if (url.pathname.startsWith('/storage/v1/object/')) {
      const objectKey = url.pathname
        .replace(/^\/storage\/v1\/object\/(?:authenticated\/|public\/)?/, '')
        .replace(/^\/+/, '');
      if (method === 'POST' || method === 'PUT') {
        const bytes = await readRequestBytes(init);
        const ct =
          (init?.headers && !Array.isArray(init.headers) && !(init.headers instanceof Headers)
            ? (init.headers as Record<string, string>)['content-type'] ||
              (init.headers as Record<string, string>)['Content-Type']
            : init?.headers instanceof Headers
              ? init.headers.get('content-type')
              : null) || 'application/octet-stream';
        storageObjects.set(objectKey, { bytes, contentType: ct });
        return new Response(JSON.stringify({ Key: objectKey, Id: objectKey }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (method === 'GET') {
        const obj = storageObjects.get(objectKey);
        if (!obj) {
          return new Response(JSON.stringify({ statusCode: '404', error: 'not_found', message: 'Object not found' }), {
            status: 404,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response(new Uint8Array(obj.bytes).buffer as ArrayBuffer, {
          status: 200,
          headers: { 'Content-Type': obj.contentType },
        });
      }
    }

    // 2. PostgREST /rest/v1/staff_users
    if (url.pathname === '/rest/v1/staff_users') {
      if (method === 'GET') {
        const selectRaw = url.searchParams.get('select') || '*';
        const cols = selectRaw.split(',').map((c) => c.trim());
        if (!withAvatarColumns && (cols.includes('avatar_url') || cols.includes('bio'))) {
          return new Response(
            JSON.stringify({
              code: 'PGRST204',
              message: "Could not find the 'avatar_url' column of 'staff_users' in the schema cache",
            }),
            { status: 400, headers: { 'Content-Type': 'application/json' } },
          );
        }
        const rows = staffRows.map((row) => {
          if (selectRaw === '*') return { ...row };
          const out: Record<string, unknown> = {};
          for (const c of cols) {
            if (c in row) out[c] = row[c];
          }
          return out;
        });
        return new Response(JSON.stringify(rows), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (method === 'PATCH' || method === 'POST') {
        const bodyText =
          typeof init?.body === 'string'
            ? init.body
            : init?.body instanceof Uint8Array
              ? new TextDecoder().decode(init.body)
              : '{}';
        const payload = JSON.parse(bodyText) as Record<string, unknown>;
        if (!withAvatarColumns && ('avatar_url' in payload || 'bio' in payload)) {
          return new Response(
            JSON.stringify({
              code: 'PGRST204',
              message: "Could not find the 'avatar_url' column of 'staff_users' in the schema cache",
            }),
            { status: 400, headers: { 'Content-Type': 'application/json' } },
          );
        }
        const eqId = url.searchParams.get('id')?.replace(/^eq\./, '');
        const row = staffRows.find((r) => r.id === eqId);
        if (row) {
          Object.assign(row, payload);
        }
        return new Response(JSON.stringify(row ? [row] : []), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }

    // 3. Other PostgREST tables return empty success arrays
    if (url.pathname.startsWith('/rest/v1/')) {
      return new Response(JSON.stringify([]), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return origFetch(input, init);
  }) as typeof fetch;

  return {
    storageObjects,
    staffRows,
    restore: () => {
      globalThis.fetch = origFetch;
    },
  };
}

async function callRouter(
  router: ReturnType<typeof createCloudRouter>,
  pathname: string,
  options?: { method?: string; token?: string; body?: unknown },
) {
  const headers: Record<string, string> = {};
  if (options?.token) headers.Authorization = `Bearer ${options.token}`;
  if (options?.body !== undefined) headers['Content-Type'] = 'application/json';
  const req = new Request(`https://drtechcares.example${pathname}`, {
    method: options?.method ?? 'GET',
    headers,
    body: options?.body !== undefined ? JSON.stringify(options.body) : undefined,
  });
  const res = await router(req);
  const json = await res.json();
  return { status: res.status, json };
}

describe('Cross-Device Staff Profile Photo & Bio Synchronization', () => {
  let restoreFetch: (() => void) | null = null;

  beforeEach(() => {
    resetUnsupportedColumnsForTests();
    resetSupabaseCircuitForTests();
  });

  afterEach(() => {
    restoreFetch?.();
    restoreFetch = null;
  });

  it('uploads real base64 photo on Device 1 to Supabase Storage and immediately syncs to Device 2 even without staff_users.avatar_url column', async () => {
    const env = installMockSupabaseFetch({ withAvatarColumns: false });
    restoreFetch = env.restore;

    const device1Router = createCloudRouter({
      supabaseUrl: MOCK_SUPABASE_URL,
      supabaseServiceKey: MOCK_SERVICE_KEY,
    });
    const device2Router = createCloudRouter({
      supabaseUrl: MOCK_SUPABASE_URL,
      supabaseServiceKey: MOCK_SERVICE_KEY,
    });

    // Device 1 logs in as @admin
    const login1 = await callRouter(device1Router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    expect(login1.status).toBe(200);
    const token1 = login1.json.token as string;
    const adminId = login1.json.user.id as string;

    // Device 1 uploads a real >12KB JPEG photo (previously stripped to null by stripOversizedBase64ForDb!)
    const uploadedPhotoBase64 = `data:image/jpeg;base64,${Buffer.from('X'.repeat(14000)).toString('base64')}`;
    const updatedBio = 'ผู้อำนวยการโรงพยาบาล อัปเดตจากเครื่องที่ 1';
    const saveRes = await callRouter(device1Router, `/api/users/${adminId}`, {
      method: 'PUT',
      token: token1,
      body: {
        fullName: 'นพ.สมชาย ผู้อำนวยการโรงพยาบาล (Super Admin)',
        phone: '0914443232',
        specialty: 'Hospital Director',
        licenseNo: 'MD-10001',
        avatarUrl: uploadedPhotoBase64,
        bio: updatedBio,
      },
    });
    expect(saveRes.status).toBe(200);
    const cloudAvatarUrl = saveRes.json.user.avatarUrl as string;
    expect(cloudAvatarUrl).toMatch(
      /^https:\/\/mock-sync-project\.supabase\.co\/storage\/v1\/object\/public\/patient-faces\/staff-avatars\/.+\.jpg$/,
    );
    expect(saveRes.json.user.bio).toBe(updatedBio);
    expect(env.storageObjects.has('patient-faces/staff-profiles/manifest.json')).toBe(true);

    // Device 2 (brand new Worker isolate!) logs in and calls /api/auth/me?refresh=1 and /api/users?refresh=1
    const login2 = await callRouter(device2Router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    expect(login2.status).toBe(200);
    expect(login2.json.user.avatarUrl).toBe(cloudAvatarUrl);
    expect(login2.json.user.bio).toBe(updatedBio);

    const token2 = login2.json.token as string;
    const me2 = await callRouter(device2Router, '/api/auth/me?refresh=1', { token: token2 });
    expect(me2.status).toBe(200);
    expect(me2.json.user.avatarUrl).toBe(cloudAvatarUrl);
    expect(me2.json.user.bio).toBe(updatedBio);

    const users2 = await callRouter(device2Router, '/api/users?refresh=1', { token: token2 });
    expect(users2.status).toBe(200);
    const adminOnDevice2 = (users2.json.users as UserAccount[]).find((u) => u.username === 'admin');
    expect(adminOnDevice2?.avatarUrl).toBe(cloudAvatarUrl);
    expect(adminOnDevice2?.bio).toBe(updatedBio);

    // Device 1 changes avatar again to a preset SVG -> Device 2 immediately sees the new preset on refresh=1
    const presetSvg =
      'data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3C%2Fsvg%3E';
    const savePreset = await callRouter(device1Router, `/api/users/${adminId}`, {
      method: 'PUT',
      token: token1,
      body: { avatarUrl: presetSvg },
    });
    expect(savePreset.status).toBe(200);
    expect(savePreset.json.user.avatarUrl).toBe(presetSvg);

    const users2AfterPreset = await callRouter(device2Router, '/api/users?refresh=1', { token: token2 });
    const adminOnDevice2AfterPreset = (users2AfterPreset.json.users as UserAccount[]).find(
      (u) => u.username === 'admin',
    );
    expect(adminOnDevice2AfterPreset?.avatarUrl).toBe(presetSvg);
  });

  it('prefers fresh cloud avatarUrl over stale per-device localStorage overrides in resolveUserProfile & reconciles local cache', () => {
    const staleCartoonSvg = 'data:image/svg+xml;utf8,%3Csvg%3Eold-cartoon%3C%2Fsvg%3E';
    const freshCloudUrl =
      'https://vyjuwjhbdhknjdzyqctz.supabase.co/storage/v1/object/public/patient-faces/staff-avatars/admin-999.jpg';

    const serverUser: UserAccount = {
      id: '00000000-0000-4000-8000-000000000001',
      username: 'admin',
      role: 'super_admin',
      fullName: 'นพ.สมชาย ผู้อำนวยการโรงพยาบาล (Super Admin)',
      phone: '0914443232',
      specialty: 'Hospital Director',
      licenseNo: 'MD-10001',
      avatarUrl: freshCloudUrl,
      bio: 'อัปเดตประวัติใหม่บนคลาวด์',
      active: true,
      createdAt: '2026-01-01T00:00:00.000Z',
    };

    const staleDevice2Overrides = {
      '00000000-0000-4000-8000-000000000001': {
        avatarUrl: staleCartoonSvg,
        bio: 'ข้อความเก่าในเครื่อง 2',
      },
      'usr:admin': {
        avatarUrl: staleCartoonSvg,
        bio: 'ข้อความเก่าในเครื่อง 2',
      },
    };

    const resolved = resolveUserProfile(serverUser, staleDevice2Overrides);
    expect(resolved.avatarUrl).toBe(freshCloudUrl);
    expect(resolved.bio).toBe('อัปเดตประวัติใหม่บนคลาวด์');

    const reconciled = reconcileStaffProfileOverridesFromServer(staleDevice2Overrides, [serverUser]);
    expect(reconciled[serverUser.id]?.avatarUrl).toBe(freshCloudUrl);
    expect(reconciled['usr:admin']?.avatarUrl).toBe(freshCloudUrl);
  });

  it('contains zero Unicode emojis across all new cross-device profile sync files', () => {
    const root = process.cwd();
    const files = [
      path.join(root, 'supabase', 'migrations', '011_staff_users_avatar_bio_sync.sql'),
      path.join(root, 'src', '__tests__', 'staffProfileCrossDeviceSync.test.ts'),
    ];
    for (const file of files) {
      const text = fs.readFileSync(file, 'utf8');
      expect(UNICODE_EMOJI_RE.test(text)).toBe(false);
    }
  });
});
