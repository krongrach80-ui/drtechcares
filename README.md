# DRTECHCARES — AI Smart Hospital Rehabilitation Platform

Monorepo for the DRTECHCARES hospital rehabilitation kiosk, physical therapist portal, patient personal portal, and super-admin console.

## Portals & Routes

| Route | Portal | Purpose |
| --- | --- | --- |
| `/` | Smart Rehab Kiosk | Touchless Face ID login, cognitive card matching & guided rehab |
| `/login` | Staff & Patient Login | Credential sign-in for Super Admin, PT, and Patient portals |
| `/pt` | Physical Therapist Portal | 10-shot biometric enrollment, weekly schedule prescription, append-only notes, case handover |
| `/admin` | Super Admin Console | RBAC user management, security bans, force logout, audit logs, AI feature toggles |
| `/patient` | Patient Personal Portal | Personal rehab schedule, session history, and progress |

## Default Seeded Accounts (Development / Initial Edge State)

| Role | Username | Password |
| --- | --- | --- |
| Super Admin | `admin` | `Admin@2026` |
| Physical Therapist | `pt_somchai` | `Pt@2026` |
| Physical Therapist | `pt_waranya` | `Pt@2026` |

> Change default credentials immediately in production deployments.

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key variables:
- `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`: Supabase project URL and anon key for client RPC.
- `VITE_ARCFACE_MODEL_URL`: Optional CDN/R2 URL for `w600k_r50.onnx` (166 MB). On Cloudflare Assets, `scripts/strip-models.mjs` strips any `.onnx` or `.wasm` file > 24 MiB from `dist/models/` and `dist/assets/` (writing `*.status.json`); the browser automatically falls back to `hybrid-512d` unless both `VITE_ARCFACE_MODEL_URL` and `VITE_ORT_WASM_PATHS` are hosted externally.
- `VITE_ORT_WASM_PATHS`: Optional CDN base URL for `onnxruntime-web` WASM files (when using ArcFace ONNX on Cloudflare Workers).
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`: Cloudflare Worker bindings/secrets.
- `DRTECH_MODEL_CACHE_DIR`: Optional local directory for `scripts/copy-wasm.mjs` to copy `.onnx`/`.task` model assets into `public/models/`.

## Database Migration (Supabase PostgreSQL + pgvector)

Apply `supabase/migrations/20261113000001_drtechcares_schema_and_rpc.sql` to your Supabase project (via Supabase SQL Editor or `supabase db push`). It provisions:
- Core clinical tables (`staff_users`, `patients`, `face_enrollments`, `face_templates`, `exercises`, `modpacks`, `patient_courses`, `rehab_sessions`, `staff_calls`, `audit_logs`)
- Extended security & workflow tables (`medical_notes`, `treatment_reports`, `banned_targets`, `active_sessions`, `system_settings`, `kiosks`)
- `match_patient_face` pgvector HNSW similarity RPC

## Scripts & Commands

```bash
# Start Vite frontend + local Express API (port 8787)
npm run dev

# Run TypeScript check
npm run typecheck

# Run full Vitest suite
npm test

# Production build (WASM/models copy, tsc, Vite dist/, Worker dist-worker/worker.js, >24MiB ONNX strip)
npm run build

# Set Cloudflare Worker secrets and deploy
npx wrangler secret put SUPABASE_SERVICE_ROLE_KEY
npx wrangler deploy
```

