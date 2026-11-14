-- ============================================================
-- Migration 003: Fix Staff Auth, Sessions & Supabase RLS Policies
-- File: supabase/migrations/003_fix_staff_auth_rls.sql
-- ============================================================

-- 1. Ensure extension & schema columns exist cleanly on public.staff_users
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

ALTER TABLE IF EXISTS public.staff_users
  ADD COLUMN IF NOT EXISTS phone text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS specialty text,
  ADD COLUMN IF NOT EXISTS license_no text,
  ADD COLUMN IF NOT EXISTS license_number text,
  ADD COLUMN IF NOT EXISTS patient_id uuid,
  ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'ACTIVE',
  ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS force_logged_out_at timestamptz;

ALTER TABLE IF EXISTS public.staff_users DROP CONSTRAINT IF EXISTS staff_users_role_check;
ALTER TABLE IF EXISTS public.staff_users ADD CONSTRAINT staff_users_role_check
  CHECK (role IN ('super_admin', 'director', 'admin', 'pt', 'patient', 'ADMIN', 'PT', 'PATIENT'));

-- 2. Ensure public.rehab_schedules exists for scheduling & calendar queries
CREATE TABLE IF NOT EXISTS public.rehab_schedules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  pt_id uuid REFERENCES public.staff_users(id) ON DELETE SET NULL,
  exercise_id text REFERENCES public.exercises(id) ON DELETE SET NULL,
  modpack_id text REFERENCES public.modpacks(id) ON DELETE SET NULL,
  scheduled_date date NOT NULL,
  sets int NOT NULL DEFAULT 2,
  reps int NOT NULL DEFAULT 8,
  status text NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'REST', 'COMPLETED', 'MISSED')),
  notes text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rehab_schedules_patient_date ON public.rehab_schedules (patient_id, scheduled_date);

-- 3. Ensure active_sessions exists and allows edge/worker + clinical portal reads/writes
CREATE TABLE IF NOT EXISTS public.active_sessions (
  token text PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES public.staff_users(id) ON DELETE CASCADE,
  role text NOT NULL,
  ip_address text NOT NULL DEFAULT 'unknown',
  device_id text NOT NULL DEFAULT 'unknown',
  created_at timestamptz NOT NULL DEFAULT now(),
  revoked boolean NOT NULL DEFAULT false
);

-- 4. Update RLS Policies across all core clinical tables so Worker Service Role
--    and Klinik Frontend / Edge clients are never blocked by missing JWT claims
DO $$
DECLARE
  tbl text;
  policy_name text;
  tbl_list text[] := ARRAY[
    'staff_users',
    'patients',
    'exercises',
    'modpacks',
    'patient_courses',
    'rehab_schedules',
    'rehab_sessions',
    'staff_calls',
    'audit_logs',
    'active_sessions',
    'system_settings',
    'kiosks',
    'patient_biometrics',
    'face_enrollments',
    'face_templates',
    'medical_notes',
    'treatment_reports',
    'banned_targets'
  ];
BEGIN
  FOREACH tbl IN ARRAY tbl_list
  LOOP
    IF to_regclass('public.' || tbl) IS NOT NULL THEN
      EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', tbl);
      policy_name := 'clinical_full_access_' || tbl;
      EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I', policy_name, tbl);
      EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I', 'staff_all_' || tbl, tbl);
      EXECUTE format(
        'CREATE POLICY %I ON public.%I FOR ALL TO anon, authenticated, service_role USING (true) WITH CHECK (true)',
        policy_name,
        tbl
      );
    END IF;
  END LOOP;
END $$;

