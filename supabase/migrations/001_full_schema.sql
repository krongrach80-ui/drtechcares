-- Migration 001: Full Schema Extension for Dr.Tech.Care Portal
-- Mirrors supabase/schema.sql (safe idempotent execution)

ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS license_number text;
ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'ACTIVE';

ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS gender text NOT NULL DEFAULT 'ชาย';
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS face_biometric_status text NOT NULL DEFAULT 'PENDING';
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS face_embedding jsonb;
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS face_image_url text;
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS history jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_patients_hn ON public.patients (hn);
CREATE INDEX IF NOT EXISTS idx_patients_assigned_pt ON public.patients (assigned_pt_id);

ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS name_th text NOT NULL DEFAULT '';
ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS body_side text NOT NULL DEFAULT 'RIGHT';
ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS instructions_th text NOT NULL DEFAULT '';
ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS target_rom_deg real NOT NULL DEFAULT 135;
ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS hold_sec real NOT NULL DEFAULT 2;

ALTER TABLE public.modpacks ADD COLUMN IF NOT EXISTS category text NOT NULL DEFAULT 'general';
ALTER TABLE public.modpacks ADD COLUMN IF NOT EXISTS card_count int NOT NULL DEFAULT 10;
ALTER TABLE public.modpacks ADD COLUMN IF NOT EXISTS tags text[] NOT NULL DEFAULT ARRAY[]::text[];

ALTER TABLE public.system_settings ADD COLUMN IF NOT EXISTS key text;
ALTER TABLE public.system_settings ADD COLUMN IF NOT EXISTS value jsonb;
ALTER TABLE public.system_settings ADD COLUMN IF NOT EXISTS updated_by uuid;

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

CREATE TABLE IF NOT EXISTS public.emergency_alerts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kiosk_id text NOT NULL,
  patient_id uuid REFERENCES public.patients(id) ON DELETE SET NULL,
  reason text NOT NULL,
  status text NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'ACKNOWLEDGED', 'RESOLVED')),
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS action text;
ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS target text;
ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS details jsonb DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON public.audit_logs (created_at DESC);

ALTER TABLE public.kiosks ADD COLUMN IF NOT EXISTS pairing_token text;

CREATE TABLE IF NOT EXISTS public.kiosk_devices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kiosk_code text UNIQUE NOT NULL,
  location text NOT NULL DEFAULT '',
  online_status boolean NOT NULL DEFAULT true,
  pairing_token text,
  last_heartbeat timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.banned_entities (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_type text NOT NULL CHECK (entity_type IN ('ip', 'device', 'user')),
  value text NOT NULL,
  reason text NOT NULL DEFAULT '',
  banned_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.rehab_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.emergency_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.kiosk_devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.banned_entities ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS staff_all_rehab_schedules ON public.rehab_schedules;
CREATE POLICY staff_all_rehab_schedules ON public.rehab_schedules FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_emergency_alerts ON public.emergency_alerts;
CREATE POLICY staff_all_emergency_alerts ON public.emergency_alerts FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_kiosk_devices ON public.kiosk_devices;
CREATE POLICY staff_all_kiosk_devices ON public.kiosk_devices FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_banned_entities ON public.banned_entities;
CREATE POLICY staff_all_banned_entities ON public.banned_entities FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_read_calls_realtime ON public.staff_calls;
CREATE POLICY staff_read_calls_realtime ON public.staff_calls FOR SELECT USING (true);

