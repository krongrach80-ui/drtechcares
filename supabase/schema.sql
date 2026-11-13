-- Dr.Tech.Care Complete Production Schema (Supabase / PostgreSQL + pgvector + RLS)
-- Safe to re-run (idempotent)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Staff & Portal Users
CREATE TABLE IF NOT EXISTS public.staff_users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  username text UNIQUE NOT NULL,
  password_hash text NOT NULL,
  full_name text NOT NULL,
  phone text NOT NULL DEFAULT '',
  specialty text,
  license_no text,
  license_number text,
  patient_id uuid,
  role text NOT NULL CHECK (role IN ('super_admin', 'director', 'admin', 'pt', 'patient')),
  status text NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'SUSPENDED')),
  active boolean NOT NULL DEFAULT true,
  force_logged_out_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS phone text NOT NULL DEFAULT '';
ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS specialty text;
ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS license_no text;
ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS license_number text;
ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS patient_id uuid;
ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'ACTIVE';
ALTER TABLE public.staff_users ADD COLUMN IF NOT EXISTS force_logged_out_at timestamptz;
ALTER TABLE public.staff_users DROP CONSTRAINT IF EXISTS staff_users_role_check;
ALTER TABLE public.staff_users ADD CONSTRAINT staff_users_role_check
  CHECK (role IN ('super_admin', 'director', 'admin', 'pt', 'patient'));

-- 2. Patients
CREATE TABLE IF NOT EXISTS public.patients (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  hn text UNIQUE NOT NULL,
  full_name text NOT NULL,
  age int NOT NULL CHECK (age >= 0 AND age <= 130),
  gender text NOT NULL DEFAULT 'ชาย',
  phone text NOT NULL DEFAULT '',
  diagnosis text NOT NULL DEFAULT '',
  registration_status text NOT NULL DEFAULT 'active' CHECK (registration_status IN ('active', 'archived')),
  assigned_pt_id uuid REFERENCES public.staff_users(id) ON DELETE SET NULL,
  delegated_to_pt_id uuid REFERENCES public.staff_users(id) ON DELETE SET NULL,
  delegation_note text,
  next_appointment_date text,
  face_biometric_status text NOT NULL DEFAULT 'PENDING' CHECK (face_biometric_status IN ('PENDING', 'VERIFIED')),
  face_embedding jsonb,
  face_image_url text,
  history jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS gender text NOT NULL DEFAULT 'ชาย';
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS phone text NOT NULL DEFAULT '';
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS delegated_to_pt_id uuid REFERENCES public.staff_users(id) ON DELETE SET NULL;
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS delegation_note text;
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS next_appointment_date text;
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS face_biometric_status text NOT NULL DEFAULT 'PENDING';
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS face_embedding jsonb;
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS face_image_url text;
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS history jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_patients_hn ON public.patients (hn);
CREATE INDEX IF NOT EXISTS idx_patients_assigned_pt ON public.patients (assigned_pt_id);

-- 3. Multi-Angle 10-Shot Face Enrollments
CREATE TABLE IF NOT EXISTS public.face_enrollments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  shot_count int NOT NULL CHECK (shot_count = 10),
  liveness_verified boolean NOT NULL DEFAULT true,
  enrolled_by uuid REFERENCES public.staff_users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- 4. Face Templates (512-D pgvector + HNSW)
CREATE TABLE IF NOT EXISTS public.face_templates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  pose text NOT NULL CHECK (pose IN ('frontal_centroid', 'overall_centroid', 'front', 'left', 'right', 'nod')),
  embedding vector(512) NOT NULL,
  quality_score real NOT NULL DEFAULT 1.0,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_face_templates_patient_pose ON public.face_templates (patient_id, pose);
CREATE INDEX IF NOT EXISTS idx_face_templates_embedding_hnsw
  ON public.face_templates USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 5. Exercises
CREATE TABLE IF NOT EXISTS public.exercises (
  id text PRIMARY KEY,
  code text UNIQUE NOT NULL,
  category text NOT NULL,
  name_th text NOT NULL DEFAULT '',
  body_side text NOT NULL DEFAULT 'RIGHT' CHECK (body_side IN ('LEFT', 'RIGHT', 'BOTH')),
  instructions_th text NOT NULL DEFAULT '',
  target_rom_deg real NOT NULL DEFAULT 135,
  hold_sec real NOT NULL DEFAULT 2,
  name_i18n jsonb NOT NULL DEFAULT '{}'::jsonb,
  instruction_i18n jsonb NOT NULL DEFAULT '{}'::jsonb,
  detection_config jsonb NOT NULL DEFAULT '{}'::jsonb,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS name_th text NOT NULL DEFAULT '';
ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS body_side text NOT NULL DEFAULT 'RIGHT';
ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS instructions_th text NOT NULL DEFAULT '';
ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS target_rom_deg real NOT NULL DEFAULT 135;
ALTER TABLE public.exercises ADD COLUMN IF NOT EXISTS hold_sec real NOT NULL DEFAULT 2;

-- 6. Modpacks
CREATE TABLE IF NOT EXISTS public.modpacks (
  id text PRIMARY KEY,
  name text NOT NULL,
  category text NOT NULL DEFAULT 'general',
  card_count int NOT NULL DEFAULT 10,
  tags text[] NOT NULL DEFAULT ARRAY[]::text[],
  categories jsonb NOT NULL DEFAULT '[]'::jsonb,
  questions jsonb NOT NULL DEFAULT '[]'::jsonb,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.modpacks ADD COLUMN IF NOT EXISTS category text NOT NULL DEFAULT 'general';
ALTER TABLE public.modpacks ADD COLUMN IF NOT EXISTS card_count int NOT NULL DEFAULT 10;
ALTER TABLE public.modpacks ADD COLUMN IF NOT EXISTS tags text[] NOT NULL DEFAULT ARRAY[]::text[];

-- 7. Patient Courses & Detailed Rehab Schedules
CREATE TABLE IF NOT EXISTS public.patient_courses (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  title text NOT NULL,
  weekly_schedule jsonb NOT NULL DEFAULT '[]'::jsonb,
  cognitive_modpack_id text REFERENCES public.modpacks(id) ON DELETE SET NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);

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

-- 8. Rehab Sessions
CREATE TABLE IF NOT EXISTS public.rehab_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  exercise_id text NOT NULL,
  completed_sets int NOT NULL DEFAULT 0,
  completed_reps int NOT NULL DEFAULT 0,
  accuracy_percent real NOT NULL DEFAULT 100,
  max_rom_deg real NOT NULL DEFAULT 0,
  pain_stop_triggered boolean NOT NULL DEFAULT false,
  kinematics_replay jsonb NOT NULL DEFAULT '[]'::jsonb,
  minigame_score jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- 9. Staff Calls & Emergency Alerts
CREATE TABLE IF NOT EXISTS public.staff_calls (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kiosk_id text NOT NULL,
  patient_id uuid REFERENCES public.patients(id) ON DELETE SET NULL,
  reason text NOT NULL,
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acknowledged', 'resolved')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.emergency_alerts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kiosk_id text NOT NULL,
  patient_id uuid REFERENCES public.patients(id) ON DELETE SET NULL,
  reason text NOT NULL,
  status text NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'ACKNOWLEDGED', 'RESOLVED')),
  created_at timestamptz NOT NULL DEFAULT now()
);

-- 10. Audit Logs
CREATE TABLE IF NOT EXISTS public.audit_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type text NOT NULL,
  actor_role text NOT NULL,
  actor_id text NOT NULL,
  actor_name text NOT NULL DEFAULT '',
  action text,
  target text,
  details jsonb DEFAULT '{}'::jsonb,
  ip_address text NOT NULL DEFAULT 'unknown',
  device_id text NOT NULL DEFAULT 'unknown',
  target_id text,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS actor_name text NOT NULL DEFAULT '';
ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS action text;
ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS target text;
ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS details jsonb DEFAULT '{}'::jsonb;
ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS ip_address text NOT NULL DEFAULT 'unknown';
ALTER TABLE public.audit_logs ADD COLUMN IF NOT EXISTS device_id text NOT NULL DEFAULT 'unknown';
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON public.audit_logs (created_at DESC);

-- 11. Medical Notes & Treatment Reports
CREATE TABLE IF NOT EXISTS public.medical_notes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  author_id text NOT NULL,
  author_name text NOT NULL,
  content text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.treatment_reports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  pt_id text NOT NULL,
  pt_name text NOT NULL,
  summary text NOT NULL,
  ai_recommendation text NOT NULL DEFAULT '',
  score_summary text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- 12. Bans, Sessions, System Settings & Kiosk Devices
CREATE TABLE IF NOT EXISTS public.banned_targets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  target_type text NOT NULL CHECK (target_type IN ('ip', 'device')),
  value text NOT NULL,
  reason text NOT NULL DEFAULT '',
  banned_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.banned_entities (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_type text NOT NULL CHECK (entity_type IN ('ip', 'device', 'user')),
  value text NOT NULL,
  reason text NOT NULL DEFAULT '',
  banned_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.active_sessions (
  token text PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES public.staff_users(id) ON DELETE CASCADE,
  role text NOT NULL,
  ip_address text NOT NULL DEFAULT 'unknown',
  device_id text NOT NULL DEFAULT 'unknown',
  created_at timestamptz NOT NULL DEFAULT now(),
  revoked boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS public.system_settings (
  id int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  key text,
  value jsonb,
  ai_face_scan_enabled boolean NOT NULL DEFAULT true,
  ai_face_threshold real NOT NULL DEFAULT 0.925,
  ai_pose_detection_enabled boolean NOT NULL DEFAULT false,
  kiosk_auto_reset_sec int NOT NULL DEFAULT 6,
  updated_by uuid,
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.system_settings ADD COLUMN IF NOT EXISTS key text;
ALTER TABLE public.system_settings ADD COLUMN IF NOT EXISTS value jsonb;
ALTER TABLE public.system_settings ADD COLUMN IF NOT EXISTS updated_by uuid;

INSERT INTO public.system_settings (id, key, value, ai_face_scan_enabled, ai_face_threshold, ai_pose_detection_enabled, kiosk_auto_reset_sec)
VALUES (1, 'ai_pose_detection_unlocked', 'false'::jsonb, true, 0.925, false, 6)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS public.kiosks (
  kiosk_id text PRIMARY KEY,
  location text NOT NULL DEFAULT '',
  online boolean NOT NULL DEFAULT true,
  current_state text NOT NULL DEFAULT 'STANDBY',
  pairing_token text,
  last_heartbeat_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.kiosks ADD COLUMN IF NOT EXISTS pairing_token text;

CREATE TABLE IF NOT EXISTS public.kiosk_devices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kiosk_code text UNIQUE NOT NULL,
  location text NOT NULL DEFAULT '',
  online_status boolean NOT NULL DEFAULT true,
  pairing_token text,
  last_heartbeat timestamptz NOT NULL DEFAULT now()
);

-- Seed Default Accounts, Modpacks, Kiosks (Idempotent)
INSERT INTO public.staff_users (id, username, password_hash, full_name, phone, specialty, license_no, license_number, role, status, active)
VALUES
  ('00000000-0000-4000-8000-000000000001', 'admin', 'pbkdf2:dceb3f63b5ff16065454e730ac190076:405a025959bb0a88efc4ee2b0d7f1cc91f8c274e286980cb053b1409d17626aa', 'นพ.สมชาย ผู้อำนวยการโรงพยาบาล (Super Admin)', '02-000-0001', 'Hospital Director', 'MD-10001', 'MD-10001', 'super_admin', 'ACTIVE', true),
  ('00000000-0000-4000-8000-000000000002', 'pt_somchai', 'pbkdf2:61d9355274f9c2334e628def69572f75:f9129631c2636dd24bccf0efbb017453d4c36b0a0d787c7c4a6886a7c0c91820', 'กภ.สมชาย ใจดี (PT)', '081-111-2233', 'Orthopedic & Shoulder Rehab', 'PT-44012', 'PT-44012', 'pt', 'ACTIVE', true),
  ('00000000-0000-4000-8000-000000000003', 'pt_waranya', 'pbkdf2:864c6fb40a97587a6f5bcf2c59440d91:3209e7c3aca7ae8b21fd98e3dd36e4c3acb9c9f549beaa99dfe2e2c1243da077', 'กภ.วรัญญา สุขสันต์ (PT)', '089-222-3344', 'Neuro & Cognitive Rehab', 'PT-44089', 'PT-44089', 'pt', 'ACTIVE', true)
ON CONFLICT (username) DO NOTHING;

INSERT INTO public.modpacks (id, name, category, card_count, categories, questions, active)
VALUES
  ('modpack-core-99', 'Clinical Cognitive Core (99 Cards)', 'cognitive', 99, '["food_fruit","objects","animals","counting","safety_routine","color_shape_time","position"]'::jsonb, '[]'::jsonb, true),
  ('modpack-memory-safety', 'Fall Prevention & Daily Safety Pack', 'safety', 42, '["safety_routine","counting","position"]'::jsonb, '[]'::jsonb, true)
ON CONFLICT (id) DO NOTHING;

INSERT INTO public.kiosks (kiosk_id, location, online, current_state, pairing_token)
VALUES ('KIOSK-OPD-01', 'OPD Rehab Zone A', true, 'STANDBY', 'KPAIR-OPD01')
ON CONFLICT (kiosk_id) DO NOTHING;

-- RLS Policies
ALTER TABLE public.staff_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.face_enrollments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.face_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.exercises ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.modpacks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.patient_courses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rehab_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rehab_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.staff_calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.emergency_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.medical_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.treatment_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.banned_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.banned_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.active_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.system_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.kiosks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.kiosk_devices ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS staff_all_staff_users ON public.staff_users;
CREATE POLICY staff_all_staff_users ON public.staff_users FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_patients ON public.patients;
CREATE POLICY staff_all_patients ON public.patients FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_face_enrollments ON public.face_enrollments;
CREATE POLICY staff_all_face_enrollments ON public.face_enrollments FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_face_templates ON public.face_templates;
CREATE POLICY staff_all_face_templates ON public.face_templates FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_courses ON public.patient_courses;
CREATE POLICY staff_all_courses ON public.patient_courses FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_rehab_schedules ON public.rehab_schedules;
CREATE POLICY staff_all_rehab_schedules ON public.rehab_schedules FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_sessions ON public.rehab_sessions;
CREATE POLICY staff_all_sessions ON public.rehab_sessions FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_calls ON public.staff_calls;
CREATE POLICY staff_all_calls ON public.staff_calls FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_emergency_alerts ON public.emergency_alerts;
CREATE POLICY staff_all_emergency_alerts ON public.emergency_alerts FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_audit_logs ON public.audit_logs;
CREATE POLICY staff_all_audit_logs ON public.audit_logs FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_medical_notes ON public.medical_notes;
CREATE POLICY staff_all_medical_notes ON public.medical_notes FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_treatment_reports ON public.treatment_reports;
CREATE POLICY staff_all_treatment_reports ON public.treatment_reports FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_banned_targets ON public.banned_targets;
CREATE POLICY staff_all_banned_targets ON public.banned_targets FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_banned_entities ON public.banned_entities;
CREATE POLICY staff_all_banned_entities ON public.banned_entities FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_active_sessions ON public.active_sessions;
CREATE POLICY staff_all_active_sessions ON public.active_sessions FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_system_settings ON public.system_settings;
CREATE POLICY staff_all_system_settings ON public.system_settings FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_kiosks ON public.kiosks;
CREATE POLICY staff_all_kiosks ON public.kiosks FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS staff_all_kiosk_devices ON public.kiosk_devices;
CREATE POLICY staff_all_kiosk_devices ON public.kiosk_devices FOR ALL USING (auth.role() IN ('service_role', 'authenticated'));
DROP POLICY IF EXISTS public_read_exercises ON public.exercises;
CREATE POLICY public_read_exercises ON public.exercises FOR SELECT USING (true);
DROP POLICY IF EXISTS public_read_modpacks ON public.modpacks;
CREATE POLICY public_read_modpacks ON public.modpacks FOR SELECT USING (true);
DROP POLICY IF EXISTS staff_read_calls_realtime ON public.staff_calls;
CREATE POLICY staff_read_calls_realtime ON public.staff_calls FOR SELECT USING (true);

CREATE OR REPLACE FUNCTION public.match_patient_face(
  query_embedding vector(512),
  match_count int DEFAULT 5
)
RETURNS TABLE (
  patient_id uuid,
  full_name text,
  pose text,
  embedding float4[],
  raw_cosine float8
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    ft.patient_id,
    p.full_name,
    ft.pose,
    ft.embedding::float4[] AS embedding,
    (1 - (ft.embedding <=> query_embedding))::float8 AS raw_cosine
  FROM public.face_templates ft
  INNER JOIN public.patients p ON p.id = ft.patient_id
  WHERE p.registration_status = 'active'
    AND ft.pose IN ('frontal_centroid', 'front')
  ORDER BY ft.embedding <=> query_embedding ASC
  LIMIT GREATEST(1, LEAST(match_count, 20));
$$;

