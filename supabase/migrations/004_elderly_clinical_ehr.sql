-- ============================================================
-- Migration 004: Elderly Clinical Rehab, Mahidol 0-6 Scale & Cross-Clinic EHR
-- File: supabase/migrations/004_elderly_clinical_ehr.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. Extend public.patients with Elderly & ADL Rehabilitation clinical fields
ALTER TABLE IF EXISTS public.patients
  ADD COLUMN IF NOT EXISTS care_group text NOT NULL DEFAULT 'ELDERLY_GENERAL',
  ADD COLUMN IF NOT EXISTS affected_side text NOT NULL DEFAULT 'NONE',
  ADD COLUMN IF NOT EXISTS motor_assessment_score smallint NOT NULL DEFAULT 3,
  ADD COLUMN IF NOT EXISTS adl_dressing_score smallint NOT NULL DEFAULT 3,
  ADD COLUMN IF NOT EXISTS max_safe_rom_deg numeric NOT NULL DEFAULT 150,
  ADD COLUMN IF NOT EXISTS universal_health_link_code text;

ALTER TABLE IF EXISTS public.patients DROP CONSTRAINT IF EXISTS patients_care_group_check;
ALTER TABLE IF EXISTS public.patients ADD CONSTRAINT patients_care_group_check
  CHECK (care_group IN ('MOBILITY_RECOVERY', 'CLOSE_CARE', 'ELDERLY_GENERAL', 'ORTHOPEDIC', 'NEURO'));

ALTER TABLE IF EXISTS public.patients DROP CONSTRAINT IF EXISTS patients_affected_side_check;
ALTER TABLE IF EXISTS public.patients ADD CONSTRAINT patients_affected_side_check
  CHECK (affected_side IN ('LEFT', 'RIGHT', 'BOTH', 'NONE'));

ALTER TABLE IF EXISTS public.patients DROP CONSTRAINT IF EXISTS patients_motor_assessment_score_check;
ALTER TABLE IF EXISTS public.patients ADD CONSTRAINT patients_motor_assessment_score_check
  CHECK (motor_assessment_score BETWEEN 0 AND 6);

ALTER TABLE IF EXISTS public.patients DROP CONSTRAINT IF EXISTS patients_adl_dressing_score_check;
ALTER TABLE IF EXISTS public.patients ADD CONSTRAINT patients_adl_dressing_score_check
  CHECK (adl_dressing_score BETWEEN 0 AND 5);

CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_universal_health_link_code
  ON public.patients (universal_health_link_code)
  WHERE universal_health_link_code IS NOT NULL;

-- Backfill universal_health_link_code deterministically for existing patients lacking one
UPDATE public.patients
SET universal_health_link_code = 'DTC-EHR-' || UPPER(SUBSTRING(REPLACE(id::text, '-', ''), 1, 8))
WHERE universal_health_link_code IS NULL;

-- 2. Longitudinal ROM & Mahidol Motor Assessment history table (Cross-Clinic compatible)
CREATE TABLE IF NOT EXISTS public.rehab_session_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  hn text NOT NULL DEFAULT '',
  session_date timestamptz NOT NULL DEFAULT now(),
  exercise_code text NOT NULL,
  exercise_name text NOT NULL DEFAULT '',
  joint_name text NOT NULL DEFAULT 'upper_limb',
  body_side text NOT NULL DEFAULT 'BOTH',
  baseline_rom_deg numeric NOT NULL DEFAULT 0,
  achieved_max_rom_deg numeric NOT NULL DEFAULT 0,
  rom_delta_deg numeric NOT NULL DEFAULT 0,
  motor_score_at_session smallint CHECK (motor_score_at_session BETWEEN 0 AND 6),
  valid_reps integer NOT NULL DEFAULT 0,
  compensations_count integer NOT NULL DEFAULT 0,
  facility_name text NOT NULL DEFAULT 'Dr.Tech.Care Clinic',
  recorded_by_pt_id text,
  recorded_by_pt_name text,
  clinical_notes text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rehab_session_history_patient_date
  ON public.rehab_session_history (patient_id, session_date DESC);

CREATE INDEX IF NOT EXISTS idx_rehab_session_history_exercise_code
  ON public.rehab_session_history (exercise_code);

-- 3. Emergency Fall / Balance Halt Alerts table (used by Elderly Safety Guard & Edge Vision)
CREATE TABLE IF NOT EXISTS public.emergency_alerts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kiosk_id text NOT NULL DEFAULT 'KIOSK-OPD-01',
  patient_id uuid REFERENCES public.patients(id) ON DELETE SET NULL,
  reason text NOT NULL,
  status text NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'ACKNOWLEDGED', 'RESOLVED')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_emergency_alerts_status_created
  ON public.emergency_alerts (status, created_at DESC);

-- 4. Enable RLS with permissive clinic service policies matching Migration 003
DO $$
DECLARE
  tbl text;
  policy_name text;
  tbl_list text[] := ARRAY['rehab_session_history', 'emergency_alerts'];
BEGIN
  FOREACH tbl IN ARRAY tbl_list
  LOOP
    IF to_regclass('public.' || tbl) IS NOT NULL THEN
      EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', tbl);
      policy_name := 'drtechcares_service_all_' || tbl;
      EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I', policy_name, tbl);
      EXECUTE format(
        'CREATE POLICY %I ON public.%I FOR ALL TO public USING (true) WITH CHECK (true)',
        policy_name,
        tbl
      );
    END IF;
  END LOOP;
END $$;

