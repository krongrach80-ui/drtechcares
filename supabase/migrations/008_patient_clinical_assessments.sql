-- ============================================================
-- Migration 008: Individual Patient Clinical Assessments (0-6 Scale)
-- Longitudinal ROM Comparison & Cross-Facility Record Linking
-- File: supabase/migrations/008_patient_clinical_assessments.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS public.patient_clinical_assessments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id text NOT NULL,
  patient_hn text NOT NULL DEFAULT '',
  universal_link_code text NOT NULL DEFAULT '',
  visit_number integer NOT NULL DEFAULT 1,
  assessed_at timestamptz NOT NULL DEFAULT now(),
  facility_name text NOT NULL DEFAULT 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
  assessor_staff_id text NOT NULL DEFAULT '',
  assessor_staff_name text NOT NULL DEFAULT '',
  exercise_code text NOT NULL,
  exercise_name_th text NOT NULL DEFAULT '',
  target_side text NOT NULL DEFAULT 'RIGHT' CHECK (target_side IN ('LEFT', 'RIGHT', 'BOTH')),
  initial_score_0_to_6 integer NOT NULL CHECK (initial_score_0_to_6 >= 0 AND initial_score_0_to_6 <= 6),
  measured_rom_deg numeric(5,1) NOT NULL DEFAULT 0,
  target_rom_deg numeric(5,1) NOT NULL DEFAULT 120.0,
  pain_vas_0_to_10 integer CHECK (pain_vas_0_to_10 IS NULL OR (pain_vas_0_to_10 >= 0 AND pain_vas_0_to_10 <= 10)),
  chief_complaint text NOT NULL DEFAULT '',
  clinical_handover_note text NOT NULL DEFAULT '',
  next_session_goal text NOT NULL DEFAULT '',
  source_channel text NOT NULL DEFAULT 'STAFF_MANUAL_ASSESSMENT'
    CHECK (source_channel IN ('STAFF_MANUAL_ASSESSMENT', 'KIOSK_SESSION_AUTO_SYNC')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_patient_clinical_assessments_patient_id
  ON public.patient_clinical_assessments (patient_id);

CREATE INDEX IF NOT EXISTS idx_patient_clinical_assessments_patient_hn
  ON public.patient_clinical_assessments (patient_hn);

CREATE INDEX IF NOT EXISTS idx_patient_clinical_assessments_universal_link_code
  ON public.patient_clinical_assessments (universal_link_code);

CREATE INDEX IF NOT EXISTS idx_patient_clinical_assessments_patient_assessed_at
  ON public.patient_clinical_assessments (patient_id, assessed_at DESC);

ALTER TABLE public.patient_clinical_assessments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS drtechcares_service_all_patient_clinical_assessments ON public.patient_clinical_assessments;
CREATE POLICY drtechcares_service_all_patient_clinical_assessments
  ON public.patient_clinical_assessments
  FOR ALL
  TO public
  USING (true)
  WITH CHECK (true);

