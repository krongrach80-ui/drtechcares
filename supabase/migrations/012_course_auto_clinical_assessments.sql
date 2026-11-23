-- ============================================================
-- Migration 012: Support COURSE_AUTO_CALCULATED Assessments
-- Adds COURSE_AUTO_CALCULATED to source_channel CHECK constraint
-- and adds source_session_id for idempotent course sync deduplication.
-- File: supabase/migrations/012_course_auto_clinical_assessments.sql
-- ============================================================

ALTER TABLE public.patient_clinical_assessments
  DROP CONSTRAINT IF EXISTS patient_clinical_assessments_source_channel_check;

ALTER TABLE public.patient_clinical_assessments
  ADD CONSTRAINT patient_clinical_assessments_source_channel_check
  CHECK (
    source_channel IN (
      'STAFF_MANUAL_ASSESSMENT',
      'KIOSK_SESSION_AUTO_SYNC',
      'COURSE_AUTO_CALCULATED'
    )
  );

ALTER TABLE public.patient_clinical_assessments
  ADD COLUMN IF NOT EXISTS source_session_id text;

CREATE INDEX IF NOT EXISTS idx_patient_clinical_assessments_source_session
  ON public.patient_clinical_assessments (patient_id, source_session_id)
  WHERE source_session_id IS NOT NULL;

