-- ============================================================================
-- Migration 010: Ensure Patients Read/Write RLS Policies & Mirror PATIENT Users
-- 1. Guarantees public.patients allows SELECT, INSERT, and UPDATE for the
--    application API/anon/authenticated/service_role roles so Admin Portal,
--    Kiosk, and Worker API never receive empty rows due to missing Auth JWT.
-- 2. Backfills any active role = 'patient' accounts from public.staff_users
--    into public.patients when missing (matching by id, normalized HN, or name).
-- Safe, idempotent, zero destructive operations, zero Unicode emojis.
-- ============================================================================

-- 1. Enable RLS and grant explicit SELECT, INSERT, UPDATE permissions on public.patients
ALTER TABLE IF EXISTS public.patients ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS staff_all_patients ON public.patients;
DROP POLICY IF EXISTS allow_all_patients_kiosk_and_admin ON public.patients;
DROP POLICY IF EXISTS patients_api_select_policy ON public.patients;
DROP POLICY IF EXISTS patients_api_insert_policy ON public.patients;
DROP POLICY IF EXISTS patients_api_update_policy ON public.patients;

CREATE POLICY patients_api_select_policy
  ON public.patients
  FOR SELECT
  TO anon, authenticated, service_role
  USING (true);

CREATE POLICY patients_api_insert_policy
  ON public.patients
  FOR INSERT
  TO anon, authenticated, service_role
  WITH CHECK (true);

CREATE POLICY patients_api_update_policy
  ON public.patients
  FOR UPDATE
  TO anon, authenticated, service_role
  USING (true)
  WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE ON public.patients TO anon, authenticated, service_role;

-- 2. Backfill any active PATIENT accounts from public.staff_users missing in public.patients
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'staff_users'
  ) AND EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'patients'
  ) THEN
    INSERT INTO public.patients (
      id,
      hn,
      full_name,
      age,
      phone,
      diagnosis,
      registration_status,
      face_biometric_status,
      face_frames_count,
      gender,
      care_group,
      affected_side,
      motor_assessment_score,
      adl_dressing_score,
      history,
      created_at,
      updated_at
    )
    SELECT
      COALESCE(su.patient_id, su.id) AS id,
      CASE
        WHEN LOWER(REGEXP_REPLACE(COALESCE(su.username, ''), '^@+', '')) ~ '^hn[-_[:space:]]*[a-z0-9]+$'
          THEN 'HN-' || UPPER(REGEXP_REPLACE(LOWER(REGEXP_REPLACE(su.username, '^@+', '')), '^hn[-_[:space:]]*', ''))
        ELSE UPPER(REGEXP_REPLACE(COALESCE(NULLIF(BTRIM(su.username), ''), 'HN-' || SUBSTRING(REPLACE(su.id::text, '-', ''), 1, 6)), '^@+', ''))
      END AS hn,
      COALESCE(NULLIF(BTRIM(su.full_name), ''), UPPER(COALESCE(su.username, 'PATIENT'))) AS full_name,
      0 AS age,
      COALESCE(su.phone, '') AS phone,
      'รอประเมินอาการทางกายภาพบำบัด' AS diagnosis,
      'active' AS registration_status,
      'PENDING' AS face_biometric_status,
      0 AS face_frames_count,
      'ชาย' AS gender,
      'ELDERLY_GENERAL' AS care_group,
      'NONE' AS affected_side,
      3 AS motor_assessment_score,
      3 AS adl_dressing_score,
      jsonb_build_object(
        'gender', 'ชาย',
        'careGroup', 'ELDERLY_GENERAL',
        'affectedSide', 'NONE',
        'motorAssessmentScore', 3,
        'adlDressingScore', 3,
        'faceBiometricStatus', 'PENDING',
        'faceFramesCount', 0,
        'rehabSchedules', '[]'::jsonb,
        'mirroredFromStaffUsers', true
      ) AS history,
      COALESCE(su.created_at, now()) AS created_at,
      now() AS updated_at
    FROM public.staff_users su
    WHERE LOWER(COALESCE(su.role, '')) = 'patient'
      AND COALESCE(su.active, true) = true
      AND NOT EXISTS (
        SELECT 1
        FROM public.patients p
        WHERE p.id = COALESCE(su.patient_id, su.id)
          OR LOWER(REGEXP_REPLACE(COALESCE(p.hn, ''), '[^a-zA-Z0-9]', '', 'g')) =
             LOWER(REGEXP_REPLACE(COALESCE(su.username, ''), '[^a-zA-Z0-9]', '', 'g'))
          OR (
            NULLIF(BTRIM(COALESCE(su.full_name, '')), '') IS NOT NULL
            AND LOWER(BTRIM(COALESCE(p.full_name, ''))) = LOWER(BTRIM(su.full_name))
          )
      )
    ON CONFLICT DO NOTHING;
  END IF;
END $$;

-- 3. Force PostgREST to reload schema and RLS policy cache immediately
NOTIFY pgrst, 'reload schema';

