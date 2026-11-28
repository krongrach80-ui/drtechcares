-- ============================================================================
-- Migration 009: Backfill Patient Biometrics & Profile Fields From history JSON
-- Syncs top-level public.patients columns (face_biometric_status,
-- face_frames_count, face_image_url, master_embedding, angle_embeddings,
-- face_embedding, biometric_data, biometric_updated_at, gender, care_group,
-- affected_side, motor_assessment_score, adl_dressing_score) whenever data was
-- written inside patients.history JSONB while top-level columns stayed empty
-- or stuck at PENDING / 0.
-- Safe, idempotent, zero destructive operations.
-- ============================================================================

-- 1. Ensure all target top-level biometric & clinical profile columns exist
ALTER TABLE IF EXISTS public.patients
  ADD COLUMN IF NOT EXISTS face_biometric_status TEXT DEFAULT 'PENDING',
  ADD COLUMN IF NOT EXISTS face_frames_count INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS face_image_url TEXT,
  ADD COLUMN IF NOT EXISTS master_embedding JSONB,
  ADD COLUMN IF NOT EXISTS angle_embeddings JSONB,
  ADD COLUMN IF NOT EXISTS face_embedding JSONB,
  ADD COLUMN IF NOT EXISTS biometric_data JSONB,
  ADD COLUMN IF NOT EXISTS biometric_updated_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS gender TEXT,
  ADD COLUMN IF NOT EXISTS care_group TEXT,
  ADD COLUMN IF NOT EXISTS affected_side TEXT,
  ADD COLUMN IF NOT EXISTS motor_assessment_score NUMERIC,
  ADD COLUMN IF NOT EXISTS adl_dressing_score NUMERIC;

-- 2. Backfill top-level public.patients columns from history JSONB safely
WITH parsed AS (
  SELECT
    p.id,
    (
      jsonb_typeof(p.history -> 'masterEmbedding') = 'array'
      AND jsonb_array_length(p.history -> 'masterEmbedding') > 0
    ) AS has_hist_master,
    (
      jsonb_typeof(p.history -> 'angleEmbeddings') = 'object'
      AND (p.history -> 'angleEmbeddings') <> '{}'::jsonb
    ) AS has_hist_angles,
    (
      CASE
        WHEN (p.history ->> 'faceFramesCount') ~ '^\d+$'
          THEN (p.history ->> 'faceFramesCount')::int
        ELSE 0
      END
    ) AS hist_frames,
    UPPER(BTRIM(COALESCE(p.history ->> 'faceBiometricStatus', ''))) AS hist_status,
    NULLIF(BTRIM(COALESCE(p.history ->> 'faceImageUrl', '')), '') AS hist_face_image_url,
    NULLIF(BTRIM(COALESCE(p.history ->> 'gender', '')), '') AS hist_gender,
    CASE
      WHEN UPPER(BTRIM(COALESCE(p.history ->> 'careGroup', '')))
        IN ('MOBILITY_RECOVERY', 'CLOSE_CARE', 'ELDERLY_GENERAL', 'ORTHOPEDIC', 'NEURO')
      THEN UPPER(BTRIM(p.history ->> 'careGroup'))
      ELSE NULL
    END AS hist_care_group,
    CASE
      WHEN UPPER(BTRIM(COALESCE(p.history ->> 'affectedSide', '')))
        IN ('LEFT', 'RIGHT', 'BOTH', 'NONE')
      THEN UPPER(BTRIM(p.history ->> 'affectedSide'))
      ELSE NULL
    END AS hist_affected_side,
    CASE
      WHEN (p.history ->> 'motorAssessmentScore') ~ '^\d+(\.\d+)?$'
        AND (p.history ->> 'motorAssessmentScore')::numeric BETWEEN 0 AND 6
      THEN (p.history ->> 'motorAssessmentScore')::numeric
      ELSE NULL
    END AS hist_motor_score,
    CASE
      WHEN (p.history ->> 'adlDressingScore') ~ '^\d+(\.\d+)?$'
        AND (p.history ->> 'adlDressingScore')::numeric BETWEEN 0 AND 5
      THEN (p.history ->> 'adlDressingScore')::numeric
      ELSE NULL
    END AS hist_adl_score,
    COALESCE(
      NULLIF(BTRIM(COALESCE(p.history ->> 'biometricEngine', '')), ''),
      CASE
        WHEN jsonb_typeof(p.history -> 'masterEmbedding') = 'array'
          AND jsonb_array_length(p.history -> 'masterEmbedding') >= 768
          THEN 'human-faceres-1024d'
        WHEN jsonb_typeof(p.history -> 'masterEmbedding') = 'array'
          AND jsonb_array_length(p.history -> 'masterEmbedding') > 0
          THEN 'hybrid-512d'
        ELSE NULL
      END
    ) AS hist_engine
  FROM public.patients p
  WHERE p.history IS NOT NULL
    AND jsonb_typeof(p.history) = 'object'
)
UPDATE public.patients p
SET
  face_biometric_status = CASE
    WHEN parsed.hist_status = 'VERIFIED'
      OR (parsed.hist_frames > 0 AND parsed.has_hist_master)
      OR parsed.has_hist_master
      THEN 'VERIFIED'
    ELSE p.face_biometric_status
  END,
  face_frames_count = COALESCE(
    NULLIF(p.face_frames_count, 0),
    NULLIF(parsed.hist_frames, 0),
    CASE WHEN parsed.has_hist_master OR parsed.hist_status = 'VERIFIED' THEN 120 ELSE 0 END,
    0
  ),
  face_image_url = COALESCE(
    NULLIF(BTRIM(COALESCE(p.face_image_url, '')), ''),
    parsed.hist_face_image_url
  ),
  master_embedding = CASE
    WHEN (p.master_embedding IS NULL OR p.master_embedding IN ('[]'::jsonb, 'null'::jsonb))
      AND parsed.has_hist_master
      THEN p.history -> 'masterEmbedding'
    ELSE p.master_embedding
  END,
  angle_embeddings = CASE
    WHEN (p.angle_embeddings IS NULL OR p.angle_embeddings IN ('{}'::jsonb, 'null'::jsonb))
      AND parsed.has_hist_angles
      THEN p.history -> 'angleEmbeddings'
    ELSE p.angle_embeddings
  END,
  biometric_data = CASE
    WHEN (p.biometric_data IS NULL OR p.biometric_data IN ('{}'::jsonb, 'null'::jsonb))
      AND (parsed.has_hist_master OR parsed.has_hist_angles)
      THEN jsonb_strip_nulls(
        jsonb_build_object(
          'engine', parsed.hist_engine,
          'dimension', CASE
            WHEN parsed.has_hist_master THEN jsonb_array_length(p.history -> 'masterEmbedding')
            ELSE NULL
          END,
          'master_embedding', CASE
            WHEN parsed.has_hist_master THEN p.history -> 'masterEmbedding'
            ELSE NULL
          END,
          'angle_embeddings', CASE
            WHEN parsed.has_hist_angles THEN p.history -> 'angleEmbeddings'
            ELSE NULL
          END,
          'exemplar_embeddings', CASE
            WHEN jsonb_typeof(p.history -> 'exemplarEmbeddings') = 'array'
              THEN p.history -> 'exemplarEmbeddings'
            ELSE '[]'::jsonb
          END,
          'geometric_ratios', CASE
            WHEN jsonb_typeof(p.history -> 'geometricRatios') = 'object'
              THEN p.history -> 'geometricRatios'
            ELSE NULL
          END,
          'frames_captured', COALESCE(
            NULLIF(p.face_frames_count, 0),
            NULLIF(parsed.hist_frames, 0),
            120
          ),
          'updated_at', COALESCE(p.biometric_updated_at, p.updated_at, now())
        )
      )
    ELSE p.biometric_data
  END,
  face_embedding = CASE
    WHEN (p.face_embedding IS NULL OR p.face_embedding IN ('{}'::jsonb, 'null'::jsonb))
      AND (parsed.has_hist_master OR parsed.has_hist_angles)
      THEN jsonb_strip_nulls(
        jsonb_build_object(
          'engine', parsed.hist_engine,
          'dimension', CASE
            WHEN parsed.has_hist_master THEN jsonb_array_length(p.history -> 'masterEmbedding')
            ELSE NULL
          END,
          'master_embedding', CASE
            WHEN parsed.has_hist_master THEN p.history -> 'masterEmbedding'
            ELSE NULL
          END,
          'angle_embeddings', CASE
            WHEN parsed.has_hist_angles THEN p.history -> 'angleEmbeddings'
            ELSE NULL
          END,
          'exemplar_embeddings', CASE
            WHEN jsonb_typeof(p.history -> 'exemplarEmbeddings') = 'array'
              THEN p.history -> 'exemplarEmbeddings'
            ELSE '[]'::jsonb
          END,
          'geometric_ratios', CASE
            WHEN jsonb_typeof(p.history -> 'geometricRatios') = 'object'
              THEN p.history -> 'geometricRatios'
            ELSE NULL
          END
        )
      )
    ELSE p.face_embedding
  END,
  biometric_updated_at = CASE
    WHEN parsed.has_hist_master
      OR parsed.has_hist_angles
      OR parsed.hist_status = 'VERIFIED'
      OR UPPER(COALESCE(p.face_biometric_status, '')) = 'VERIFIED'
      THEN COALESCE(p.biometric_updated_at, p.updated_at, now())
    ELSE p.biometric_updated_at
  END,
  gender = COALESCE(NULLIF(BTRIM(COALESCE(p.gender, '')), ''), parsed.hist_gender),
  care_group = COALESCE(NULLIF(BTRIM(COALESCE(p.care_group, '')), ''), parsed.hist_care_group),
  affected_side = COALESCE(NULLIF(BTRIM(COALESCE(p.affected_side, '')), ''), parsed.hist_affected_side),
  motor_assessment_score = COALESCE(p.motor_assessment_score, parsed.hist_motor_score),
  adl_dressing_score = COALESCE(p.adl_dressing_score, parsed.hist_adl_score),
  updated_at = COALESCE(p.updated_at, now())
FROM parsed
WHERE p.id = parsed.id
  AND (
    (
      UPPER(COALESCE(p.face_biometric_status, '')) <> 'VERIFIED'
      AND (
        parsed.hist_status = 'VERIFIED'
        OR (parsed.hist_frames > 0 AND parsed.has_hist_master)
        OR parsed.has_hist_master
      )
    )
    OR (COALESCE(p.face_frames_count, 0) = 0 AND (parsed.hist_frames > 0 OR parsed.has_hist_master))
    OR (NULLIF(BTRIM(COALESCE(p.face_image_url, '')), '') IS NULL AND parsed.hist_face_image_url IS NOT NULL)
    OR ((p.master_embedding IS NULL OR p.master_embedding IN ('[]'::jsonb, 'null'::jsonb)) AND parsed.has_hist_master)
    OR ((p.angle_embeddings IS NULL OR p.angle_embeddings IN ('{}'::jsonb, 'null'::jsonb)) AND parsed.has_hist_angles)
    OR ((p.biometric_data IS NULL OR p.biometric_data IN ('{}'::jsonb, 'null'::jsonb)) AND (parsed.has_hist_master OR parsed.has_hist_angles))
    OR ((p.face_embedding IS NULL OR p.face_embedding IN ('{}'::jsonb, 'null'::jsonb)) AND (parsed.has_hist_master OR parsed.has_hist_angles))
    OR (p.biometric_updated_at IS NULL AND (parsed.has_hist_master OR parsed.has_hist_angles OR parsed.hist_status = 'VERIFIED'))
    OR (NULLIF(BTRIM(COALESCE(p.gender, '')), '') IS NULL AND parsed.hist_gender IS NOT NULL)
    OR (NULLIF(BTRIM(COALESCE(p.care_group, '')), '') IS NULL AND parsed.hist_care_group IS NOT NULL)
    OR (NULLIF(BTRIM(COALESCE(p.affected_side, '')), '') IS NULL AND parsed.hist_affected_side IS NOT NULL)
    OR (p.motor_assessment_score IS NULL AND parsed.hist_motor_score IS NOT NULL)
    OR (p.adl_dressing_score IS NULL AND parsed.hist_adl_score IS NOT NULL)
  );

-- 3. Backfill public.patient_biometrics for any patient whose vectors existed only in patients.history
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'patient_biometrics'
  ) THEN
    INSERT INTO public.patient_biometrics (
      patient_id,
      hn,
      master_embedding,
      angle_embeddings,
      exemplar_embeddings,
      embedding_dim,
      frames_captured,
      biometric_engine,
      updated_at
    )
    SELECT
      p.id,
      p.hn,
      p.master_embedding,
      COALESCE(p.angle_embeddings, '{}'::jsonb),
      CASE
        WHEN jsonb_typeof(p.history -> 'exemplarEmbeddings') = 'array'
          THEN p.history -> 'exemplarEmbeddings'
        ELSE '[]'::jsonb
      END,
      jsonb_array_length(p.master_embedding),
      COALESCE(NULLIF(p.face_frames_count, 0), 120),
      COALESCE(
        NULLIF(BTRIM(COALESCE(p.history ->> 'biometricEngine', '')), ''),
        CASE WHEN jsonb_array_length(p.master_embedding) >= 768 THEN 'human-faceres-1024d' ELSE 'hybrid-512d' END
      ),
      COALESCE(p.biometric_updated_at, p.updated_at, now())
    FROM public.patients p
    WHERE p.master_embedding IS NOT NULL
      AND jsonb_typeof(p.master_embedding) = 'array'
      AND jsonb_array_length(p.master_embedding) >= 128
      AND NOT EXISTS (
        SELECT 1 FROM public.patient_biometrics pb WHERE pb.patient_id = p.id
      )
    ON CONFLICT (patient_id) DO NOTHING;
  END IF;
END $$;

-- 4. Reload PostgREST schema cache so newly synced columns are immediately queryable
NOTIFY pgrst, 'reload schema';

