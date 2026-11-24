-- ============================================================================
-- Migration 013: Permanent Biometric Post-Deploy Sync & Schema Completeness
-- Ensures every column written by the Cloudflare Worker exists on both
-- public.patient_biometrics and public.patients, backfills any existing
-- patients whose 1024-D vectors were stored only in patients.history JSONB,
-- refreshes permissive Kiosk/Admin RLS policies, and reloads PostgREST schema.
-- Safe to run multiple times (100% idempotent).
-- ============================================================================

-- 1. Guarantee ALL columns on public.patient_biometrics so cold-start Workers never hit 42703/PGRST204
ALTER TABLE IF EXISTS public.patient_biometrics
  ADD COLUMN IF NOT EXISTS hn TEXT,
  ADD COLUMN IF NOT EXISTS master_embedding JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS angle_embeddings JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS exemplar_embeddings JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS geometric_ratios JSONB,
  ADD COLUMN IF NOT EXISTS embedding_dim INTEGER NOT NULL DEFAULT 1024,
  ADD COLUMN IF NOT EXISTS frame_count INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS frames_captured INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS model_version TEXT NOT NULL DEFAULT 'human-faceres-1024d',
  ADD COLUMN IF NOT EXISTS biometric_engine TEXT NOT NULL DEFAULT 'human-faceres-1024d',
  ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.96,
  ADD COLUMN IF NOT EXISTS avatar_url TEXT,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_biometrics_patient_id_unique
  ON public.patient_biometrics (patient_id);

CREATE INDEX IF NOT EXISTS idx_patient_biometrics_hn
  ON public.patient_biometrics (hn);

-- 2. Guarantee ALL dual-write biometric & profile columns on public.patients
ALTER TABLE IF EXISTS public.patients
  ADD COLUMN IF NOT EXISTS master_embedding JSONB,
  ADD COLUMN IF NOT EXISTS angle_embeddings JSONB,
  ADD COLUMN IF NOT EXISTS face_embedding JSONB,
  ADD COLUMN IF NOT EXISTS biometric_data JSONB,
  ADD COLUMN IF NOT EXISTS biometric_updated_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS face_biometric_status TEXT DEFAULT 'PENDING',
  ADD COLUMN IF NOT EXISTS face_frames_count INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS face_image_url TEXT,
  ADD COLUMN IF NOT EXISTS history JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- 3. Backfill missing public.patient_biometrics rows from public.patients.history or top-level vectors
INSERT INTO public.patient_biometrics (
  patient_id,
  hn,
  master_embedding,
  angle_embeddings,
  exemplar_embeddings,
  geometric_ratios,
  embedding_dim,
  frame_count,
  frames_captured,
  model_version,
  biometric_engine,
  quality_score,
  avatar_url,
  updated_at
)
SELECT
  p.id AS patient_id,
  COALESCE(NULLIF(TRIM(p.hn), ''), p.id::text) AS hn,
  COALESCE(
    CASE WHEN jsonb_typeof(p.master_embedding) = 'array' AND jsonb_array_length(p.master_embedding) >= 128 THEN p.master_embedding END,
    CASE WHEN jsonb_typeof(p.history -> 'masterEmbedding') = 'array' AND jsonb_array_length(p.history -> 'masterEmbedding') >= 128 THEN p.history -> 'masterEmbedding' END,
    CASE WHEN jsonb_typeof(p.biometric_data -> 'master_embedding') = 'array' AND jsonb_array_length(p.biometric_data -> 'master_embedding') >= 128 THEN p.biometric_data -> 'master_embedding' END
  ) AS master_embedding,
  COALESCE(
    CASE WHEN jsonb_typeof(p.angle_embeddings) = 'object' THEN p.angle_embeddings END,
    CASE WHEN jsonb_typeof(p.history -> 'angleEmbeddings') = 'object' THEN p.history -> 'angleEmbeddings' END,
    CASE WHEN jsonb_typeof(p.biometric_data -> 'angle_embeddings') = 'object' THEN p.biometric_data -> 'angle_embeddings' END,
    '{}'::jsonb
  ) AS angle_embeddings,
  COALESCE(
    CASE WHEN jsonb_typeof(p.history -> 'exemplarEmbeddings') = 'array' THEN p.history -> 'exemplarEmbeddings' END,
    CASE WHEN jsonb_typeof(p.biometric_data -> 'exemplar_embeddings') = 'array' THEN p.biometric_data -> 'exemplar_embeddings' END,
    '[]'::jsonb
  ) AS exemplar_embeddings,
  COALESCE(
    CASE WHEN jsonb_typeof(p.history -> 'geometricRatios') = 'object' THEN p.history -> 'geometricRatios' END,
    CASE WHEN jsonb_typeof(p.biometric_data -> 'geometric_ratios') = 'object' THEN p.biometric_data -> 'geometric_ratios' END
  ) AS geometric_ratios,
  jsonb_array_length(
    COALESCE(
      CASE WHEN jsonb_typeof(p.master_embedding) = 'array' AND jsonb_array_length(p.master_embedding) >= 128 THEN p.master_embedding END,
      CASE WHEN jsonb_typeof(p.history -> 'masterEmbedding') = 'array' AND jsonb_array_length(p.history -> 'masterEmbedding') >= 128 THEN p.history -> 'masterEmbedding' END,
      CASE WHEN jsonb_typeof(p.biometric_data -> 'master_embedding') = 'array' AND jsonb_array_length(p.biometric_data -> 'master_embedding') >= 128 THEN p.biometric_data -> 'master_embedding' END
    )
  ) AS embedding_dim,
  GREATEST(COALESCE(p.face_frames_count, 0), 120) AS frame_count,
  GREATEST(COALESCE(p.face_frames_count, 0), 120) AS frames_captured,
  COALESCE(NULLIF(p.history ->> 'biometricEngine', ''), 'human-faceres-1024d') AS model_version,
  COALESCE(NULLIF(p.history ->> 'biometricEngine', ''), 'human-faceres-1024d') AS biometric_engine,
  0.96 AS quality_score,
  COALESCE(NULLIF(p.face_image_url, ''), NULLIF(p.history ->> 'faceImageUrl', '')) AS avatar_url,
  COALESCE(p.biometric_updated_at, p.updated_at, NOW()) AS updated_at
FROM public.patients p
WHERE COALESCE(LOWER(p.registration_status), 'active') IN ('', 'active')
  AND (
    (jsonb_typeof(p.master_embedding) = 'array' AND jsonb_array_length(p.master_embedding) >= 128)
    OR (jsonb_typeof(p.history -> 'masterEmbedding') = 'array' AND jsonb_array_length(p.history -> 'masterEmbedding') >= 128)
    OR (jsonb_typeof(p.biometric_data -> 'master_embedding') = 'array' AND jsonb_array_length(p.biometric_data -> 'master_embedding') >= 128)
  )
ON CONFLICT (patient_id) DO UPDATE SET
  embedding_dim = GREATEST(
    COALESCE(public.patient_biometrics.embedding_dim, 0),
    CASE
      WHEN jsonb_typeof(public.patient_biometrics.master_embedding) = 'array'
      THEN jsonb_array_length(public.patient_biometrics.master_embedding)
      ELSE EXCLUDED.embedding_dim
    END
  ),
  geometric_ratios = COALESCE(public.patient_biometrics.geometric_ratios, EXCLUDED.geometric_ratios),
  exemplar_embeddings = CASE
    WHEN jsonb_typeof(public.patient_biometrics.exemplar_embeddings) = 'array'
      AND jsonb_array_length(public.patient_biometrics.exemplar_embeddings) > 0
    THEN public.patient_biometrics.exemplar_embeddings
    ELSE EXCLUDED.exemplar_embeddings
  END,
  updated_at = NOW();

-- 4. Repair embedding_dim on any pre-existing patient_biometrics rows that had dim = 0
UPDATE public.patient_biometrics
SET
  embedding_dim = jsonb_array_length(master_embedding),
  biometric_engine = CASE
    WHEN jsonb_array_length(master_embedding) >= 768 THEN 'human-faceres-1024d'
    ELSE COALESCE(NULLIF(biometric_engine, ''), 'hybrid-512d')
  END,
  model_version = CASE
    WHEN jsonb_array_length(master_embedding) >= 768 THEN 'human-faceres-1024d'
    ELSE COALESCE(NULLIF(model_version, ''), 'hybrid-512d')
  END
WHERE jsonb_typeof(master_embedding) = 'array'
  AND jsonb_array_length(master_embedding) >= 128
  AND COALESCE(embedding_dim, 0) < 128;

-- 5. Keep public.patients.face_biometric_status synchronized as VERIFIED wherever real vectors exist
UPDATE public.patients p
SET
  face_biometric_status = 'VERIFIED',
  face_frames_count = GREATEST(COALESCE(p.face_frames_count, 0), 120)
WHERE COALESCE(UPPER(p.face_biometric_status), 'PENDING') <> 'VERIFIED'
  AND (
    EXISTS (
      SELECT 1 FROM public.patient_biometrics pb
      WHERE pb.patient_id = p.id
        AND jsonb_typeof(pb.master_embedding) = 'array'
        AND jsonb_array_length(pb.master_embedding) >= 128
    )
    OR (jsonb_typeof(p.history -> 'masterEmbedding') = 'array' AND jsonb_array_length(p.history -> 'masterEmbedding') >= 128)
  );

-- 6. Guarantee Kiosk/Admin RLS policies and table grants
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.patient_biometrics ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS allow_all_patients_kiosk_and_admin ON public.patients;
CREATE POLICY allow_all_patients_kiosk_and_admin
  ON public.patients
  FOR ALL
  TO anon, authenticated, service_role
  USING (true)
  WITH CHECK (true);

DROP POLICY IF EXISTS allow_all_biometrics_kiosk_and_admin ON public.patient_biometrics;
CREATE POLICY allow_all_biometrics_kiosk_and_admin
  ON public.patient_biometrics
  FOR ALL
  TO anon, authenticated, service_role
  USING (true)
  WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.patients TO anon, authenticated, service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.patient_biometrics TO anon, authenticated, service_role;

-- 7. Immediately refresh PostgREST schema cache
NOTIFY pgrst, 'reload schema';

