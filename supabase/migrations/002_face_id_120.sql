-- ============================================================
-- Dr.Tech.Care 120-Frame iPhone-Style 3D Face ID Schema Migration
-- File: supabase/migrations/002_face_id_120.sql
-- ============================================================

-- 1. Ensure extension for UUID generation exists
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 2. Upgrade patients table with 120-frame biometric metadata columns
ALTER TABLE IF EXISTS public.patients
  ADD COLUMN IF NOT EXISTS face_biometric_status TEXT DEFAULT 'PENDING',
  ADD COLUMN IF NOT EXISTS face_frames_count INT DEFAULT 0,
  ADD COLUMN IF NOT EXISTS face_image_url TEXT;

-- Backfill existing verified rows if templates exist
UPDATE public.patients p
SET
  face_biometric_status = 'VERIFIED',
  face_frames_count = COALESCE(NULLIF(p.face_frames_count, 0), 120)
WHERE EXISTS (
  SELECT 1 FROM public.face_templates ft WHERE ft.patient_id = p.id
);

-- 3. Create patient_biometrics table for 5-angle + master 120-frame centroids
CREATE TABLE IF NOT EXISTS public.patient_biometrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE UNIQUE,
  hn TEXT NOT NULL,
  master_embedding JSONB NOT NULL,
  angle_embeddings JSONB NOT NULL DEFAULT '{}'::jsonb,
  frames_captured INT NOT NULL DEFAULT 120,
  quality_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  avatar_url TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patient_biometrics_patient_id ON public.patient_biometrics(patient_id);
CREATE INDEX IF NOT EXISTS idx_patient_biometrics_hn ON public.patient_biometrics(hn);

-- 4. Enable Row Level Security & Realtime policies
ALTER TABLE public.patient_biometrics ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow Kiosk and public read patient_biometrics" ON public.patient_biometrics;
CREATE POLICY "Allow Kiosk and public read patient_biometrics"
  ON public.patient_biometrics
  FOR SELECT
  TO anon, authenticated, service_role
  USING (true);

DROP POLICY IF EXISTS "Allow staff insert patient_biometrics" ON public.patient_biometrics;
CREATE POLICY "Allow staff insert patient_biometrics"
  ON public.patient_biometrics
  FOR INSERT
  TO anon, authenticated, service_role
  WITH CHECK (true);

DROP POLICY IF EXISTS "Allow staff update patient_biometrics" ON public.patient_biometrics;
CREATE POLICY "Allow staff update patient_biometrics"
  ON public.patient_biometrics
  FOR UPDATE
  TO anon, authenticated, service_role
  USING (true)
  WITH CHECK (true);

DROP POLICY IF EXISTS "Allow staff delete patient_biometrics" ON public.patient_biometrics;
CREATE POLICY "Allow staff delete patient_biometrics"
  ON public.patient_biometrics
  FOR DELETE
  TO anon, authenticated, service_role
  USING (true);

-- 5. Enable Realtime replication on patient_biometrics if publication exists
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
    BEGIN
      ALTER PUBLICATION supabase_realtime ADD TABLE public.patient_biometrics;
    EXCEPTION WHEN duplicate_object THEN
      NULL;
    END;
  END IF;
END $$;

