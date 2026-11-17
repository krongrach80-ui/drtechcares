-- ============================================================================
-- Migration 005: Permanent Cloud Biometric Vault & Live Kiosk RLS Policies
-- Guarantees permanent cross-day/cross-device face template storage on
-- public.patients and public.patient_biometrics with PostgREST schema reload.
-- ============================================================================

-- 1. Guarantee public.patients exists and has all permanent biometric columns
CREATE TABLE IF NOT EXISTS public.patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hn TEXT UNIQUE,
    full_name TEXT NOT NULL DEFAULT '',
    age INTEGER NOT NULL DEFAULT 0,
    phone TEXT NOT NULL DEFAULT '',
    diagnosis TEXT NOT NULL DEFAULT '',
    registration_status TEXT DEFAULT 'active',
    face_embedding JSONB,
    biometric_data JSONB,
    biometric_updated_at TIMESTAMPTZ,
    face_biometric_status TEXT DEFAULT 'pending',
    face_frames_count INTEGER DEFAULT 0,
    face_image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.patients
    ADD COLUMN IF NOT EXISTS face_embedding JSONB,
    ADD COLUMN IF NOT EXISTS biometric_data JSONB,
    ADD COLUMN IF NOT EXISTS biometric_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS face_biometric_status TEXT DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS face_frames_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS face_image_url TEXT;

CREATE INDEX IF NOT EXISTS idx_patients_biometric_status
    ON public.patients (face_biometric_status)
    WHERE biometric_data IS NOT NULL OR face_embedding IS NOT NULL;

-- 2. Guarantee public.patient_biometrics exists and has all 15-exemplar / 5-angle columns
CREATE TABLE IF NOT EXISTS public.patient_biometrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL UNIQUE REFERENCES public.patients(id) ON DELETE CASCADE,
    hn TEXT,
    master_embedding JSONB NOT NULL DEFAULT '[]'::jsonb,
    angle_embeddings JSONB NOT NULL DEFAULT '{}'::jsonb,
    exemplar_embeddings JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding_dim INTEGER NOT NULL DEFAULT 1024,
    frame_count INTEGER NOT NULL DEFAULT 0,
    frames_captured INTEGER NOT NULL DEFAULT 0,
    model_version TEXT NOT NULL DEFAULT 'human-faceres-1024d',
    biometric_engine TEXT NOT NULL DEFAULT 'human-faceres-1024d',
    quality_score NUMERIC DEFAULT 0,
    avatar_url TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.patient_biometrics
    ADD COLUMN IF NOT EXISTS hn TEXT,
    ADD COLUMN IF NOT EXISTS master_embedding JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS angle_embeddings JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS exemplar_embeddings JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS embedding_dim INTEGER NOT NULL DEFAULT 1024,
    ADD COLUMN IF NOT EXISTS frame_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS frames_captured INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS model_version TEXT NOT NULL DEFAULT 'human-faceres-1024d',
    ADD COLUMN IF NOT EXISTS biometric_engine TEXT NOT NULL DEFAULT 'human-faceres-1024d',
    ADD COLUMN IF NOT EXISTS quality_score NUMERIC DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avatar_url TEXT,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_biometrics_patient_id_unique
    ON public.patient_biometrics (patient_id);

CREATE INDEX IF NOT EXISTS idx_patient_biometrics_hn
    ON public.patient_biometrics (hn);

-- 3. Enable RLS with explicit Kiosk/Admin policies for anon + authenticated + service_role
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

-- 4. Enable Realtime replication cleanly (idempotent)
DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE public.patients;
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN undefined_object THEN NULL;
    WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE public.patient_biometrics;
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN undefined_object THEN NULL;
    WHEN OTHERS THEN NULL;
END $$;

-- 5. Force PostgREST to reload schema cache immediately
NOTIFY pgrst, 'reload schema';

