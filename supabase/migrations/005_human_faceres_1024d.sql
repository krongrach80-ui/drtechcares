-- ============================================================
-- Dr.Tech.Care Shared Deep Face Biometric Engine Migration
-- Supports 1024-D Deep FaceRes Embeddings + Top-10 KNN Exemplars
-- File: supabase/migrations/005_human_faceres_1024d.sql
-- ============================================================

ALTER TABLE IF EXISTS public.patients
  ADD COLUMN IF NOT EXISTS biometric_data JSONB;

ALTER TABLE IF EXISTS public.patient_biometrics
  ADD COLUMN IF NOT EXISTS exemplar_embeddings JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS biometric_engine TEXT,
  ADD COLUMN IF NOT EXISTS embedding_dim INT NOT NULL DEFAULT 1024;

