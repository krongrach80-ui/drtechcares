-- ============================================================================
-- Migration 011: Cross-Device Staff Profile Avatar & Bio Synchronization
-- Adds avatar_url, bio, and updated_at columns to public.staff_users so staff
-- profile photos (Supabase Storage public URLs) and bios sync across devices.
-- ============================================================================

ALTER TABLE IF EXISTS public.staff_users
  ADD COLUMN IF NOT EXISTS avatar_url text,
  ADD COLUMN IF NOT EXISTS bio text,
  ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

CREATE INDEX IF NOT EXISTS idx_staff_users_username_lower
  ON public.staff_users (lower(username));

