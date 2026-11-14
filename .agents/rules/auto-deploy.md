# Automatic Deployment Rule (GitHub + Cloudflare Workers)

After completing and verifying every code modification in this workspace:

1. **Build & Verify**:
   - Run `npm run typecheck; if ($LASTEXITCODE -eq 0) { npm test }; if ($LASTEXITCODE -eq 0) { npm run build }` and confirm zero errors before deploying.
2. **Deploy to Cloudflare Workers**:
   - Run `npx wrangler deploy` so the live site at `https://drtechcares-platform.krongrach80.workers.dev` stays up to date automatically.
3. **Commit & Push to GitHub (`origin/main`)**:
   - Stage updated files (NEVER stage `wrangler.jsonc`, `.env`, `.env.*`, or `edge-vision/.venv/` which contain secrets or virtualenvs):
     `git add .gitignore package.json .agents edge-vision src apps packages scripts supabase` (or specific modified files, verifying `wrangler.jsonc` is unstaged via `git reset HEAD wrangler.jsonc`).
   - Commit with a descriptive message and run `git push origin main`.
