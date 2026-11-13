import { createCloudDelegatedRouter } from '@drtechcares/api-client';

let cachedRouterKey = '';
let cachedRouter = null;

function getOrCreateRouter(env) {
  const supabaseUrl = env.SUPABASE_URL || env.VITE_SUPABASE_URL || '';
  const supabaseAnonKey = env.SUPABASE_ANON_KEY || env.VITE_SUPABASE_ANON_KEY || '';
  const supabaseServiceKey = env.SUPABASE_SERVICE_ROLE_KEY || '';
  const key = `${supabaseUrl}|${supabaseAnonKey}|${supabaseServiceKey}`;

  if (!cachedRouter || cachedRouterKey !== key) {
    cachedRouter = createCloudDelegatedRouter({
      supabaseUrl,
      supabaseAnonKey,
      supabaseServiceKey,
    });
    cachedRouterKey = key;
  }
  return cachedRouter;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Delegate /api/* directly to cached cloud router (with Supabase PostgreSQL persistence)
    if (url.pathname.startsWith('/api/')) {
      const router = getOrCreateRouter(env);
      return router(request);
    }

    // Serve static assets from Cloudflare ASSETS binding
    if (env.ASSETS) {
      const assetRes = await env.ASSETS.fetch(request);
      if (assetRes.status !== 404) {
        return assetRes;
      }

      // SPA fallback for /, /kiosk, /login, /pt, /admin, /patient
      const spaRoutes = ['/', '/kiosk', '/login', '/pt', '/admin', '/patient'];
      if (spaRoutes.some((prefix) => url.pathname === prefix || url.pathname.startsWith(`${prefix}/`))) {
        const indexUrl = new URL('/index.html', url.origin);
        return env.ASSETS.fetch(new Request(indexUrl.toString(), request));
      }
      return assetRes;
    }

    return new Response('DRTECHCARES Worker Online', { status: 200 });
  },
};
