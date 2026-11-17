import { createCloudDelegatedRouter } from '@drtechcares/api-client';

let cachedRouterKey = '';
let cachedRouter = null;

// Reuses only pure in-memory router closure; createCloudDelegatedRouter never shares pending
// I/O Promises (such as hydrationPromise) across separate Cloudflare Worker fetch invocations.
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
    const pathnameLower = url.pathname.toLowerCase();

    // Explicitly block probing of source, config, env, git, and sourcemap paths with 404
    const blockedPrefixes = [
      '/src',
      '/packages',
      '/apps',
      '/scripts',
      '/edge-vision',
      '/supabase',
      '/.git',
      '/.env',
      '/.agents',
    ];
    if (
      blockedPrefixes.some((p) => pathnameLower === p || pathnameLower.startsWith(`${p}/`)) ||
      pathnameLower.endsWith('.map') ||
      pathnameLower.endsWith('.ts') ||
      pathnameLower.endsWith('.tsx') ||
      pathnameLower.endsWith('.jsonc') ||
      pathnameLower.endsWith('.py')
    ) {
      return new Response('404 Not Found', {
        status: 404,
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'no-store',
          'X-Robots-Tag': 'noindex, nofollow',
        },
      });
    }

    // Delegate /api/* directly to cached cloud router (with Supabase PostgreSQL persistence)
    if (url.pathname.startsWith('/api/')) {
      try {
        const router = getOrCreateRouter(env);
        return await router(request);
      } catch (err) {
        return new Response(
          JSON.stringify({
            error: 'INTERNAL_WORKER_ERROR',
            message: err instanceof Error ? err.message : String(err),
          }),
          {
            status: 500,
            headers: {
              'Content-Type': 'application/json; charset=utf-8',
              'Access-Control-Allow-Origin': '*',
            },
          },
        );
      }
    }

    // Serve static assets from Cloudflare ASSETS binding
    if (env.ASSETS) {
      // Reassemble chunked >24MiB AI models (.onnx / .wasm) transparently
      if (
        (request.method === 'GET' || request.method === 'HEAD') &&
        (url.pathname.endsWith('.onnx') || url.pathname.endsWith('.wasm'))
      ) {
        const metaUrl = new URL(`${url.pathname}.status.json`, url.origin);
        const metaRes = await env.ASSETS.fetch(new Request(metaUrl.toString()));
        if (metaRes.ok) {
          const meta = await metaRes.json().catch(() => null);
          if (meta && meta.chunked === true && Array.isArray(meta.parts)) {
            const headers = {
              'Content-Type': meta.contentType || 'application/octet-stream',
              'Content-Length': String(meta.sizeBytes),
              'Cache-Control': 'public, max-age=31536000, immutable',
            };
            if (request.method === 'HEAD') {
              return new Response(null, { status: 200, headers });
            }
            const slashIdx = url.pathname.lastIndexOf('/');
            const baseDir = slashIdx >= 0 ? url.pathname.slice(0, slashIdx) : '';
            const origin = url.origin;
            const stream = new ReadableStream({
              async start(controller) {
                try {
                  for (const partName of meta.parts) {
                    const partUrl = new URL(`${baseDir}/${partName}`, origin);
                    const partRes = await env.ASSETS.fetch(new Request(partUrl.toString()));
                    if (!partRes.ok || !partRes.body) {
                      throw new Error(`Missing model chunk ${partName} (${partRes.status})`);
                    }
                    const reader = partRes.body.getReader();
                    while (true) {
                      const { done, value } = await reader.read();
                      if (done) break;
                      if (value) controller.enqueue(value);
                    }
                  }
                  controller.close();
                } catch (err) {
                  controller.error(err);
                }
              },
            });
            return new Response(stream, { status: 200, headers });
          }
        }
      }

      const assetRes = await env.ASSETS.fetch(request);
      if (assetRes.status !== 404) {
        const headers = new Headers(assetRes.headers);
        headers.set('X-Robots-Tag', 'noindex, nofollow');
        headers.set('X-Content-Type-Options', 'nosniff');
        return new Response(assetRes.body, {
          status: assetRes.status,
          statusText: assetRes.statusText,
          headers,
        });
      }

      // SPA fallback for /, /kiosk, /login, /pt, /admin, /patient
      const spaRoutes = ['/', '/kiosk', '/login', '/pt', '/admin', '/patient'];
      if (spaRoutes.some((prefix) => url.pathname === prefix || url.pathname.startsWith(`${prefix}/`))) {
        const indexUrl = new URL('/index.html', url.origin);
        const spaRes = await env.ASSETS.fetch(new Request(indexUrl.toString(), request));
        const headers = new Headers(spaRes.headers);
        headers.set('X-Robots-Tag', 'noindex, nofollow');
        headers.set('X-Content-Type-Options', 'nosniff');
        return new Response(spaRes.body, {
          status: spaRes.status,
          statusText: spaRes.statusText,
          headers,
        });
      }
      return new Response('404 Not Found', { status: 404 });
    }

    return new Response('DRTECHCARES Worker Online', { status: 200 });
  },
};
