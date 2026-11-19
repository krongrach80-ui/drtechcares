export type SupportedTtsLang = 'th' | 'en' | 'zh-CN';

const TTS_UPSTREAM_TIMEOUT_MS = 2800;
const MAX_TTS_CACHE_ENTRIES = 96;
const ttsMemoryCache = new Map<string, SynthesizedTtsAudio>();

/** Normalize frontend/BCP-47 language codes into canonical Google/Cloud TTS language tags. */
export function normalizeTtsLang(rawLang?: string | null): SupportedTtsLang {
  const lower = (rawLang ?? 'th').trim().toLowerCase();
  if (lower.startsWith('th')) return 'th';
  if (lower.startsWith('zh') || lower === 'cn' || lower.startsWith('cmn')) return 'zh-CN';
  return 'en';
}

/** Strip emojis/dingbats, collapse whitespace, and clamp at word/clause boundaries within ~200 char limits. */
export function clampTtsText(text: string, maxLen = 190): string {
  const collapsed = text
    .replace(/[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!collapsed) return '';
  if (collapsed.length <= maxLen) return collapsed;

  const sliced = collapsed.slice(0, maxLen);
  const lastSpace = sliced.lastIndexOf(' ');
  if (lastSpace >= Math.floor(maxLen * 0.65)) {
    return sliced.slice(0, lastSpace).trim();
  }

  // Fallback for spaceless Thai/CJK scripts: use Intl.Segmenter word boundaries when available
  try {
    const IntlObj = globalThis.Intl as unknown as {
      Segmenter?: new (
        locales?: string,
        options?: { granularity: 'word' | 'grapheme' | 'sentence' },
      ) => { segment(input: string): Iterable<{ segment: string; index: number }> };
    };
    if (typeof IntlObj?.Segmenter === 'function') {
      const segmenter = new IntlObj.Segmenter('th', { granularity: 'word' });
      let endOffset = 0;
      for (const seg of segmenter.segment(collapsed)) {
        const nextEnd = seg.index + seg.segment.length;
        if (nextEnd <= maxLen) {
          endOffset = nextEnd;
        } else {
          break;
        }
      }
      if (endOffset >= Math.floor(maxLen * 0.5)) {
        return collapsed.slice(0, endOffset).trim();
      }
    }
  } catch {
    // Fall back to hard slice below
  }

  return sliced.trim();
}

/** Build ordered upstream TTS synthesis URLs (fetched server-side without browser Referer). */
export function buildUpstreamTtsUrls(lang: SupportedTtsLang, text: string): string[] {
  const tl = encodeURIComponent(lang);
  const q = encodeURIComponent(text);
  return [
    `https://translate.googleapis.com/translate_tts?ie=UTF-8&client=tw-ob&tl=${tl}&q=${q}`,
    `https://translate.google.com/translate_tts?ie=UTF-8&client=gtx&tl=${tl}&q=${q}`,
  ];
}

export interface SynthesizedTtsAudio {
  audio: ArrayBuffer;
  contentType: string;
  provider: string;
}

/**
 * Fetch synthesized speech MP3 audio server-side (no Referer header) with bounded timeout
 * and LRU caching so repeated prompts play quickly without stalling browsers.
 */
export async function synthesizeTtsAudio(
  rawLang: string | undefined | null,
  rawText: string,
  fetchImpl: typeof fetch = globalThis.fetch,
): Promise<SynthesizedTtsAudio | null> {
  const clean = clampTtsText(rawText);
  if (!clean || typeof fetchImpl !== 'function') {
    return null;
  }
  const lang = normalizeTtsLang(rawLang);
  const cacheKey = `${lang}::${clean}`;
  if (fetchImpl === globalThis.fetch) {
    const cached = ttsMemoryCache.get(cacheKey);
    if (cached) {
      // Touch for LRU ordering and return a fresh buffer slice copy
      ttsMemoryCache.delete(cacheKey);
      ttsMemoryCache.set(cacheKey, cached);
      return {
        audio: cached.audio.slice(0),
        contentType: cached.contentType,
        provider: `${cached.provider}-cached`,
      };
    }
  }

  const urls = buildUpstreamTtsUrls(lang, clean);

  for (const upstreamUrl of urls) {
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer =
      controller && typeof setTimeout === 'function'
        ? setTimeout(() => controller.abort(), TTS_UPSTREAM_TIMEOUT_MS)
        : null;
    try {
      const res = await fetchImpl(upstreamUrl, {
        method: 'GET',
        ...(controller ? { signal: controller.signal } : {}),
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
          Accept: 'audio/mpeg, audio/*;q=0.9',
        },
      });
      if (!res.ok) continue;
      const contentType = (res.headers.get('content-type') || 'audio/mpeg').toLowerCase();
      if (contentType.includes('text/html')) continue;
      const audio = await res.arrayBuffer();
      if (audio.byteLength <= 128) continue;
      const entry: SynthesizedTtsAudio = {
        audio,
        contentType: contentType.startsWith('audio/') ? contentType : 'audio/mpeg',
        provider: upstreamUrl.includes('client=tw-ob') ? 'google-tw-ob' : 'google-gtx',
      };
      if (fetchImpl === globalThis.fetch) {
        if (ttsMemoryCache.size >= MAX_TTS_CACHE_ENTRIES) {
          const oldestKey = ttsMemoryCache.keys().next().value;
          if (oldestKey) ttsMemoryCache.delete(oldestKey);
        }
        ttsMemoryCache.set(cacheKey, {
          audio: audio.slice(0),
          contentType: entry.contentType,
          provider: entry.provider,
        });
      }
      return entry;
    } catch {
      // Try next upstream provider
    } finally {
      if (timer !== null && typeof clearTimeout === 'function') {
        clearTimeout(timer);
      }
    }
  }
  return null;
}

