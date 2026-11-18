export type SupportedTtsLang = 'th' | 'en' | 'zh-CN';

/** Normalize frontend/BCP-47 language codes into canonical Google/Cloud TTS language tags. */
export function normalizeTtsLang(rawLang?: string | null): SupportedTtsLang {
  const lower = (rawLang ?? 'th').trim().toLowerCase();
  if (lower.startsWith('th')) return 'th';
  if (lower.startsWith('zh') || lower === 'cn' || lower.startsWith('cmn')) return 'zh-CN';
  return 'en';
}

/** Strip emojis/dingbats, collapse whitespace, and clamp so TTS requests stay clean & within ~200 char limits. */
export function clampTtsText(text: string, maxLen = 190): string {
  const collapsed = text
    .replace(/[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!collapsed) return '';
  if (collapsed.length <= maxLen) return collapsed;

  const sliced = collapsed.slice(0, maxLen);
  const lastSpace = sliced.lastIndexOf(' ');
  if (lastSpace >= Math.floor(maxLen * 0.5)) {
    return sliced.slice(0, lastSpace).trim();
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
 * Fetch synthesized speech MP3 audio server-side (no Referer header) so browsers without
 * local OS Thai/Chinese voices can play natural voice prompts on every platform.
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
  const urls = buildUpstreamTtsUrls(lang, clean);

  for (const upstreamUrl of urls) {
    try {
      const res = await fetchImpl(upstreamUrl, {
        method: 'GET',
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
      return {
        audio,
        contentType: contentType.startsWith('audio/') ? contentType : 'audio/mpeg',
        provider: upstreamUrl.includes('client=tw-ob') ? 'google-tw-ob' : 'google-gtx',
      };
    } catch {
      // Try next upstream provider
    }
  }
  return null;
}

