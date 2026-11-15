import { describe, expect, it } from 'vitest';
import {
  COGNITIVE_CARD_BANK_99,
  COGNITIVE_CARD_BANK_100,
  sampleTenCognitiveCards,
} from '../PhysicalCardGameData';
import {
  MINIGAME_VOICE_LANG_OPTIONS,
  buildMinigameTtsUrl,
  formatMinigameAnswerSpeech,
  formatMinigameSummarySpeech,
  normalizeClinicalLangToMinigameVoice,
  pickCardTextByVoiceLang,
} from '../PhysicalCardGame';

describe('PhysicalCardGame Cognitive Card Bank (100 Question-Matched SVG Cards)', () => {
  it('contains 100 unique cards with distinct art keys and valid SVG illustrations', () => {
    expect(COGNITIVE_CARD_BANK_100).toHaveLength(100);
    expect(COGNITIVE_CARD_BANK_99).toBe(COGNITIVE_CARD_BANK_100);

    const ids = new Set<string>();
    const artKeys = new Set<string>();
    const imageUris = new Set<string>();

    for (const [index, card] of COGNITIVE_CARD_BANK_100.entries()) {
      const expectedBadge = `#${String(index + 1).padStart(3, '0')}`;
      ids.add(card.id);
      artKeys.add(card.artKey);
      imageUris.add(card.imageUrl);

      expect(card.imageUrl.startsWith('data:image/svg+xml;utf8,')).toBe(true);
      const decodedSvg = decodeURIComponent(card.imageUrl.replace('data:image/svg+xml;utf8,', ''));
      expect(decodedSvg).toContain('<svg');
      expect(decodedSvg).toContain('</svg>');
      expect(decodedSvg).toContain(expectedBadge);

      expect(card.subject.th.length).toBeGreaterThan(0);
      expect(card.subject.en.length).toBeGreaterThan(0);
      expect(card.subject.cn.length).toBeGreaterThan(0);
      expect(card.question.th.length).toBeGreaterThan(0);
      expect(card.question.en.length).toBeGreaterThan(0);
      expect(card.question.cn.length).toBeGreaterThan(0);
    }

    expect(ids.size).toBe(100);
    expect(artKeys.size).toBe(100);
    expect(imageUris.size).toBe(100);
  });

  it('maintains a balanced 50 yes / 50 no answer distribution', () => {
    const yesCount = COGNITIVE_CARD_BANK_100.filter((c) => c.correctAnswer === 'yes').length;
    const noCount = COGNITIVE_CARD_BANK_100.filter((c) => c.correctAnswer === 'no').length;
    expect(yesCount).toBe(50);
    expect(noCount).toBe(50);
  });

  it('samples 10 unique cards from the full bank and respects category filters', () => {
    const sample = sampleTenCognitiveCards();
    expect(sample).toHaveLength(10);
    expect(new Set(sample.map((c) => c.id)).size).toBe(10);

    const positionSample = sampleTenCognitiveCards(['position']);
    expect(positionSample).toHaveLength(8);
    expect(positionSample.every((c) => c.category === 'position')).toBe(true);
  });
});

describe('PhysicalCardGame Trilingual AI Voiceover Helpers', () => {
  it('exposes the 3 voiceover language options in Thai, English, and Chinese order', () => {
    expect(MINIGAME_VOICE_LANG_OPTIONS.map((opt) => [opt.code, opt.shortLabel])).toEqual([
      ['th', '1. ไทย'],
      ['en', '2. อังกฤษ (EN)'],
      ['zh', '3. จีน (中文)'],
    ]);
    expect(normalizeClinicalLangToMinigameVoice('cn')).toBe('zh');
    expect(normalizeClinicalLangToMinigameVoice('en')).toBe('en');
    expect(normalizeClinicalLangToMinigameVoice('th')).toBe('th');
  });

  it('picks trilingual card questions, answer explanations, and summary speech per selected language', () => {
    const card = COGNITIVE_CARD_BANK_100[2];
    expect(pickCardTextByVoiceLang(card.question, 'th')).toBe(card.question.th);
    expect(pickCardTextByVoiceLang(card.question, 'en')).toBe(card.question.en);
    expect(pickCardTextByVoiceLang(card.question, 'zh')).toBe(card.question.cn);

    expect(formatMinigameAnswerSpeech(card, true, 'th')).toContain(card.explanation.th);
    expect(formatMinigameAnswerSpeech(card, false, 'en')).toContain('Not quite.');
    expect(formatMinigameAnswerSpeech(card, false, 'en')).toContain(card.explanation.en);
    expect(formatMinigameAnswerSpeech(card, true, 'zh')).toContain('回答正确！');
    expect(formatMinigameAnswerSpeech(card, true, 'zh')).toContain(card.explanation.cn);

    const summary = {
      correct: 9,
      total: 10,
      category: 'mixed-99',
      durationSec: 42,
      accuracyPct: 90,
    };
    expect(formatMinigameSummarySpeech(summary, 'th')).toContain('9 เต็ม 10');
    expect(formatMinigameSummarySpeech(summary, 'en')).toContain('9 out of 10');
    expect(formatMinigameSummarySpeech(summary, 'zh')).toContain('答对了10题中的9题');
  });

  it('builds valid /api/tts proxy URLs and clamps overly long prompts cleanly', () => {
    const urlZh = buildMinigameTtsUrl('这是一张绿色西兰花的图片吗？', 'zh');
    expect(urlZh.startsWith('/api/tts?')).toBe(true);
    const parsed = new URL(`https://example.com${urlZh}`);
    expect(parsed.searchParams.get('lang')).toBe('zh');
    expect(parsed.searchParams.get('text')).toBe('这是一张绿色西兰花的图片吗？');

    const longText = 'คำถามทดสอบความจำกายภาพ '.repeat(20);
    const clampedUrl = buildMinigameTtsUrl(longText, 'th');
    const parsedClamped = new URL(`https://example.com${clampedUrl}`);
    expect((parsedClamped.searchParams.get('text') ?? '').length).toBeLessThanOrEqual(190);
  });
});

