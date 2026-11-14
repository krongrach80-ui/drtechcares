import { describe, expect, it } from 'vitest';
import {
  COGNITIVE_CARD_BANK_99,
  COGNITIVE_CARD_BANK_100,
  sampleTenCognitiveCards,
} from '../PhysicalCardGameData';

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

