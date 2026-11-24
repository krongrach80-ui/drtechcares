import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  FloatingScorePopup,
  GameCelebrationBurst,
  GameComboBadge,
  GameRankBadge,
  GameStarRating,
  GameXpProgressBar,
  resolveClinicalRankTier,
} from '../components/GameJuiceEffects';
import { DrTechCareMascot } from '../components/DrTechCareMascot';
import { RomGauge } from '../components/RomGauge';
import {
  isGameSfxMuted,
  playGameSfx,
  resetGameSfxStateForTests,
} from '../lib/gameUiSfx';

const UNICODE_EMOJI_RE = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/u;

describe('Game-Like Clinical UX/UI Kit (Zero Emojis & 60FPS GPU)', () => {
  beforeEach(() => {
    resetGameSfxStateForTests();
  });

  it('safely no-ops Web Audio SFX in headless/SSR Node environments and honors explicit mute', () => {
    expect(isGameSfxMuted(true)).toBe(true);
    expect(playGameSfx('tap', { muted: true })).toBe(false);
    expect(playGameSfx('rep')).toBe(false);
    expect(playGameSfx('combo', { comboCount: 5 })).toBe(false);
    expect(playGameSfx('victory')).toBe(false);
    expect(playGameSfx('unlock')).toBe(false);
  });

  it('maps clinical percentages deterministically to S+/S/A/B/C mastery tiers', () => {
    expect(resolveClinicalRankTier(98).grade).toBe('S+');
    expect(resolveClinicalRankTier(88).grade).toBe('S');
    expect(resolveClinicalRankTier(74).grade).toBe('A');
    expect(resolveClinicalRankTier(55).grade).toBe('B');
    expect(resolveClinicalRankTier(20).grade).toBe('C');
  });

  it('renders pure inline-SVG juice components with zero Unicode emojis', () => {
    const html = renderToStaticMarkup(
      React.createElement(
        'div',
        null,
        React.createElement(GameCelebrationBurst, { triggerKey: 1, variant: 'victory' }),
        React.createElement(FloatingScorePopup, { triggerKey: 2, label: '+1 ครั้ง!', sublabel: '+10 XP', tone: 'gold' }),
        React.createElement(GameComboBadge, { count: 4, labelTh: 'คอมโบ' }),
        React.createElement(GameXpProgressBar, {
          value: 6,
          max: 10,
          label: 'ภารกิจประจำวัน',
          badgeText: '6/10 ท่า',
        }),
        React.createElement(GameRankBadge, { pct: 96 }),
        React.createElement(GameStarRating, { earned: 3, total: 3, size: 'lg' }),
      ),
    );

    expect(html).toContain('data-testid="game-celebration-burst"');
    expect(html).toContain('data-testid="game-floating-score"');
    expect(html).toContain('data-testid="game-combo-badge"');
    expect(html).toContain('คอมโบ x4');
    expect(html).toContain('data-testid="game-xp-progress-bar"');
    expect(html).toContain('RANK S+');
    expect(html).toContain('data-testid="game-star-rating"');
    expect(html).toContain('<svg');
    expect(UNICODE_EMOJI_RE.test(html)).toBe(false);
  });

  it('hides GameComboBadge when streak is below 2 and hides burst when triggerKey is 0', () => {
    const emptyCombo = renderToStaticMarkup(React.createElement(GameComboBadge, { count: 1 }));
    const emptyBurst = renderToStaticMarkup(React.createElement(GameCelebrationBurst, { triggerKey: 0 }));
    expect(emptyCombo).toBe('');
    expect(emptyBurst).toBe('');
  });

  it('renders upgraded DrTechCareMascot (celebrating + greeting) and RomGauge target badge cleanly', () => {
    const mascotHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, { pose: 'celebrating', isSpeaking: true, size: 'lg' }),
    );
    expect(mascotHtml).toContain('data-testid="drtechcare-mascot"');
    expect(mascotHtml).toContain('game-mascot-float');
    expect(mascotHtml).toContain('game-star-twinkle');
    expect(UNICODE_EMOJI_RE.test(mascotHtml)).toBe(false);

    const gaugeHtml = renderToStaticMarkup(
      React.createElement(RomGauge, {
        currentRomDeg: 105,
        targetRomDeg: 90,
        completionPct: 100,
        kioskState: 'HOLD_PHASE',
        highlightColor: 'GREEN',
        holdRemainingSec: 1.2,
        holdTargetSec: 3,
      }),
    );
    expect(gaugeHtml).toContain('105°');
    expect(gaugeHtml).toContain('ถึงเป้าแล้ว!');
    expect(gaugeHtml).toContain('game-progress-stripes');
    expect(UNICODE_EMOJI_RE.test(gaugeHtml)).toBe(false);
  });

  it('defines all 60FPS playful CSS classes in src/index.css with zero emojis', () => {
    const cssPath = path.resolve(process.cwd(), 'src/index.css');
    const css = fs.readFileSync(cssPath, 'utf8');

    expect(css).toContain('.btn-game-3d-emerald');
    expect(css).toContain('.btn-game-3d-blue');
    expect(css).toContain('.btn-game-3d-rose');
    expect(css).toContain('.btn-game-3d-white');
    expect(css).toContain('.game-card-surface');
    expect(css).toContain('.game-progress-stripes');
    expect(css).toContain('.game-badge-shimmer::after');
    expect(css).toContain('@keyframes gameParticleBurst');
    expect(css).toContain('@keyframes gameFloatScoreUp');
    expect(css).toContain('@keyframes gameMascotBob');
    expect(UNICODE_EMOJI_RE.test(css)).toBe(false);
  });

  it('contains zero Unicode emojis across newly created game-UI source modules', () => {
    const filesToCheck = [
      'src/lib/gameUiSfx.ts',
      'src/components/GameJuiceEffects.tsx',
      'src/components/DrTechCareMascot.tsx',
      'src/components/RomGauge.tsx',
    ];
    for (const rel of filesToCheck) {
      const content = fs.readFileSync(path.resolve(process.cwd(), rel), 'utf8');
      expect(UNICODE_EMOJI_RE.test(content), `${rel} must contain zero Unicode emojis`).toBe(false);
    }
  });
});

