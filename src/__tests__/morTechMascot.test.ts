import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  MOR_TECH_MASCOT_MESSAGES,
  MOR_TECH_MASCOT_WIDTH_PX,
  MorTechMascot,
  computeMascotPatrolBounds,
  computeSpeechBubbleLayout,
  getMorTechMascotMessages,
  type MorTechMascotMode,
} from '../components/MorTechMascot';
import { DrTechCareMascot } from '../components/DrTechCareMascot';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

const ALL_MODES: readonly MorTechMascotMode[] = [
  'standby',
  'face-login',
  'rehab',
  'patient-portal',
];

describe('MorTechMascot (Nong MorTech Sea Otter Companion)', () => {
  it('renders custom SVG sea otter with context-aware elderly-friendly speech bubble across all 4 modes', () => {
    for (const mode of ALL_MODES) {
      const html = renderToStaticMarkup(React.createElement(MorTechMascot, { mode }));
      expect(html).toContain('data-testid="mortech-mascot-companion"');
      expect(html).toContain(`data-mode="${mode}"`);
      expect(html).toContain('data-testid="mortech-otter-svg"');
      expect(html).toContain('data-testid="mortech-mascot-speech-bubble"');
      expect(html).toContain('data-testid="mortech-mascot-close-bubble"');
      expect(html).toContain('น้องหมอเทค');
      expect(html).toContain(MOR_TECH_MASCOT_MESSAGES[mode][0].text);
      expect(html).toContain('fixed bottom-3');
      expect(html).toContain('z-40');
      expect(html).toContain('pointer-events-auto');
      expect(html).toContain('text-lg sm:text-xl');
      expect(html).toContain('border-emerald-400');
      expect(EMOJI_RE.test(html)).toBe(false);
    }
  });

  it('supports minimized speech bubble state, custom message override, hidden mode, and EN/CN i18n without emojis', () => {
    const minimizedHtml = renderToStaticMarkup(
      React.createElement(MorTechMascot, { mode: 'rehab', defaultSpeechVisible: false }),
    );
    expect(minimizedHtml).not.toContain('data-testid="mortech-mascot-speech-bubble"');
    expect(minimizedHtml).toContain('data-testid="mortech-mascot-minimized-pill"');

    const customHtml = renderToStaticMarkup(
      React.createElement(MorTechMascot, {
        mode: 'rehab',
        customMessage: 'ยกแขนขวาช้าๆ ตามหุ่นสาธิตนะครับ',
      }),
    );
    expect(customHtml).toContain('ยกแขนขวาช้าๆ ตามหุ่นสาธิตนะครับ');

    const hiddenHtml = renderToStaticMarkup(
      React.createElement(MorTechMascot, { mode: 'face-login', hidden: true }),
    );
    expect(hiddenHtml).toBe('');

    for (const lang of ['th', 'en', 'cn'] as const) {
      for (const mode of ALL_MODES) {
        const list = getMorTechMascotMessages(mode, lang);
        expect(list.length).toBeGreaterThanOrEqual(2);
        for (const item of list) {
          expect(item.text.trim().length).toBeGreaterThan(8);
          expect(EMOJI_RE.test(item.text)).toBe(false);
        }
      }
    }
  });

  it('keeps patrol bounds and speech bubble strictly inside every viewport width with zero horizontal overflow', () => {
    const widths = [360, 480, 540, 768, 1024, 1440, 1920];
    for (const vw of widths) {
      for (const mode of ALL_MODES) {
        const bounds = computeMascotPatrolBounds(vw, mode, MOR_TECH_MASCOT_WIDTH_PX);
        expect(bounds.minX).toBeGreaterThanOrEqual(10);
        expect(bounds.maxX).toBeGreaterThanOrEqual(bounds.minX);
        expect(bounds.maxX + MOR_TECH_MASCOT_WIDTH_PX).toBeLessThanOrEqual(vw);

        for (const sampleX of [bounds.minX, bounds.midX, bounds.maxX]) {
          const layout = computeSpeechBubbleLayout(sampleX, vw, MOR_TECH_MASCOT_WIDTH_PX);
          const bubbleLeft = sampleX + layout.relativeLeftPx;
          const bubbleRight = bubbleLeft + layout.bubbleWidthPx;
          expect(bubbleLeft).toBeGreaterThanOrEqual(12);
          expect(bubbleRight).toBeLessThanOrEqual(vw - 12);
          expect(layout.tailLeftPx).toBeGreaterThanOrEqual(22);
          expect(layout.tailLeftPx).toBeLessThanOrEqual(layout.bubbleWidthPx - 22);
        }
      }
    }

    // Rehab mode reserves extra right-edge safety clearance for emergency & settings buttons
    const standbyBounds = computeMascotPatrolBounds(1024, 'standby');
    const rehabBounds = computeMascotPatrolBounds(1024, 'rehab');
    expect(rehabBounds.maxX).toBeLessThan(standbyBounds.maxX);
  });

  it('contains zero Unicode emojis in source and is integrated across all 4 screen modes', () => {
    const root = process.cwd();
    const mascotSrc = fs.readFileSync(path.join(root, 'src/components/MorTechMascot.tsx'), 'utf8');
    expect(EMOJI_RE.test(mascotSrc)).toBe(false);
    expect(mascotSrc).toContain('scaleX');
    expect(mascotSrc).toContain('fixed bottom-3');
    expect(mascotSrc).toContain('pointer-events-auto');
    // Must stay animated even on Windows kiosks with OS prefers-reduced-motion enabled
    expect(mascotSrc).toContain('reducedMotion="never"');
    expect(mascotSrc).not.toContain('useReducedMotion');

    const standbySrc = fs.readFileSync(path.join(root, 'src/components/KioskStandbyScreen.tsx'), 'utf8');
    const faceLoginSrc = fs.readFileSync(path.join(root, 'src/components/KioskBiometricLoginView.tsx'), 'utf8');
    const rehabSrc = fs.readFileSync(path.join(root, 'src/components/SmartRehabKioskUI.tsx'), 'utf8');
    const portalSrc = fs.readFileSync(path.join(root, 'src/PatientPersonalPortal.tsx'), 'utf8');

    expect(standbySrc).toContain('<MorTechMascot mode="standby"');
    expect(faceLoginSrc).toContain('mode="face-login"');
    expect(rehabSrc).toContain('mode="rehab"');
    expect(portalSrc).toContain('<MorTechMascot mode="patient-portal"');
  });

  it('unifies DrTechCareMascot into the same Nong MorTech sea otter character across all poses and burrow pop-up with zero emojis', () => {
    for (const pose of ['idle', 'explaining', 'celebrating', 'encouraging', 'speaking', 'greeting'] as const) {
      const html = renderToStaticMarkup(
        React.createElement(DrTechCareMascot, { pose, isSpeaking: pose === 'speaking', size: 'md' }),
      );
      expect(html).toContain('data-testid="drtechcare-mascot"');
      expect(html).toContain('data-character="mortech-sea-otter"');
      expect(html).toContain(`data-pose="${pose}"`);
      expect(html).toContain('น้องหมอเทค');
      expect(EMOJI_RE.test(html)).toBe(false);
    }

    const burrowHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, {
        pose: 'greeting',
        popFromBurrow: true,
        isSpeaking: true,
        size: 'md',
      }),
    );
    expect(burrowHtml).toContain('data-testid="drtechcare-mascot"');
    expect(burrowHtml).toContain('data-burrow="true"');
    expect(burrowHtml).toContain('data-testid="mascot-burrow-popup"');
    expect(burrowHtml).toContain('data-testid="mascot-burrow-hole"');
    expect(burrowHtml).toContain('data-testid="mascot-burrow-otter-pop"');
    expect(burrowHtml).toContain('data-testid="mascot-burrow-waving-arm"');
    expect(burrowHtml).toContain('<clipPath');
    expect(EMOJI_RE.test(burrowHtml)).toBe(false);

    const loginSrc = fs.readFileSync(
      path.join(process.cwd(), 'src/components/KioskBiometricLoginView.tsx'),
      'utf8',
    );
    expect(loginSrc).toContain('data-testid="verified-centered-mascot-stage"');
    expect(loginSrc).toContain('popFromBurrow');
  });
});

