import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  DrTechCareMascot,
  MASCOT_ENCOURAGING_DURATION_MS,
  MASCOT_GREETING_DURATION_MS,
  getMascotMouthPath,
  resolveMascotPose,
  type DrTechCareMascotPose,
} from '../components/DrTechCareMascot';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('Animated Dr. TechCare Chibi Mascot & Voice Coach Integration', () => {
  it('resolves pose precedence (encouraging > speaking > greeting > explaining > idle) and distinct phoneme paths', () => {
    expect(MASCOT_ENCOURAGING_DURATION_MS).toBe(1600);
    expect(MASCOT_GREETING_DURATION_MS).toBe(2600);

    expect(
      resolveMascotPose({
        isGreeting: true,
        isEncouraging: true,
        isSpeaking: true,
        isExplaining: true,
      })
    ).toBe('encouraging');

    expect(
      resolveMascotPose({
        isGreeting: true,
        isEncouraging: false,
        isSpeaking: true,
        isExplaining: true,
      })
    ).toBe('speaking');

    expect(
      resolveMascotPose({
        isGreeting: true,
        isEncouraging: false,
        isSpeaking: false,
        isExplaining: true,
      })
    ).toBe('greeting');

    expect(
      resolveMascotPose({
        isGreeting: false,
        isEncouraging: false,
        isSpeaking: false,
        isExplaining: true,
      })
    ).toBe('explaining');

    expect(
      resolveMascotPose({
        isGreeting: false,
        isEncouraging: false,
        isSpeaking: false,
        isExplaining: false,
      })
    ).toBe('idle');

    const speakingShapes = [0, 1, 2, 3].map((f) => getMascotMouthPath('speaking', true, f).d);
    expect(new Set(speakingShapes).size).toBe(4);
    expect(getMascotMouthPath('idle', false, 0).fill).toBe('none');
    expect(getMascotMouthPath('encouraging', false, 0).showTongue).toBe(true);
  });

  it('renders all 5 poses matching character design specs without emojis', () => {
    const poses: DrTechCareMascotPose[] = [
      'idle',
      'greeting',
      'explaining',
      'speaking',
      'encouraging',
    ];

    for (const pose of poses) {
      const html = renderToStaticMarkup(
        React.createElement(DrTechCareMascot, {
          pose,
          isSpeaking: pose === 'speaking',
        })
      );
      expect(html).toContain(`data-pose="${pose}"`);
      expect(html).toContain('viewBox="0 0 160 180"');
      expect(html).toContain('#f5de92');
      expect(html).toContain('#e9c46a');
      if (pose === 'encouraging') {
        expect(html).toContain('#facc15');
      } else {
        expect(html).toContain('#9b5de5');
      }
      expect(html).toContain('#06b6d4');
      expect(html).toContain('#2dd4bf');
      expect(html).toContain('#10b981');
      expect(html).toContain('#0284c7');
      expect(html).toContain('duration-300 ease-in-out');
      expect(EMOJI_RE.test(html)).toBe(false);
    }

    const greetingHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, { pose: 'greeting', isSpeaking: false, size: 'sm' })
    );
    expect(greetingHtml).toContain('w-12 h-14');
    expect(greetingHtml).toContain('animate-gentle-wave');
    expect(greetingHtml).toContain('สวัสดีครับ!');

    const explainingHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, { pose: 'explaining', isSpeaking: false, size: 'md' })
    );
    expect(explainingHtml).toContain('w-20 h-24');
    expect(explainingHtml).toContain('translateY(25px)');
    expect(explainingHtml).toContain('M33 51 L45 63 L68 38');

    const speakingHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, { pose: 'speaking', isSpeaking: true, size: 'lg' })
    );
    expect(speakingHtml).toContain('w-32 h-36');
    expect(speakingHtml).toContain('animate-audio-waves');
    expect(speakingHtml).toContain('data-testid="mascot-soundwaves"');
    expect(speakingHtml).toContain('ค่อยๆ');
    expect(speakingHtml).toContain('ยกแขนขึ้น');
    expect(speakingHtml).toContain('ช้าๆ นะครับ');

    const encouragingHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, { pose: 'encouraging', isSpeaking: false })
    );
    expect(encouragingHtml).toContain('#facc15');
    expect(encouragingHtml).toContain('animate-fist-pump');
    expect(encouragingHtml).toContain('data-testid="mascot-encouraging-particles"');
    expect(encouragingHtml).toContain('สู้ๆครับ!');

    const hiddenBubbleHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, {
        pose: 'encouraging',
        isSpeaking: false,
        showSpeechBubble: false,
      })
    );
    expect(hiddenBubbleHtml).not.toContain('data-testid="mascot-speech-bubble"');
  });

  it('wires DrTechCareMascot, CSS keyframes, VoiceCoachBanner speaker hook, and keeps 100dvh with zero emojis', () => {
    const css = fs.readFileSync(path.resolve(process.cwd(), 'src/index.css'), 'utf8');
    for (const token of [
      '@keyframes mascotBreath',
      '@keyframes mascotBlink',
      '@keyframes gentleWave',
      '@keyframes fistCelebration',
      '@keyframes audioWaveCascade',
      '.animate-gentle-wave',
      '.animate-fist-pump',
      '.animate-audio-waves',
    ]) {
      expect(css).toContain(token);
    }

    const kioskSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/SmartRehabKioskUI.tsx'),
      'utf8'
    );
    expect(kioskSrc).toContain('h-[100dvh] max-h-[100dvh] w-full overflow-hidden');
    expect(kioskSrc).toContain('w-14 h-16 shrink-0 mr-3 animate-view-enter');
    expect(kioskSrc).toContain('onSpeakingChange={setVoiceCoachSpeaking}');
    expect(kioskSrc).toContain('MASCOT_ENCOURAGING_DURATION_MS');

    const voiceSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/VoiceCoachBanner.tsx'),
      'utf8'
    );
    expect(voiceSrc).toContain('onSpeakingChange?: (speaking: boolean) => void;');

    const staffSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/ClinicalStaffPortalShell.tsx'),
      'utf8'
    );
    expect(staffSrc).toContain('staff-portal-mascot');

    const patientSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/PatientPersonalPortal.tsx'),
      'utf8'
    );
    expect(patientSrc).toContain('patient-portal-mascot');

    for (const relPath of [
      'src/index.css',
      'src/components/DrTechCareMascot.tsx',
      'src/components/VoiceCoachBanner.tsx',
      'src/components/SmartRehabKioskUI.tsx',
      'src/components/ClinicalStaffPortalShell.tsx',
      'src/PatientPersonalPortal.tsx',
      'src/__tests__/drTechCareMascot.test.ts',
    ]) {
      const content = fs.readFileSync(path.resolve(process.cwd(), relPath), 'utf8');
      expect(EMOJI_RE.test(content)).toBe(false);
    }
  });
});

