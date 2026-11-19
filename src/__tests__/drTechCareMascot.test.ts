import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import {
  getGlobalMascotSpeechState,
  inferMascotPoseFromSpeechText,
  notifyMascotSpeechEnd,
  notifyMascotSpeechStart,
  subscribeGlobalMascotSpeech,
} from '@drtechcares/kiosk-input';
import {
  DrTechCareMascot,
  MASCOT_ENCOURAGING_DURATION_MS,
  MASCOT_GREETING_DURATION_MS,
  MASCOT_POSTER_SRC,
  MASCOT_VIDEO_SEGMENTS,
  MASCOT_VIDEO_SRC,
  getMascotMouthPath,
  resolveMascotPose,
  type DrTechCareMascotPose,
} from '../components/DrTechCareMascot';
import {
  GlobalSpeechMascotDock,
  shouldRenderGlobalSpeechDock,
} from '../components/GlobalSpeechMascotDock';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('Dr. TechCare Video Mascot Segments, Site-Wide Speech Bus & Dock Integration', () => {
  it('provides trimmed non-bleeding video clip assets & segment windows for all 5 poses', () => {
    expect(MASCOT_VIDEO_SRC).toBe('/mascot/dr-techcare-mascot.mp4');
    expect(MASCOT_POSTER_SRC).toBe('/mascot/dr-techcare-poster.jpg');

    const mp4Path = path.resolve(process.cwd(), 'public/mascot/dr-techcare-mascot.mp4');
    const posterPath = path.resolve(process.cwd(), 'public/mascot/dr-techcare-poster.jpg');
    expect(fs.existsSync(mp4Path)).toBe(true);
    expect(fs.existsSync(posterPath)).toBe(true);
    expect(fs.statSync(mp4Path).size).toBeGreaterThan(100_000);
    expect(fs.statSync(posterPath).size).toBeGreaterThan(1_000);

    expect(MASCOT_VIDEO_SEGMENTS.greeting.startSec).toBeGreaterThanOrEqual(0);
    expect(MASCOT_VIDEO_SEGMENTS.greeting.endSec).toBeLessThan(2.0);
    expect(MASCOT_VIDEO_SEGMENTS.explaining.startSec).toBeGreaterThan(2.0);
    expect(MASCOT_VIDEO_SEGMENTS.explaining.endSec).toBeLessThan(4.0);
    expect(MASCOT_VIDEO_SEGMENTS.speaking.startSec).toBeGreaterThan(4.0);
    expect(MASCOT_VIDEO_SEGMENTS.speaking.endSec).toBeLessThan(6.45);
    expect(MASCOT_VIDEO_SEGMENTS.encouraging.startSec).toBeGreaterThan(6.46);
    expect(MASCOT_VIDEO_SEGMENTS.encouraging.endSec).toBeLessThanOrEqual(10.0);
    expect(MASCOT_VIDEO_SEGMENTS.idle.startSec).toBeGreaterThanOrEqual(0);
    expect(MASCOT_VIDEO_SEGMENTS.idle.endSec).toBeLessThan(MASCOT_VIDEO_SEGMENTS.speaking.endSec);

    const poses: DrTechCareMascotPose[] = [
      'idle',
      'greeting',
      'explaining',
      'speaking',
      'encouraging',
    ];

    for (const pose of poses) {
      const seg = MASCOT_VIDEO_SEGMENTS[pose];
      expect(seg.startSec).toBeLessThan(seg.endSec);
      const html = renderToStaticMarkup(
        React.createElement(DrTechCareMascot, {
          pose,
          isSpeaking: pose === 'speaking',
        })
      );
      expect(html).toContain('data-testid="mascot-video"');
      expect(html).toContain('src="/mascot/dr-techcare-mascot.mp4"');
      expect(html).toContain('poster="/mascot/dr-techcare-poster.jpg"');
      expect(html).toContain(`data-video-start="${seg.startSec}"`);
      expect(html).toContain(`data-video-end="${seg.endSec}"`);
      expect(html).toContain('muted=""');
      expect(html).toContain('playsInline=""');
      expect(EMOJI_RE.test(html)).toBe(false);
    }
  });

  it('resolves pose precedence, phoneme paths, and 2D vector SVG fallback cleanly', () => {
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

    const speakingShapes = [0, 1, 2, 3].map((f) => getMascotMouthPath('speaking', true, f).d);
    expect(new Set(speakingShapes).size).toBe(4);

    const svgFallbackHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, {
        pose: 'explaining',
        isSpeaking: false,
        useVideo: false,
      })
    );
    expect(svgFallbackHtml).toContain('viewBox="0 0 160 180"');
    expect(svgFallbackHtml).toContain('translateY(25px)');
    expect(svgFallbackHtml).not.toContain('data-testid="mascot-video"');
  });

  it('classifies Thai/EN/CN utterances into poses and coordinates site-wide speech bus & dock', () => {
    expect(inferMascotPoseFromSpeechText('สวัสดีครับ ยินดีต้อนรับสู่การฝึกกายภาพ')).toBe('greeting');
    expect(inferMascotPoseFromSpeechText('Please keep your back straight and hold for 3 seconds')).toBe(
      'explaining'
    );
    expect(inferMascotPoseFromSpeechText('เก่งมากครับ ได้ +1 แต้มแล้ว!')).toBe('encouraging');
    expect(inferMascotPoseFromSpeechText('ค่อยๆ ยกแขนขึ้น ช้าๆ นะครับ')).toBe('speaking');
    expect(inferMascotPoseFromSpeechText('请保持这个姿势 3 秒')).toBe('explaining');
    expect(inferMascotPoseFromSpeechText('太棒了！已完成全部动作')).toBe('encouraging');

    const seen: string[] = [];
    const unsubscribe = subscribeGlobalMascotSpeech((st) => {
      seen.push(`${st.source}:${st.pose}:${st.isSpeaking}`);
    });

    try {
      const started = notifyMascotSpeechStart({
        text: 'ยืนยันตัวตนสำเร็จ พร้อมเริ่มฝึกครับ',
        source: 'kiosk_login',
        pose: 'greeting',
      });
      expect(started.isSpeaking).toBe(true);
      expect(started.pose).toBe('greeting');
      expect(getGlobalMascotSpeechState().source).toBe('kiosk_login');

      // Ending a different inactive source must NOT interrupt kiosk_login
      notifyMascotSpeechEnd('card_game');
      expect(getGlobalMascotSpeechState().isSpeaking).toBe(true);
      expect(getGlobalMascotSpeechState().source).toBe('kiosk_login');

      // Ending the active source stops speaking cleanly
      const ended = notifyMascotSpeechEnd('kiosk_login');
      expect(ended.isSpeaking).toBe(false);
      expect(ended.pose).toBe('idle');
      expect(seen).toContain('kiosk_login:greeting:true');
      expect(seen).toContain('kiosk_login:idle:false');
    } finally {
      unsubscribe();
      notifyMascotSpeechEnd();
    }

    // GlobalSpeechMascotDock renders cleanly for non-voice_coach speech
    const cardGameState = {
      isSpeaking: true,
      text: 'คำถามข้อที่ 1 กรุณาเลือกคำตอบที่ถูกต้องครับ',
      pose: 'explaining' as const,
      source: 'card_game' as const,
      updatedAt: 1000,
    };
    expect(shouldRenderGlobalSpeechDock(cardGameState)).toBe(true);
    const dockHtml = renderToStaticMarkup(
      React.createElement(GlobalSpeechMascotDock, { initialState: cardGameState })
    );
    expect(dockHtml).toContain('data-testid="global-speech-mascot-dock"');
    expect(dockHtml).toContain('กำลังอธิบาย');
    expect(dockHtml).toContain('คำถามข้อที่ 1');
    expect(dockHtml).toContain('/mascot/dr-techcare-mascot.mp4');

    // GlobalSpeechMascotDock suppresses itself when SmartRehabKioskUI's VoiceCoachBanner speaks
    const voiceCoachState = {
      isSpeaking: true,
      text: 'ค่อยๆ ยกแขนขึ้น ช้าๆ นะครับ',
      pose: 'speaking' as const,
      source: 'voice_coach' as const,
      updatedAt: 2000,
    };
    expect(shouldRenderGlobalSpeechDock(voiceCoachState)).toBe(false);
    const suppressedHtml = renderToStaticMarkup(
      React.createElement(GlobalSpeechMascotDock, { initialState: voiceCoachState })
    );
    expect(suppressedHtml).toBe('');
  });

  it('auto-resets stuck speech state after watchdog timeout without manual end call', () => {
    vi.useFakeTimers();
    try {
      notifyMascotSpeechStart({
        text: 'ทดสอบนาฬิกาตัดจบอัตโนมัติ',
        source: 'other',
      });
      expect(getGlobalMascotSpeechState().isSpeaking).toBe(true);
      vi.advanceTimersByTime(15000);
      expect(getGlobalMascotSpeechState().isSpeaking).toBe(false);
    } finally {
      notifyMascotSpeechEnd();
      vi.useRealTimers();
    }
  });

  it('wires speech bus across all voice engines and enforces zero emojis across touched files', () => {
    const appSrc = fs.readFileSync(path.resolve(process.cwd(), 'src/App.tsx'), 'utf8');
    expect(appSrc).toContain("source: 'kiosk_overview'");

    const loginSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/KioskBiometricLoginView.tsx'),
      'utf8'
    );
    expect(loginSrc).toContain("source: 'kiosk_login'");

    const faceEnrollSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/FaceID120Scanner.tsx'),
      'utf8'
    );
    expect(faceEnrollSrc).toContain("source: 'face_enroll'");

    const cardGameSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'apps/kiosk/src/PhysicalCardGame.tsx'),
      'utf8'
    );
    expect(cardGameSrc).toContain("source: 'card_game'");

    const voiceBannerSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/VoiceCoachBanner.tsx'),
      'utf8'
    );
    expect(voiceBannerSrc).toContain("source: 'voice_coach'");

    const mainSrc = fs.readFileSync(path.resolve(process.cwd(), 'src/main.tsx'), 'utf8');
    expect(mainSrc).toContain('GlobalSpeechMascotDock');

    const kioskSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/SmartRehabKioskUI.tsx'),
      'utf8'
    );
    expect(kioskSrc).toContain('h-[100dvh] max-h-[100dvh] w-full overflow-hidden');
    expect(kioskSrc).toContain('w-14 h-16 shrink-0 mr-3 animate-view-enter');

    for (const relPath of [
      'packages/kiosk-input/src/mascotSpeechBus.ts',
      'packages/kiosk-input/src/voiceMute.ts',
      'packages/kiosk-input/src/index.ts',
      'src/components/DrTechCareMascot.tsx',
      'src/components/GlobalSpeechMascotDock.tsx',
      'src/components/VoiceCoachBanner.tsx',
      'src/components/SmartRehabKioskUI.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/components/FaceID120Scanner.tsx',
      'apps/kiosk/src/PhysicalCardGame.tsx',
      'src/App.tsx',
      'src/main.tsx',
      'src/__tests__/drTechCareMascot.test.ts',
    ]) {
      const content = fs.readFileSync(path.resolve(process.cwd(), relPath), 'utf8');
      expect(EMOJI_RE.test(content)).toBe(false);
    }
  });
});
