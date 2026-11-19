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
  MASCOT_LIP_SYNC_SEQUENCE,
  MASCOT_LIP_SYNC_STEP_MS,
  MASCOT_SPRITE_SHEETS,
  getLipSyncFrameAtElapsed,
  getMascotMouthPath,
  resolveMascotPose,
  type DrTechCareMascotPose,
} from '../components/DrTechCareMascot';
import {
  GlobalSpeechMascotDock,
  shouldRenderGlobalSpeechDock,
} from '../components/GlobalSpeechMascotDock';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('Dr. TechCare 4-Frame Transparent Lip-Sync Sprite Mascot & Site-Wide Poses', () => {
  it('provides valid transparent RGBA PNG sprite sheets and 60FPS lip-sync frame selector for all 5 poses', () => {
    const spriteFiles = [
      'public/mascot/sprite-greeting.png',
      'public/mascot/sprite-explaining.png',
      'public/mascot/sprite-speaking.png',
      'public/mascot/sprite-encouraging.png',
    ];
    for (const rel of spriteFiles) {
      const fullPath = path.resolve(process.cwd(), rel);
      expect(fs.existsSync(fullPath)).toBe(true);
      const buf = fs.readFileSync(fullPath);
      expect(buf.length).toBeGreaterThan(50_000);
      // PNG magic header check
      expect(buf.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
    }

    // Lip-sync helper: silent snaps to 0 (closed smile), speaking cycles real open/round frames 0..3
    expect(MASCOT_LIP_SYNC_STEP_MS).toBe(90);
    expect(getLipSyncFrameAtElapsed(0, false)).toBe(0);
    expect(getLipSyncFrameAtElapsed(270, false)).toBe(0);

    const activeFrames = MASCOT_LIP_SYNC_SEQUENCE.map((_, idx) =>
      getLipSyncFrameAtElapsed(idx * MASCOT_LIP_SYNC_STEP_MS, true)
    );
    expect(new Set(activeFrames)).toEqual(new Set([0, 1, 2, 3]));

    const poses: DrTechCareMascotPose[] = [
      'greeting',
      'explaining',
      'speaking',
      'encouraging',
      'idle',
    ];

    for (const pose of poses) {
      const html = renderToStaticMarkup(
        React.createElement(DrTechCareMascot, {
          pose,
          isSpeaking: pose === 'speaking',
        })
      );
      expect(html).toContain('data-testid="mascot-sprite"');
      expect(html).toContain(`src="${MASCOT_SPRITE_SHEETS[pose]}"`);
      expect(html).toContain(`data-sprite-pose="${pose}"`);
      expect(html).toContain('data-mouth-frame="0"');
      expect(html).toContain('width:400%');
      expect(html).toContain('translate3d(-0%, 0, 0)');
      expect(EMOJI_RE.test(html)).toBe(false);
    }

    const speakingHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, {
        pose: 'speaking',
        isSpeaking: true,
      })
    );
    expect(speakingHtml).toContain('data-testid="mascot-soundwaves"');

    const encouragingHtml = renderToStaticMarkup(
      React.createElement(DrTechCareMascot, {
        pose: 'encouraging',
        isSpeaking: false,
      })
    );
    expect(encouragingHtml).toContain('data-testid="mascot-encouraging-particles"');
  });

  it('retains SVG fallback and pure pose/phoneme helper compatibility', () => {
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
    expect(svgFallbackHtml).not.toContain('data-testid="mascot-sprite"');
  });

  it('coordinates site-wide mascotSpeechBus and GlobalSpeechMascotDock suppression cleanly', () => {
    expect(inferMascotPoseFromSpeechText('สวัสดีครับ ยินดีต้อนรับ')).toBe('greeting');
    expect(inferMascotPoseFromSpeechText('กรุณายกแขนค้าง 3 วินาที')).toBe('explaining');
    expect(inferMascotPoseFromSpeechText('เก่งมากครับ ได้ +1 แต้มแล้ว!')).toBe('encouraging');
    expect(inferMascotPoseFromSpeechText('บันทึกผลวันนี้เรียบร้อยแล้วครับ')).toBe('encouraging');

    const events: string[] = [];
    const unsubscribe = subscribeGlobalMascotSpeech((st) => {
      events.push(`${st.source}:${st.pose}:${st.isSpeaking}`);
    });

    try {
      notifyMascotSpeechStart({
        text: 'สวัสดีครับ พร้อมเริ่มฝึกแล้วครับ',
        source: 'kiosk_login',
        pose: 'greeting',
      });
      expect(getGlobalMascotSpeechState().isSpeaking).toBe(true);
      expect(getGlobalMascotSpeechState().pose).toBe('greeting');

      notifyMascotSpeechEnd('card_game');
      expect(getGlobalMascotSpeechState().isSpeaking).toBe(true);

      notifyMascotSpeechEnd('kiosk_login');
      expect(getGlobalMascotSpeechState().isSpeaking).toBe(false);
      expect(events).toContain('kiosk_login:greeting:true');
      expect(events).toContain('kiosk_login:idle:false');
    } finally {
      unsubscribe();
      notifyMascotSpeechEnd();
    }

    const cardGameState = {
      isSpeaking: true,
      text: 'คำถามข้อที่ 1 กรุณาเลือกคำตอบครับ',
      pose: 'explaining' as const,
      source: 'card_game' as const,
      updatedAt: 1000,
    };
    expect(shouldRenderGlobalSpeechDock(cardGameState)).toBe(true);
    const dockHtml = renderToStaticMarkup(
      React.createElement(GlobalSpeechMascotDock, { initialState: cardGameState })
    );
    expect(dockHtml).toContain('data-testid="global-speech-mascot-dock"');
    expect(dockHtml).toContain('/mascot/sprite-explaining.png');
    expect(dockHtml).toContain('กำลังอธิบาย');

    expect(
      shouldRenderGlobalSpeechDock({
        isSpeaking: true,
        text: 'ค่อยๆ ยกแขนขึ้น',
        pose: 'explaining',
        source: 'voice_coach',
        updatedAt: 2000,
      })
    ).toBe(false);
    expect(
      shouldRenderGlobalSpeechDock({
        isSpeaking: true,
        text: 'หันหน้าซ้ายช้าๆ ครับ',
        pose: 'explaining',
        source: 'face_enroll',
        updatedAt: 3000,
      })
    ).toBe(false);
  });

  it('resets stuck speech state via watchdog timeout', () => {
    vi.useFakeTimers();
    try {
      notifyMascotSpeechStart({ text: 'ทดสอบนาฬิกาหยุดพูด', source: 'other' });
      expect(getGlobalMascotSpeechState().isSpeaking).toBe(true);
      vi.advanceTimersByTime(15000);
      expect(getGlobalMascotSpeechState().isSpeaking).toBe(false);
    } finally {
      notifyMascotSpeechEnd();
      vi.useRealTimers();
    }
  });

  it('wires exact poses per screen across the website and keeps zero Unicode emojis', () => {
    const loginView = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/KioskBiometricLoginView.tsx'),
      'utf8'
    );
    expect(loginView).toContain('kiosk-landing-greeting-mascot');
    expect(loginView).toContain('pose="greeting"');

    const rehabKiosk = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/SmartRehabKioskUI.tsx'),
      'utf8'
    );
    expect(rehabKiosk).toContain('h-[100dvh] max-h-[100dvh] w-full overflow-hidden');
    expect(rehabKiosk).toContain('w-14 h-16 shrink-0 mr-3 animate-view-enter');
    expect(rehabKiosk).toContain("? 'encouraging'");
    expect(rehabKiosk).toContain(": 'explaining'");

    const cardGame = fs.readFileSync(
      path.resolve(process.cwd(), 'apps/kiosk/src/PhysicalCardGame.tsx'),
      'utf8'
    );
    expect(cardGame).toContain(": 'explaining'");
    expect(cardGame).toContain("? 'encouraging'");

    const faceEnroll = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/FaceID120Scanner.tsx'),
      'utf8'
    );
    expect(faceEnroll).toContain('face-id-scanner-mascot');
    expect(faceEnroll).toContain("pose={phase === 'completed' ? 'speaking' : 'explaining'}");

    const enrollModal = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/StaffAssistedEnrollmentModal.tsx'),
      'utf8'
    );
    expect(enrollModal).toContain('patient-enroll-success-mascot');
    expect(enrollModal).toContain('pose="speaking"');

    const staffLogin = fs.readFileSync(
      path.resolve(process.cwd(), 'src/StaffLoginModalOrPage.tsx'),
      'utf8'
    );
    expect(staffLogin).toContain('staff-login-mascot');
    expect(staffLogin).toContain('pose="speaking"');

    const staffPortal = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/ClinicalStaffPortalShell.tsx'),
      'utf8'
    );
    expect(staffPortal).toContain('staff-portal-mascot');
    expect(staffPortal).toContain(": portalMascotTalking\n        ? 'speaking'");

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
      'src/components/StaffAssistedEnrollmentModal.tsx',
      'src/StaffLoginModalOrPage.tsx',
      'src/components/ClinicalStaffPortalShell.tsx',
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
