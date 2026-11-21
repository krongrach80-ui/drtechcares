import fs from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  MAX_ACTIVE_RIPPLES,
  RIPPLE_LAYER_ID,
  findTactileTarget,
  installGlobalTactileMotion,
  resolveRippleTint,
  spawnTactileRipple,
} from '../lib/globalTactileMotion';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

interface MockElement {
  id: string;
  tagName: string;
  className: string;
  disabled?: boolean;
  dataset: Record<string, string | undefined>;
  attrs: Record<string, string>;
  style: Record<string, string>;
  children: MockElement[];
  parentElement: MockElement | null;
  closestMap: Record<string, MockElement | null>;
  rect: { left: number; top: number; width: number; height: number };
  listeners: Record<string, Array<(evt: unknown) => void>>;
  setAttribute: (k: string, v: string) => void;
  getAttribute: (k: string) => string | null;
  hasAttribute: (k: string) => boolean;
  closest: (sel: string) => MockElement | null;
  getBoundingClientRect: () => { left: number; top: number; width: number; height: number };
  appendChild: (child: MockElement) => MockElement;
  remove: () => void;
  addEventListener: (type: string, fn: (evt: unknown) => void) => void;
  dispatchEvent: (type: string, evt?: unknown) => void;
  readonly firstElementChild: MockElement | null;
}

function createMockElement(
  tagName = 'DIV',
  opts: {
    className?: string;
    disabled?: boolean;
    attrs?: Record<string, string>;
    rect?: { left: number; top: number; width: number; height: number };
    closestSelf?: string[];
    noRippleAncestor?: boolean;
  } = {},
): MockElement {
  const el: MockElement = {
    id: '',
    tagName: tagName.toUpperCase(),
    className: opts.className ?? '',
    disabled: opts.disabled ?? false,
    dataset: {},
    attrs: { ...(opts.attrs ?? {}) },
    style: {},
    children: [],
    parentElement: null,
    closestMap: {},
    rect: opts.rect ?? { left: 100, top: 200, width: 160, height: 48 },
    listeners: {},
    setAttribute(k, v) {
      this.attrs[k] = v;
    },
    getAttribute(k) {
      return k in this.attrs ? this.attrs[k] : null;
    },
    hasAttribute(k) {
      return k in this.attrs;
    },
    closest(sel) {
      if (sel === '[data-no-ripple="true"]') {
        return opts.noRippleAncestor || this.attrs['data-no-ripple'] === 'true' ? this : null;
      }
      if (this.closestMap[sel] !== undefined) return this.closestMap[sel];
      if (opts.closestSelf?.some((token) => sel.includes(token))) {
        return this;
      }
      if (this.tagName === 'BUTTON' && sel.includes('button') && !this.disabled) {
        return this;
      }
      return null;
    },
    getBoundingClientRect() {
      return this.rect;
    },
    appendChild(child) {
      child.parentElement = this;
      this.children.push(child);
      return child;
    },
    remove() {
      if (!this.parentElement) return;
      const idx = this.parentElement.children.indexOf(this);
      if (idx >= 0) this.parentElement.children.splice(idx, 1);
      this.parentElement = null;
    },
    addEventListener(type, fn) {
      this.listeners[type] = this.listeners[type] ?? [];
      this.listeners[type].push(fn);
    },
    dispatchEvent(type, evt = {}) {
      for (const fn of this.listeners[type] ?? []) {
        fn(evt);
      }
    },
    get firstElementChild() {
      return this.children[0] ?? null;
    },
  };
  return el;
}

function createMockDocument(computedStyles?: Map<MockElement, { borderRadius?: string; backgroundColor?: string; backgroundImage?: string }>) {
  const body = createMockElement('BODY');
  const docListeners: Record<string, Array<(evt: unknown) => void>> = {};

  const doc = {
    body,
    defaultView: {
      getComputedStyle: (el: MockElement) => {
        const custom = computedStyles?.get(el);
        return {
          borderRadius: custom?.borderRadius ?? '16px',
          backgroundColor: custom?.backgroundColor ?? 'rgba(0, 0, 0, 0)',
          backgroundImage: custom?.backgroundImage ?? 'none',
        };
      },
    },
    createElement: (tag: string) => createMockElement(tag),
    getElementById: (id: string): MockElement | null => {
      const stack = [...body.children];
      while (stack.length > 0) {
        const next = stack.shift()!;
        if (next.id === id) return next;
        stack.push(...next.children);
      }
      return null;
    },
    addEventListener: (type: string, fn: (evt: unknown) => void) => {
      docListeners[type] = docListeners[type] ?? [];
      docListeners[type].push(fn);
    },
    removeEventListener: (type: string, fn: (evt: unknown) => void) => {
      docListeners[type] = (docListeners[type] ?? []).filter((f) => f !== fn);
    },
    dispatchDocEvent: (type: string, evt: Record<string, unknown>) => {
      for (const fn of [...(docListeners[type] ?? [])]) {
        fn(evt);
      }
    },
    getListenerCount: (type: string) => (docListeners[type] ?? []).length,
  };

  return doc;
}

describe('Global 60FPS Tactile Ripple, Spring Buttons & View Transitions', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('defines GPU spring button rules, click-ripple layer, panel switch, stagger, and safe reduced-motion rules in index.css', () => {
    const css = fs.readFileSync(path.resolve(process.cwd(), 'src', 'index.css'), 'utf8');

    for (const requiredToken of [
      'scroll-behavior: smooth;',
      '@keyframes panelSwitchIn',
      '.animate-panel-switch',
      '@keyframes drtechClickRipple',
      '#drtech-motion-ripple-layer',
      '.drtech-ripple-clip',
      '.drtech-click-ripple',
      '.btn-primary-sheen',
      '.animate-stagger-children > *',
      'button:not(:disabled):not([data-no-lift="true"])',
      'cubic-bezier(0.34, 1.56, 0.64, 1)',
      '@media (prefers-reduced-motion: reduce)',
    ]) {
      expect(css).toContain(requiredToken);
    }

    const reducedIdx = css.indexOf('@media (prefers-reduced-motion: reduce)');
    expect(reducedIdx).toBeGreaterThan(-1);
    const reducedSection = css.slice(reducedIdx);
    expect(reducedSection).not.toContain('.animate-standby-');
    expect(reducedSection).not.toContain('transform: none !important');
  });

  it('guards SSR safely when document is unavailable', () => {
    const mockEl = createMockElement('BUTTON') as unknown as HTMLElement;
    expect(spawnTactileRipple(mockEl, 20, 20, undefined)).toBe(false);
    const dispose = installGlobalTactileMotion(undefined);
    expect(typeof dispose).toBe('function');
    dispose();
  });

  it('findTactileTarget resolves enabled controls & text nodes while skipping disabled or data-no-ripple zones', () => {
    const btn = createMockElement('BUTTON', { className: 'btn-tactile' });
    expect(findTactileTarget(btn as unknown as EventTarget)).toBe(btn);

    const textNode = { nodeType: 3, parentElement: btn } as unknown as EventTarget;
    expect(findTactileTarget(textNode)).toBe(btn);

    const disabledBtn = createMockElement('BUTTON', { disabled: true });
    expect(findTactileTarget(disabledBtn as unknown as EventTarget)).toBeNull();

    const ariaDisabled = createMockElement('DIV', {
      attrs: { role: 'button', 'aria-disabled': 'true' },
      closestSelf: ['[role="button"]'],
    });
    expect(findTactileTarget(ariaDisabled as unknown as EventTarget)).toBeNull();

    const standbyScreen = createMockElement('DIV', {
      attrs: { 'data-no-ripple': 'true', role: 'button' },
      noRippleAncestor: true,
    });
    expect(findTactileTarget(standbyScreen as unknown as EventTarget)).toBeNull();
  });

  it('resolveRippleTint selects white sheen for dark/gradient buttons and clinical sky tint for light buttons', () => {
    const gradientBtn = createMockElement('BUTTON', {
      className: 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white',
    });
    expect(resolveRippleTint(gradientBtn as unknown as HTMLElement, null)).toBe('rgba(255, 255, 255, 0.42)');

    const whiteBtn = createMockElement('BUTTON', {
      className: 'bg-white border border-slate-200 text-slate-800',
    });
    expect(resolveRippleTint(whiteBtn as unknown as HTMLElement, null)).toBe('rgba(14, 165, 233, 0.24)');

    const customStyles = new Map<MockElement, { backgroundColor?: string; backgroundImage?: string }>();
    const darkComputedBtn = createMockElement('BUTTON');
    customStyles.set(darkComputedBtn, { backgroundColor: 'rgb(15, 76, 129)' });
    const mockDoc = createMockDocument(customStyles);
    expect(
      resolveRippleTint(darkComputedBtn as unknown as HTMLElement, mockDoc.defaultView as unknown as Window),
    ).toBe('rgba(255, 255, 255, 0.42)');
  });

  it('spawns pointer ripples outside React DOM, caps active count, and cleans up via animationend or fallback timer', () => {
    const doc = createMockDocument();
    const btn = createMockElement('BUTTON', {
      className: 'bg-emerald-600 text-white rounded-2xl',
      rect: { left: 40, top: 80, width: 200, height: 50 },
    });

    const ok = spawnTactileRipple(btn as unknown as HTMLElement, 120, 100, doc as unknown as Document);
    expect(ok).toBe(true);
    // Never mutates React-owned button children
    expect(btn.children.length).toBe(0);

    const layer = doc.getElementById(RIPPLE_LAYER_ID);
    expect(layer).not.toBeNull();
    expect(layer!.children.length).toBe(1);
    const clip = layer!.children[0];
    expect(clip.className).toBe('drtech-ripple-clip');
    expect(clip.style.borderRadius).toBe('16px');
    expect(clip.children[0].className).toBe('drtech-click-ripple');

    // animationend removes clip immediately
    clip.children[0].dispatchEvent('animationend');
    expect(layer!.children.length).toBe(0);

    // Overflow cap & fallback timer cleanup
    for (let i = 0; i < MAX_ACTIVE_RIPPLES + 4; i += 1) {
      spawnTactileRipple(btn as unknown as HTMLElement, 50 + i, 90, doc as unknown as Document);
    }
    expect(layer!.children.length).toBe(MAX_ACTIVE_RIPPLES);

    vi.advanceTimersByTime(650);
    expect(layer!.children.length).toBe(0);
  });

  it('installs global pointerdown/keydown delegation idempotently and removes everything on dispose', () => {
    const doc = createMockDocument();
    const dispose1 = installGlobalTactileMotion(doc as unknown as Document);
    const dispose2 = installGlobalTactileMotion(doc as unknown as Document);
    expect(dispose2).toBe(dispose1);
    expect(doc.getListenerCount('pointerdown')).toBe(1);
    expect(doc.getListenerCount('keydown')).toBe(1);

    const btn = createMockElement('BUTTON', { className: 'bg-white' });
    // Right-click ignored
    doc.dispatchDocEvent('pointerdown', { button: 2, target: btn, clientX: 110, clientY: 210 });
    expect(doc.getElementById(RIPPLE_LAYER_ID)).toBeNull();

    // Left-click spawns ripple
    doc.dispatchDocEvent('pointerdown', { button: 0, target: btn, clientX: 120, clientY: 215 });
    expect(doc.getElementById(RIPPLE_LAYER_ID)!.children.length).toBe(1);

    // Input typing Enter/Space does NOT trigger stray ripples even inside interactive card
    const textInput = createMockElement('INPUT');
    doc.dispatchDocEvent('keydown', { key: ' ', repeat: false, target: textInput });
    expect(doc.getElementById(RIPPLE_LAYER_ID)!.children.length).toBe(1);

    // Keyboard activation on button spawns centered ripple
    doc.dispatchDocEvent('keydown', { key: 'Enter', repeat: false, target: btn });
    expect(doc.getElementById(RIPPLE_LAYER_ID)!.children.length).toBe(2);

    dispose1();
    expect(doc.getListenerCount('pointerdown')).toBe(0);
    expect(doc.getListenerCount('keydown')).toBe(0);
    expect(doc.getElementById(RIPPLE_LAYER_ID)).toBeNull();
  });

  it('wires motion classes across core views/portals and contains zero Unicode emojis in any updated file', () => {
    const root = process.cwd();
    const read = (rel: string) => fs.readFileSync(path.resolve(root, rel), 'utf8');

    expect(read('src/main.tsx')).toContain('installGlobalTactileMotion()');
    expect(read('src/components/KioskStandbyScreen.tsx')).toContain('data-no-ripple="true"');
    expect(read('src/components/KioskStandbyScreen.tsx')).toContain('data-no-lift="true"');
    expect(read('src/App.tsx')).toContain('key="kiosk-overview"');
    expect(read('src/App.tsx')).toContain('key="kiosk-rehab-exercises"');
    expect(read('src/App.tsx')).toContain('animate-stagger-children');
    expect(read('src/StaffLoginModalOrPage.tsx')).toContain('animate-panel-switch');
    expect(read('src/StaffLoginModalOrPage.tsx')).toContain('btn-primary-sheen');
    expect(read('src/components/KioskBiometricLoginView.tsx')).toContain('btn-primary-sheen');
    expect(read('src/components/ClinicalStaffPortalShell.tsx')).toContain('animate-stagger-children');
    expect(read('src/components/PatientClinicalRecordPanel.tsx')).toContain('card-hover-lift animate-card-enter');
    expect(read('src/components/RandomHandStartPrompt.tsx')).toContain('btn-primary-sheen');
    expect(read('src/PatientPersonalPortal.tsx')).toContain('animate-stagger-children');
    expect(read('apps/kiosk/src/DynamicExerciseSessionView.tsx')).toContain('animate-view-enter');
    expect(read('apps/kiosk/src/SmartLoginView.tsx')).toContain('animate-view-enter');
    expect(read('apps/pt/src/SmartEnrollmentView.tsx')).toContain('card-hover-lift animate-card-enter');

    const touchedFiles = [
      'src/index.css',
      'src/main.tsx',
      'src/lib/globalTactileMotion.ts',
      'src/components/KioskStandbyScreen.tsx',
      'src/App.tsx',
      'src/StaffLoginModalOrPage.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/components/ClinicalStaffPortalShell.tsx',
      'src/components/PatientClinicalRecordPanel.tsx',
      'src/components/RandomHandStartPrompt.tsx',
      'src/PatientPersonalPortal.tsx',
      'apps/kiosk/src/DynamicExerciseSessionView.tsx',
      'apps/kiosk/src/SmartLoginView.tsx',
      'apps/pt/src/SmartEnrollmentView.tsx',
      'src/__tests__/globalTactileMotion.test.ts',
    ];

    for (const rel of touchedFiles) {
      expect(EMOJI_RE.test(read(rel)), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });
});

