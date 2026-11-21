export const RIPPLE_LAYER_ID = 'drtech-motion-ripple-layer';
export const MAX_ACTIVE_RIPPLES = 8;
export const TACTILE_INTERACTIVE_SELECTOR =
  'button:not(:disabled), [role="button"]:not([aria-disabled="true"]), a[href], .btn-tactile:not(:disabled), .calendar-day-tile:not(:disabled)';

const RIPPLE_FALLBACK_MS = 580;
const DARK_CLASS_RE =
  /\b(bg-gradient-|bg-emerald-[5-9]|bg-teal-[5-9]|bg-sky-[5-9]|bg-cyan-[6-9]|bg-blue-[6-9]|bg-indigo-[5-9]|bg-red-[5-9]|bg-rose-[5-9]|bg-slate-[7-9]|bg-\[#0F4C81\]|text-white)\b/;

const installedDocs = new WeakMap<Document, () => void>();

interface PartialElementLike {
  nodeType?: number;
  tagName?: string;
  parentElement?: PartialElementLike | null;
  closest?: (selector: string) => PartialElementLike | null;
  disabled?: boolean;
  hasAttribute?: (name: string) => boolean;
  getAttribute?: (name: string) => string | null;
  dataset?: Record<string, string | undefined>;
  className?: string;
  getBoundingClientRect?: () => { left: number; top: number; width: number; height: number };
}

function asElementLike(target: EventTarget | null): PartialElementLike | null {
  if (!target || typeof target !== 'object') return null;
  const candidate = target as PartialElementLike;
  if (candidate.nodeType === 3) {
    return candidate.parentElement ?? null;
  }
  return typeof candidate.closest === 'function' ? candidate : null;
}

/**
 * Resolves the nearest enabled interactive control eligible for click/tap tactile ripples,
 * respecting `[data-no-ripple="true"]` opt-out zones and disabled controls.
 */
export function findTactileTarget(target: EventTarget | null): HTMLElement | null {
  const el = asElementLike(target);
  if (!el || typeof el.closest !== 'function') return null;

  if (el.closest('[data-no-ripple="true"]')) {
    return null;
  }

  const interactive = el.closest(TACTILE_INTERACTIVE_SELECTOR);
  if (!interactive) return null;

  const isDisabled =
    Boolean(interactive.disabled) ||
    Boolean(interactive.hasAttribute?.('disabled')) ||
    interactive.getAttribute?.('aria-disabled') === 'true' ||
    interactive.dataset?.noRipple === 'true' ||
    interactive.getAttribute?.('data-no-ripple') === 'true';

  if (isDisabled) return null;
  return interactive as unknown as HTMLElement;
}

/**
 * Chooses a high-contrast ripple tint (white sheen on dark/gradient buttons, soft medical-sky tint on light surfaces).
 */
export function resolveRippleTint(el: HTMLElement, view?: Window | null): string {
  const className = typeof el.className === 'string' ? el.className : '';
  if (DARK_CLASS_RE.test(className)) {
    return 'rgba(255, 255, 255, 0.42)';
  }

  const win = view ?? (typeof window !== 'undefined' ? window : null);
  if (win && typeof win.getComputedStyle === 'function') {
    try {
      const style = win.getComputedStyle(el);
      if (style.backgroundImage && style.backgroundImage !== 'none' && style.backgroundImage.includes('gradient')) {
        return 'rgba(255, 255, 255, 0.42)';
      }
      const rgbMatch = style.backgroundColor?.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?/i);
      if (rgbMatch) {
        const r = Number(rgbMatch[1]);
        const g = Number(rgbMatch[2]);
        const b = Number(rgbMatch[3]);
        const alpha = rgbMatch[4] !== undefined ? Number(rgbMatch[4]) : 1;
        if (alpha > 0.25) {
          const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
          if (luminance < 0.62) {
            return 'rgba(255, 255, 255, 0.42)';
          }
        }
      }
    } catch {
      // Fallback safely to clinical sky tint
    }
  }

  return 'rgba(14, 165, 233, 0.24)';
}

function getOrCreateRippleLayer(doc: Document): HTMLElement | null {
  if (!doc.body) return null;
  const existing = doc.getElementById(RIPPLE_LAYER_ID);
  if (existing) return existing;

  const layer = doc.createElement('div');
  layer.id = RIPPLE_LAYER_ID;
  layer.setAttribute('aria-hidden', 'true');
  doc.body.appendChild(layer);
  return layer;
}

/**
 * Spawns a GPU-composited ripple inside `#drtech-motion-ripple-layer` (outside React-owned DOM trees).
 */
export function spawnTactileRipple(
  el: HTMLElement,
  clientX: number,
  clientY: number,
  doc?: Document,
): boolean {
  const targetDoc = doc ?? (typeof document !== 'undefined' ? document : null);
  if (!targetDoc || typeof el.getBoundingClientRect !== 'function') return false;

  const rect = el.getBoundingClientRect();
  if (!Number.isFinite(rect.width) || !Number.isFinite(rect.height) || rect.width < 8 || rect.height < 8) {
    return false;
  }

  const layer = getOrCreateRippleLayer(targetDoc);
  if (!layer) return false;

  while (layer.children.length >= MAX_ACTIVE_RIPPLES && layer.firstElementChild) {
    layer.firstElementChild.remove();
  }

  const win = targetDoc.defaultView ?? (typeof window !== 'undefined' ? window : null);
  let borderRadius = '12px';
  if (win && typeof win.getComputedStyle === 'function') {
    try {
      borderRadius = win.getComputedStyle(el).borderRadius || '12px';
    } catch {
      borderRadius = '12px';
    }
  }

  const clip = targetDoc.createElement('span');
  clip.className = 'drtech-ripple-clip';
  clip.style.left = `${rect.left}px`;
  clip.style.top = `${rect.top}px`;
  clip.style.width = `${rect.width}px`;
  clip.style.height = `${rect.height}px`;
  clip.style.borderRadius = borderRadius;

  const localX = Math.max(0, Math.min(rect.width, (Number.isFinite(clientX) ? clientX : rect.left + rect.width / 2) - rect.left));
  const localY = Math.max(0, Math.min(rect.height, (Number.isFinite(clientY) ? clientY : rect.top + rect.height / 2) - rect.top));
  const diameter = Math.max(24, Math.ceil(Math.max(rect.width, rect.height) * 2.3));

  const wave = targetDoc.createElement('span');
  wave.className = 'drtech-click-ripple';
  wave.style.left = `${localX}px`;
  wave.style.top = `${localY}px`;
  wave.style.width = `${diameter}px`;
  wave.style.height = `${diameter}px`;
  wave.style.background = resolveRippleTint(el, win);

  clip.appendChild(wave);
  layer.appendChild(clip);

  let removed = false;
  let timerId: ReturnType<typeof setTimeout> | null = null;
  const cleanupClip = () => {
    if (removed) return;
    removed = true;
    if (timerId !== null) {
      clearTimeout(timerId);
      timerId = null;
    }
    clip.remove();
  };

  if (typeof wave.addEventListener === 'function') {
    wave.addEventListener('animationend', cleanupClip, { once: true });
  }
  timerId = setTimeout(cleanupClip, RIPPLE_FALLBACK_MS);

  return true;
}

/**
 * Installs a single delegated pointer + keyboard tactile ripple listener on `doc` (idempotent per document).
 */
export function installGlobalTactileMotion(doc?: Document): () => void {
  const targetDoc = doc ?? (typeof document !== 'undefined' ? document : null);
  if (!targetDoc) {
    return () => {};
  }

  const existingCleanup = installedDocs.get(targetDoc);
  if (existingCleanup) {
    return existingCleanup;
  }

  const handlePointerDown = (evt: Event) => {
    const pointerEvt = evt as PointerEvent;
    if (typeof pointerEvt.button === 'number' && pointerEvt.button > 0) return;
    const interactive = findTactileTarget(evt.target);
    if (!interactive) return;
    spawnTactileRipple(interactive, pointerEvt.clientX, pointerEvt.clientY, targetDoc);
  };

  const handleKeyDown = (evt: Event) => {
    const keyEvt = evt as KeyboardEvent;
    if (keyEvt.repeat || (keyEvt.key !== 'Enter' && keyEvt.key !== ' ')) return;
    const rawTarget = asElementLike(evt.target);
    if (rawTarget?.tagName === 'INPUT' || rawTarget?.tagName === 'TEXTAREA' || rawTarget?.tagName === 'SELECT') {
      return;
    }
    const interactive = findTactileTarget(evt.target);
    if (!interactive) return;
    const rect = interactive.getBoundingClientRect();
    spawnTactileRipple(
      interactive,
      rect.left + rect.width / 2,
      rect.top + rect.height / 2,
      targetDoc,
    );
  };

  targetDoc.addEventListener('pointerdown', handlePointerDown, { passive: true });
  targetDoc.addEventListener('keydown', handleKeyDown, { passive: true });

  const dispose = () => {
    targetDoc.removeEventListener('pointerdown', handlePointerDown);
    targetDoc.removeEventListener('keydown', handleKeyDown);
    const layer = targetDoc.getElementById(RIPPLE_LAYER_ID);
    layer?.remove();
    installedDocs.delete(targetDoc);
  };

  installedDocs.set(targetDoc, dispose);
  return dispose;
}

