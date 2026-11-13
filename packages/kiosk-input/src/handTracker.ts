export const HOVER_ZONE_LEFT_MAX_X = 0.25;
export const HOVER_ZONE_RIGHT_MIN_X = 0.75;
export const HOVER_DWELL_MS = 1000;

export type HoverZone = 'left' | 'right' | 'center' | null;

class SimpleOneEuro1D {
  private initialized = false;
  private xPrev = 0;
  private dxPrev = 0;
  private lastTs: number | null = null;

  constructor(private readonly minCutoff = 1.0, private readonly beta = 0.007) {}

  filter(val: number, tsMs: number): number {
    if (!this.initialized || this.lastTs === null) {
      this.initialized = true;
      this.xPrev = val;
      this.dxPrev = 0;
      this.lastTs = tsMs;
      return val;
    }
    const dt = Math.max(0.001, (tsMs - this.lastTs) / 1000);
    this.lastTs = tsMs;
    const dx = (val - this.xPrev) / dt;
    const edx = 0.3 * dx + 0.7 * this.dxPrev;
    this.dxPrev = edx;
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const tau = 1 / (2 * Math.PI * cutoff);
    const alpha = 1 / (1 + tau / dt);
    const x = alpha * val + (1 - alpha) * this.xPrev;
    this.xPrev = x;
    return x;
  }

  reset(): void {
    this.initialized = false;
    this.lastTs = null;
  }
}

export class SingleHandGestureTracker {
  private readonly filterX = new SimpleOneEuro1D(1.0, 0.007);
  private readonly filterY = new SimpleOneEuro1D(1.0, 0.007);
  private activeZone: HoverZone = null;
  private zoneStartMs: number | null = null;

  reset(): void {
    this.filterX.reset();
    this.filterY.reset();
    this.activeZone = null;
    this.zoneStartMs = null;
  }

  update(
    indexTip: { x: number; y: number } | null,
    timestampMs: number,
  ): {
    cursor: { x: number; y: number } | null;
    zone: HoverZone;
    dwellProgress: number;
    triggeredZone: 'left' | 'right' | null;
  } {
    if (!indexTip) {
      this.activeZone = null;
      this.zoneStartMs = null;
      return { cursor: null, zone: null, dwellProgress: 0, triggeredZone: null };
    }

    const x = this.filterX.filter(indexTip.x, timestampMs);
    const y = this.filterY.filter(indexTip.y, timestampMs);

    const zone: HoverZone =
      x < HOVER_ZONE_LEFT_MAX_X ? 'left' : x > HOVER_ZONE_RIGHT_MIN_X ? 'right' : 'center';

    if (zone !== 'left' && zone !== 'right') {
      this.activeZone = zone;
      this.zoneStartMs = null;
      return { cursor: { x, y }, zone, dwellProgress: 0, triggeredZone: null };
    }

    if (this.activeZone !== zone || this.zoneStartMs === null) {
      this.activeZone = zone;
      this.zoneStartMs = timestampMs;
    }

    const elapsed = timestampMs - this.zoneStartMs;
    const dwellProgress = Math.min(1, elapsed / HOVER_DWELL_MS);
    let triggeredZone: 'left' | 'right' | null = null;
    if (elapsed >= HOVER_DWELL_MS) {
      triggeredZone = zone;
      this.zoneStartMs = timestampMs + 500;
    }

    return {
      cursor: { x, y },
      zone,
      dwellProgress,
      triggeredZone,
    };
  }
}

