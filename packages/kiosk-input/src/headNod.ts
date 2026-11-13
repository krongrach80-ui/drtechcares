export const NOD_PITCH_DOWN_THRESHOLD_DEG = 12;
export const NOD_HOLD_DURATION_MS = 300;
export const NOD_RETURN_WINDOW_MS = 1000;
export const NOD_COOLDOWN_MS = 1500;

export class HeadNodDetector {
  private state: 'IDLE' | 'NODDING_DOWN' | 'WAITING_RETURN' = 'IDLE';
  private nodDownStartMs: number | null = null;
  private nodPeakMs: number | null = null;
  private lastConfirmedMs = -Infinity;

  reset(): void {
    this.state = 'IDLE';
    this.nodDownStartMs = null;
    this.nodPeakMs = null;
    this.lastConfirmedMs = -Infinity;
  }

  update(pitchDeg: number, timestampMs: number): { confirmed: boolean; state: string } {
    if (timestampMs - this.lastConfirmedMs < NOD_COOLDOWN_MS) {
      return { confirmed: false, state: 'COOLDOWN' };
    }

    switch (this.state) {
      case 'IDLE': {
        if (pitchDeg >= NOD_PITCH_DOWN_THRESHOLD_DEG) {
          this.state = 'NODDING_DOWN';
          this.nodDownStartMs = timestampMs;
        }
        return { confirmed: false, state: this.state };
      }
      case 'NODDING_DOWN': {
        if (pitchDeg < NOD_PITCH_DOWN_THRESHOLD_DEG * 0.6) {
          this.state = 'IDLE';
          this.nodDownStartMs = null;
          return { confirmed: false, state: this.state };
        }
        if (this.nodDownStartMs !== null && timestampMs - this.nodDownStartMs >= NOD_HOLD_DURATION_MS) {
          this.state = 'WAITING_RETURN';
          this.nodPeakMs = timestampMs;
        }
        return { confirmed: false, state: this.state };
      }
      case 'WAITING_RETURN': {
        if (this.nodPeakMs !== null && timestampMs - this.nodPeakMs > NOD_RETURN_WINDOW_MS) {
          this.state = 'IDLE';
          this.nodDownStartMs = null;
          this.nodPeakMs = null;
          return { confirmed: false, state: this.state };
        }
        if (pitchDeg <= 6) {
          this.state = 'IDLE';
          this.nodDownStartMs = null;
          this.nodPeakMs = null;
          this.lastConfirmedMs = timestampMs;
          return { confirmed: true, state: 'CONFIRMED' };
        }
        return { confirmed: false, state: this.state };
      }
    }
  }
}

