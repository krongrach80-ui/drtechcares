import React from 'react';
import { AlertTriangle, Home, RotateCcw } from 'lucide-react';

interface ClinicalErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
}

/**
 * Top-level React error boundary preventing white-screen crashes on Kiosk & Portal routes.
 */
export class ClinicalErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ClinicalErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = {
      hasError: false,
      errorMessage: '',
    };
  }

  static getDerivedStateFromError(error: unknown): ClinicalErrorBoundaryState {
    const message = error instanceof Error ? error.message : String(error || 'Unexpected UI error');
    return {
      hasError: true,
      errorMessage: message.slice(0, 240),
    };
  }

  componentDidCatch(error: unknown, errorInfo: React.ErrorInfo): void {
    if (typeof console !== 'undefined' && typeof console.error === 'function') {
      console.error('[ClinicalErrorBoundary]', error, errorInfo.componentStack);
    }
  }

  private handleReload = (): void => {
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  };

  private handleGoHome = (): void => {
    if (typeof window !== 'undefined') {
      window.location.assign('/');
    }
  };

  render(): React.ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div
        data-testid="clinical-error-boundary"
        className="animate-view-enter min-h-screen w-full bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-white flex items-center justify-center p-6 text-[#0F4C81] font-sans"
      >
        <div className="w-full max-w-lg rounded-3xl bg-white border-2 border-sky-200 shadow-2xl p-7 sm:p-9 text-center space-y-5 animate-popup-spring">
          <div className="mx-auto w-16 h-16 rounded-2xl bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center shadow-inner">
            <AlertTriangle className="w-8 h-8" />
          </div>

          <div className="space-y-1.5">
            <span className="inline-block px-3 py-1 rounded-full bg-sky-50 border border-sky-200 text-[11px] font-extrabold tracking-wide text-[#0F4C81]">
              DR.TECH.CARE CLINICAL RECOVERY
            </span>
            <h1 className="text-xl sm:text-2xl font-black text-[#0F4C81]">
              ระบบพบข้อขัดข้องชั่วคราว / Temporary Display Error
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
              หน้าจอกำลังกู้คืนสถานะอย่างปลอดภัย ข้อมูลผู้ป่วยและบันทึกกายภาพบำบัดยังคงปลอดภัยในระบบ
            </p>
          </div>

          {this.state.errorMessage && (
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-left text-xs font-mono text-slate-600 break-words">
              {this.state.errorMessage}
            </div>
          )}

          <div className="flex flex-col sm:flex-row gap-3 pt-1">
            <button
              type="button"
              onClick={this.handleReload}
              className="btn-tactile btn-primary-sheen flex-1 inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#10B981] text-white font-extrabold text-sm px-5 py-3.5 shadow-lg cursor-pointer"
            >
              <RotateCcw className="w-4 h-4 shrink-0" />
              <span>รีโหลดหน้าเว็บ</span>
            </button>
            <button
              type="button"
              onClick={this.handleGoHome}
              className="btn-tactile inline-flex items-center justify-center gap-2 rounded-2xl border-2 border-slate-200 bg-white hover:bg-slate-50 text-[#0F4C81] font-extrabold text-sm px-5 py-3.5 cursor-pointer"
            >
              <Home className="w-4 h-4 shrink-0" />
              <span>กลับหน้าหลัก</span>
            </button>
          </div>
        </div>
      </div>
    );
  }
}
