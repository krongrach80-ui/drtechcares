const EXCLUDED_CAMERA_LABELS = [/infrared/i, /\bir\b/i, /windows hello/i, /virtual/i, /obs/i, /snap/i];
const PREFERRED_EXTERNAL_LABELS = [/logitech/i, /usb/i, /hd pro/i, /brio/i, /c920/i, /c922/i, /webcam/i];
const FRONT_CAMERA_LABELS = [/front/i, /user/i, /facetime/i, /integrated/i];

export interface ClinicalCameraDevice {
  deviceId: string;
  label: string;
  isExternal: boolean;
  isFrontFacing: boolean;
}

function rankClinicalCameraDevices(devices: MediaDeviceInfo[]): ClinicalCameraDevice[] {
  const videoInputs = devices.filter((d) => d.kind === 'videoinput');
  if (videoInputs.length === 0) return [];

  const usable = videoInputs.filter(
    (d) => !EXCLUDED_CAMERA_LABELS.some((pattern) => pattern.test(d.label || '')),
  );
  const pool = usable.length > 0 ? usable : videoInputs;

  return pool
    .map((d, idx) => {
      const label = (d.label || '').trim() || `Camera ${idx + 1}`;
      const isIntegrated = /integrated|built.?in|internal|facetime/i.test(label);
      const isExternal =
        !isIntegrated && PREFERRED_EXTERNAL_LABELS.some((pattern) => pattern.test(label));
      const isFrontFacing = FRONT_CAMERA_LABELS.some((pattern) => pattern.test(label));
      return {
        deviceId: d.deviceId,
        label,
        isExternal,
        isFrontFacing,
      };
    })
    .sort(
      (a, b) =>
        Number(b.isExternal) - Number(a.isExternal) ||
        Number(b.isFrontFacing) - Number(a.isFrontFacing),
    );
}

export async function listClinicalCameraDevices(): Promise<ClinicalCameraDevice[]> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return rankClinicalCameraDevices(devices);
  } catch {
    return [];
  }
}

export async function selectClinicalCameraDeviceId(
  preferredDeviceId?: string | null,
): Promise<string | undefined> {
  const ranked = await listClinicalCameraDevices();
  if (ranked.length === 0) return undefined;
  if (preferredDeviceId) {
    const matched = ranked.find((d) => d.deviceId === preferredDeviceId);
    if (matched?.deviceId) return matched.deviceId;
  }
  return ranked[0]?.deviceId || undefined;
}

export async function openBestAvailableCameraStream(
  preferredDeviceId?: string | null,
  options?: { skipPostEnumerate?: boolean },
): Promise<{
  stream: MediaStream;
  activeDeviceId: string | null;
  devices: ClinicalCameraDevice[];
  probeCount: number;
  openDurationMs: number;
  usedRememberedDevice: boolean;
}> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    throw new Error('Browser does not support camera access');
  }

  const startMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
  let openedStream: MediaStream | null = null;
  let lastErr: unknown = null;
  let probeCount = 0;
  let usedRememberedDevice = false;

  // Fast direct-open path for remembered camera deviceId: NEVER enumerate or probe all 5 cameras first!
  const trimmedPreferred = preferredDeviceId?.trim() || null;
  if (trimmedPreferred) {
    const directAttempts: MediaTrackConstraints[] = [
      {
        deviceId: { exact: trimmedPreferred },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
      {
        deviceId: { exact: trimmedPreferred },
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      { deviceId: { exact: trimmedPreferred } },
    ];
    for (const vc of directAttempts) {
      probeCount += 1;
      try {
        openedStream = await navigator.mediaDevices.getUserMedia({ video: vc, audio: false });
        usedRememberedDevice = true;
        break;
      } catch (err) {
        lastErr = err;
        const errName = (err as { name?: string } | null)?.name;
        // If remembered camera was unplugged/missing, fall through immediately to ranked cameras
        if (errName === 'NotFoundError' || errName === 'DevicesNotFoundError') {
          break;
        }
      }
    }
  }

  // Fallback path ONLY when no remembered device exists or remembered device failed/unplugged
  let ranked: ClinicalCameraDevice[] = [];
  if (!openedStream) {
    ranked = await listClinicalCameraDevices();
    const fallbackIds: string[] = [];
    for (const dev of ranked) {
      if (dev.deviceId && dev.deviceId !== trimmedPreferred && !fallbackIds.includes(dev.deviceId)) {
        fallbackIds.push(dev.deviceId);
      }
    }

    const attempts: MediaTrackConstraints[] = [];
    for (const id of fallbackIds) {
      attempts.push({
        deviceId: { exact: id },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      });
      attempts.push({ deviceId: { exact: id }, width: { ideal: 640 }, height: { ideal: 480 } });
      attempts.push({ deviceId: { exact: id } });
    }
    attempts.push({ facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } });
    attempts.push({ width: { ideal: 640 }, height: { ideal: 480 } });

    for (const vc of attempts) {
      probeCount += 1;
      try {
        openedStream = await navigator.mediaDevices.getUserMedia({ video: vc, audio: false });
        break;
      } catch (err) {
        lastErr = err;
      }
    }
  }

  if (!openedStream) {
    probeCount += 1;
    try {
      openedStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (err) {
      throw lastErr ?? err;
    }
  }

  const trackSettings = openedStream.getVideoTracks()[0]?.getSettings?.();
  const activeDeviceId = trackSettings?.deviceId ?? trimmedPreferred ?? ranked[0]?.deviceId ?? null;
  const devices = options?.skipPostEnumerate ? ranked : await listClinicalCameraDevices();
  const endMs = typeof performance !== 'undefined' ? performance.now() : Date.now();

  return {
    stream: openedStream,
    activeDeviceId,
    devices,
    probeCount,
    openDurationMs: Number(Math.max(0, endMs - startMs).toFixed(1)),
    usedRememberedDevice,
  };
}
