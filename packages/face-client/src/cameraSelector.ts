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
): Promise<{
  stream: MediaStream;
  activeDeviceId: string | null;
  devices: ClinicalCameraDevice[];
}> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    throw new Error('Browser does not support camera access');
  }

  const ranked = await listClinicalCameraDevices();
  const orderedIds: string[] = [];
  if (preferredDeviceId && ranked.some((d) => d.deviceId === preferredDeviceId)) {
    orderedIds.push(preferredDeviceId);
  }
  for (const dev of ranked) {
    if (dev.deviceId && !orderedIds.includes(dev.deviceId)) {
      orderedIds.push(dev.deviceId);
    }
  }

  const attempts: MediaTrackConstraints[] = [];
  for (let i = 0; i < orderedIds.length; i += 1) {
    const id = orderedIds[i];
    if (i === 0) {
      attempts.push({ deviceId: { exact: id }, width: { ideal: 1920 }, height: { ideal: 1080 } });
    }
    attempts.push({ deviceId: { exact: id }, width: { ideal: 1280 }, height: { ideal: 720 } });
    attempts.push({ deviceId: { exact: id }, width: { ideal: 640 }, height: { ideal: 480 } });
    attempts.push({ deviceId: { exact: id } });
  }
  attempts.push({ facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } });
  attempts.push({ width: { ideal: 640 }, height: { ideal: 480 } });

  let openedStream: MediaStream | null = null;
  let lastErr: unknown = null;
  for (const vc of attempts) {
    try {
      openedStream = await navigator.mediaDevices.getUserMedia({ video: vc, audio: false });
      break;
    } catch (err) {
      lastErr = err;
    }
  }

  if (!openedStream) {
    try {
      openedStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (err) {
      throw lastErr ?? err;
    }
  }

  const trackSettings = openedStream.getVideoTracks()[0]?.getSettings?.();
  const activeDeviceId = trackSettings?.deviceId ?? orderedIds[0] ?? null;
  const refreshedDevices = await listClinicalCameraDevices();

  return {
    stream: openedStream,
    activeDeviceId,
    devices: refreshedDevices,
  };
}
