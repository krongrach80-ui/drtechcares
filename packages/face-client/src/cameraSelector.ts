const EXCLUDED_CAMERA_LABELS = [/infrared/i, /\bir\b/i, /windows hello/i, /virtual/i, /obs/i, /snap/i];
const PREFERRED_EXTERNAL_LABELS = [/logitech/i, /usb/i, /hd pro/i, /brio/i, /c920/i, /c922/i, /webcam/i];

export async function selectClinicalCameraDeviceId(): Promise<string | undefined> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.enumerateDevices) {
    return undefined;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoInputs = devices.filter((d) => d.kind === 'videoinput');
  if (videoInputs.length === 0) return undefined;

  const usable = videoInputs.filter(
    (d) => !EXCLUDED_CAMERA_LABELS.some((pattern) => pattern.test(d.label || '')),
  );
  const pool = usable.length > 0 ? usable : videoInputs;

  const external = pool.find((d) =>
    PREFERRED_EXTERNAL_LABELS.some((pattern) => pattern.test(d.label || '')),
  );
  return (external ?? pool[0]).deviceId || undefined;
}

