import { emergencyNumbers } from '../data/services';

/**
 * EmergencyBanner — شريط تنبيه الطوارئ.
 * Light Medical Theme — Red-50 background with clear red accents.
 * WCAG: role="alert", aria-live="assertive", aria-label on links.
 */
export default function EmergencyBanner({
  ambulanceNumber = emergencyNumbers.ambulance.number,
  policeNumber = emergencyNumbers.police.number,
}) {
  return (
    <div
      id="emergency-banner"
      role="alert"
      aria-live="assertive"
      aria-label="تنبيه طوارئ طبية"
      className="w-full emergency-pulse"
      style={{ direction: 'rtl' }}
    >
      <div className="bg-gradient-to-l from-red-50 via-red-100/80 to-red-50 border-b border-red-200">
        <div className="max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl animate-pulse" aria-hidden="true">🚨</span>
            <div>
              <p className="text-red-900 font-bold text-sm leading-tight">
                تنبيه طوارئ طبية فعلية؟
              </p>
              <p className="text-red-700 text-xs">
                تواصل فوراً مع خدمات الطوارئ. لا تنتظر التطبيق.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <a
              href={`tel:${ambulanceNumber}`}
              aria-label={`اتصل بالإسعاف: ${ambulanceNumber}`}
              className="flex items-center gap-1.5 bg-red-100 hover:bg-red-200 border border-red-300 rounded-xl px-3 py-1.5 transition-all duration-300 group touch-target"
            >
              <span aria-hidden="true" className="text-base">{emergencyNumbers.ambulance.icon}</span>
              <span className="text-red-800 text-sm font-semibold">
                {emergencyNumbers.ambulance.label}: {ambulanceNumber}
              </span>
            </a>
            <a
              href={`tel:${policeNumber}`}
              aria-label={`اتصل بالشرطة: ${policeNumber}`}
              className="flex items-center gap-1.5 bg-red-100 hover:bg-red-200 border border-red-300 rounded-xl px-3 py-1.5 transition-all duration-300 group touch-target"
            >
              <span aria-hidden="true" className="text-base">{emergencyNumbers.police.icon}</span>
              <span className="text-red-800 text-sm font-semibold">
                {emergencyNumbers.police.label}: {policeNumber}
              </span>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
