/**
 * ErrorState — حالة الخطأ: رفض الموقع أو عدم وجود نتائج.
 * Light Medical Theme — Red-50 error icon, blue retry button.
 */
interface ErrorStateProps {
  errorType: 'denied' | 'unavailable' | 'timeout' | 'no_results';
  onRetry: () => void;
  onManualSearch: (q: string) => void;
}

const errorMessages: Record<string, { title: string; desc: string; icon: string }> = {
  denied: { title: 'تم رفض الوصول إلى الموقع', desc: 'يرجى تفعيل خدمة الموقع من إعدادات المتصفح ثم المحاولة مرة أخرى', icon: '🚫' },
  unavailable: { title: 'خدمة الموقع غير متاحة', desc: 'تأكد من تفعيل GPS على جهازك وأن المتصفح يدعم تحديد الموقع', icon: '📡' },
  timeout: { title: 'انتهت مهلة تحديد الموقع', desc: 'استغرق الأمر وقتاً طويلاً. تأكد من اتصالك بالإنترنت وحاول مرة أخرى', icon: '⏱️' },
  no_results: { title: 'لا توجد نتائج قريبة', desc: 'لم نجد مرافق صحية في محيطك. جرب توسيع نطاق البحث أو البحث يدوياً', icon: '🔍' },
};

export default function ErrorState({ errorType, onRetry, onManualSearch }: ErrorStateProps) {
  const msg = errorMessages[errorType] || errorMessages.unavailable;

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4 animate-float-up">
      {/* Error illustration */}
      <div className="mb-6">
        <div className="w-24 h-24 mx-auto rounded-full bg-red-50 border border-red-200 flex items-center justify-center">
          <span className="text-5xl">{msg.icon}</span>
        </div>
      </div>

      <h2 className="text-2xl font-extrabold text-[var(--color-text-primary)] mb-2 text-center">{msg.title}</h2>
      <p className="text-[var(--color-text-secondary)] text-center max-w-md mb-8">{msg.desc}</p>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-3 w-full max-w-md">
        <button onClick={onRetry}
          className="flex-1 flex items-center justify-center gap-2 bg-[var(--color-primary)] text-white font-bold px-6 py-3 rounded-2xl cursor-pointer hover:scale-[1.02] transition-transform shadow-sm">
          <span>🔄</span><span>إعادة المحاولة</span>
        </button>
        <button onClick={() => onManualSearch('مراكش')}
          className="flex-1 flex items-center justify-center gap-2 bg-white border border-[var(--color-border)] text-[var(--color-text-primary)] font-medium px-6 py-3 rounded-2xl cursor-pointer hover:bg-[var(--color-bg-tertiary)] transition-colors shadow-sm">
          <span>🔍</span><span>بحث يدوي</span>
        </button>
      </div>
    </div>
  );
}
