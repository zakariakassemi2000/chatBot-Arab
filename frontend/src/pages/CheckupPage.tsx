/**
 * CheckupPage — فحص مبدئي.
 * Light Medical Theme — Teal accent, dark text on white cards.
 */
export default function CheckupPage() {
  return (
    <div className="p-6 md:p-10 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-10 animate-float-up">
        <div className="inline-flex items-center gap-2 bg-teal-50 text-teal-700 text-sm font-bold px-4 py-1.5 rounded-full border border-teal-200 mb-4">
          <span>🩺</span>
          <span>التقييم السريري</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[var(--color-text-primary)] mb-3">
          فحص مبدئي
        </h1>
        <p className="text-[var(--color-text-secondary)] text-lg max-w-xl">
          نظام تقييم سريري ذكي يعتمد على بياناتك وبيانات سريرية معتمدة لتحليل الأعراض
        </p>
      </div>

      {/* Content placeholder */}
      <div className="glass-card p-8 text-center animate-float-up-delay-1">
        <div className="text-5xl mb-4">🩺</div>
        <h2 className="text-xl font-bold text-[var(--color-text-primary)] mb-2">نظام التقييم السريري الذكي</h2>
        <p className="text-[var(--color-text-secondary)] mb-6">
          يقوم النظام الاستدلالي بالبحث عن ارتباطات الأعراض لاقتراح الحالات الممكنة
        </p>
        <div className="inline-flex items-center gap-2 bg-emerald-50 text-emerald-700 text-sm font-semibold px-6 py-3 rounded-xl border border-emerald-200">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          متصل بمحرك التحليل
        </div>
      </div>
    </div>
  );
}
