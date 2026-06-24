/**
 * MentalPage — الصحة النفسية.
 * Light Medical Theme — Purple accent, dark text on white cards.
 */
export default function MentalPage() {
  return (
    <div className="p-6 md:p-10 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-10 animate-float-up">
        <div className="inline-flex items-center gap-2 bg-purple-50 text-purple-700 text-sm font-bold px-4 py-1.5 rounded-full border border-purple-200 mb-4">
          <span>🧠</span>
          <span>دعم نفسي</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[var(--color-text-primary)] mb-3">
          الصحة النفسية
        </h1>
        <p className="text-[var(--color-text-secondary)] text-lg max-w-xl">
          دعم نفسي ومعرفي مخصص بالعربية مع مساعد الذكاء الاصطناعي
        </p>
      </div>

      {/* Content */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 animate-float-up-delay-1">
        {[
          { icon: '🧘', title: 'جلسة استرخاء', desc: 'تمارين تنفس وتأمل موجهة' },
          { icon: '📝', title: 'تقييم المزاج', desc: 'استبيان مختصر لتقييم حالتك' },
          { icon: '💬', title: 'محادثة دعم', desc: 'تحدث مع مساعد الصحة النفسية' },
          { icon: '📊', title: 'تتبع التقدم', desc: 'رسم بياني لمزاجك اليومي' },
        ].map((item) => (
          <div key={item.title} className="glass-card p-6 group">
            <div className="text-3xl mb-3">{item.icon}</div>
            <h3 className="text-lg font-bold text-[var(--color-text-primary)] mb-1">{item.title}</h3>
            <p className="text-[var(--color-text-secondary)] text-sm">{item.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
