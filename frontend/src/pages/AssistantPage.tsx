/**
 * AssistantPage — المساعد الطبي.
 * Light Medical Theme — Blue accents, dark text on white cards.
 */
export default function AssistantPage() {
  return (
    <div className="p-6 md:p-10 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-10 animate-float-up">
        <div className="inline-flex items-center gap-2 bg-sky-50 text-sky-700 text-sm font-bold px-4 py-1.5 rounded-full border border-sky-200 mb-4">
          <span>💬</span>
          <span>محادثة ذكية</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[var(--color-text-primary)] mb-3">
          المساعد الطبي
        </h1>
        <p className="text-[var(--color-text-secondary)] text-lg max-w-xl">
          اطرح أسئلتك وثق بمحرك SHIFA للإجابة الدقيقة والآمنة
        </p>
      </div>

      {/* Chat placeholder */}
      <div className="glass-card p-8 animate-float-up-delay-1">
        <div className="flex flex-col items-center text-center py-8">
          <div className="text-5xl mb-4">💬</div>
          <h2 className="text-xl font-bold text-[var(--color-text-primary)] mb-2">المحادثة الطبية الذكية</h2>
          <p className="text-[var(--color-text-secondary)] mb-6">
            محادثة ذكية لتقييم حالتك الصحية والإجابة عن أسئلتك الطبية
          </p>

          {/* Simulated chat bubbles */}
          <div className="w-full max-w-lg space-y-3 text-start">
            <div className="bg-[var(--color-primary)]/6 border border-[var(--color-primary)]/12 rounded-2xl rounded-br-md p-4">
              <p className="text-sm text-[var(--color-text-secondary)]">💡 اقتراح: ما هي الأعراض المبكرة للسكري؟</p>
            </div>
            <div className="bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-2xl rounded-bl-md p-4 ms-8">
              <p className="text-sm text-[var(--color-text-secondary)]">اكتب سؤالك الطبي للبدء...</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
