/**
 * LoadingState — حالة التحميل: جاري البحث عن الموقع.
 * Light Medical Theme — Slate skeleton loaders, blue spinner.
 */
export default function LoadingState() {
  return (
    <div className="flex flex-col lg:flex-row gap-5 h-[calc(100vh-120px)] p-4 md:p-6 animate-float-up">
      {/* Map area with spinner */}
      <div className="flex-1 relative rounded-2xl overflow-hidden bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] min-h-[300px]">
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full border-2 border-[var(--color-primary)]/20 animate-ping absolute inset-0" />
            <div className="w-16 h-16 rounded-full bg-[var(--color-primary)]/8 border border-[var(--color-primary)]/20 flex items-center justify-center relative">
              <svg className="w-7 h-7 text-[var(--color-primary)] animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
              </svg>
            </div>
          </div>
          <p className="text-[var(--color-text-primary)] font-bold text-lg">جاري تحديد موقعك...</p>
          <p className="text-[var(--color-text-muted)] text-sm">يرجى السماح بالوصول إلى الموقع</p>
        </div>
      </div>
      {/* Skeleton panel */}
      <div className="w-full lg:w-80 xl:w-96 flex flex-col gap-3">
        <div className="flex gap-2">
          {[1,2,3].map(i=><div key={i} className="h-9 flex-1 rounded-xl bg-slate-100 animate-pulse"/>)}
        </div>
        {[1,2,3,4].map(i=>(
          <div key={i} className="rounded-2xl bg-white border border-[var(--color-border)] p-4 space-y-3 animate-pulse" style={{animationDelay:`${i*150}ms`}}>
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-xl bg-slate-100"/>
              <div className="flex-1 space-y-2"><div className="h-4 w-3/4 rounded bg-slate-100"/><div className="h-3 w-1/2 rounded bg-slate-100"/></div>
            </div>
            <div className="flex gap-2"><div className="h-8 flex-1 rounded-lg bg-slate-100"/><div className="h-8 flex-1 rounded-lg bg-slate-100"/></div>
          </div>
        ))}
      </div>
    </div>
  );
}
