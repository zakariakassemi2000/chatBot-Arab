import { useNavigate } from 'react-router-dom';

/**
 * HeroSection — القسم الرئيسي مع العنوان وزر البدء.
 * Light Medical Theme — Blue gradient text, blue CTA.
 */
export default function HeroSection() {
  const navigate = useNavigate();

  return (
    <section
      id="hero-section"
      className="relative pt-12 pb-16 flex flex-col items-center text-center px-6"
    >
      {/* Background decorative orbs */}
      <div className="absolute top-20 right-1/4 w-72 h-72 bg-[var(--color-primary)]/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-sky-500/4 rounded-full blur-3xl pointer-events-none" />

      {/* Logo */}
      <div className="animate-float-up mb-6">
        <div className="relative">
          <img
            src="/logo.png"
            alt="SHIFA AI Logo"
            className="h-24 w-auto drop-shadow-lg"
          />
          <div className="absolute inset-0 bg-[var(--color-primary)]/8 rounded-full blur-2xl scale-150 pointer-events-none" />
        </div>
      </div>

      {/* Title H1 */}
      <h1 className="animate-float-up-delay-1 gradient-text text-6xl sm:text-7xl md:text-8xl font-black mb-4 leading-tight tracking-tight">
        SHIFA AI
      </h1>

      {/* Subtitle */}
      <p className="animate-float-up-delay-2 text-[var(--color-text-secondary)] text-lg sm:text-xl md:text-2xl font-medium max-w-2xl mb-12 leading-relaxed">
        مساعدك الطبي الذكي لتقييم الأعراض والتوجيه الصحي
        <br />
        <span className="text-[var(--color-text-muted)] text-base">
          مدعوم بالذكاء الاصطناعي · متاح على مدار الساعة
        </span>
      </p>

      {/* CTA Button — Medical Blue */}
      <div className="animate-float-up-delay-3">
        <button
          onClick={() => navigate('/assistant')}
          id="cta-main"
          className="cta-glow inline-flex items-center gap-3 text-white font-extrabold text-2xl sm:text-3xl px-14 py-6 rounded-2xl transition-all duration-400 cursor-pointer border-none"
        >
          <span>ابدأ الآن</span>
          <svg
            className="w-8 h-8 rotate-180"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2.5}
              d="M13 7l5 5m0 0l-5 5m5-5H6"
            />
          </svg>
        </button>
      </div>

      {/* Trust badges */}
      <div className="animate-float-up-delay-4 flex items-center gap-6 mt-10 text-[var(--color-text-muted)] text-sm">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>النظام نشط</span>
        </div>
        <div className="w-px h-4 bg-[var(--color-border)]" />
        <span>🔒 آمن ومشفر</span>
        <div className="w-px h-4 bg-[var(--color-border)]" />
        <span>⚡ استجابة فورية</span>
      </div>
    </section>
  );
}
