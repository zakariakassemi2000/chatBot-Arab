/**
 * Footer — إخلاء المسؤولية وحقوق النشر.
 * Light Medical Theme — Clean white/slate footer.
 */
export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer id="app-footer" className="section-gap border-t border-[var(--color-border)]">
      <div className="max-w-6xl mx-auto px-4 py-12">
        {/* Disclaimer */}
        <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6 mb-10">
          <div className="flex items-start gap-4">
            <div className="text-3xl mt-1">⚕️</div>
            <div>
              <h3 className="text-[var(--color-text-primary)] font-bold text-lg mb-2">
                تنبيه إخلاء المسؤولية
              </h3>
              <p className="text-[var(--color-text-secondary)] text-sm leading-relaxed">
                المنصة توفر دعماً معلوماتياً فقط. لا تُغني أبداً عن استشارة الطبيب المختص أو زيارة العيادة.
                جميع النتائج والتحليلات مخصصة للأغراض التعليمية والأكاديمية.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-[var(--color-text-muted)] text-sm">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="SHIFA AI" className="h-8 w-auto opacity-70" />
            <span className="font-semibold text-[var(--color-text-secondary)]">
              SHIFA AI
            </span>
            <span className="text-[var(--color-text-muted)]">·</span>
            <span>المنصة الطبية الذكية</span>
          </div>

          {/* Links */}
          <div className="flex flex-wrap items-center gap-4">
            <a
              href="http://localhost:8503"
              className="hover:text-[var(--color-primary)] transition-colors"
            >
              👨‍⚕️ فضاء الطبيب
            </a>
            <div className="w-px h-4 bg-[var(--color-border)]" />
            <a href="#" className="hover:text-[var(--color-primary)] transition-colors">سياسة الخصوصية</a>
            <div className="w-px h-4 bg-[var(--color-border)]" />
            <a href="#" className="hover:text-[var(--color-primary)] transition-colors">شروط الاستخدام</a>
            <div className="w-px h-4 bg-[var(--color-border)]" />
            <a href="#" className="hover:text-[var(--color-primary)] transition-colors">اتصل بنا</a>
            <div className="w-px h-4 bg-[var(--color-border)]" />
            <span>© {currentYear} SHIFA AI</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
