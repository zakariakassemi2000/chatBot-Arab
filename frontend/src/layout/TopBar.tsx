import { useNavigation } from '../context/NavigationContext';
import { emergencyNumbers } from '../data/services';

/**
 * TopBar — الشريط العلوي الثابت (64px).
 * Light Medical Theme — White surface, blue accents.
 * WCAG: role="banner", aria-label, aria-live for breadcrumb.
 */
export default function TopBar() {
  const {
    breadcrumb,
    breadcrumbIcon,
    isSidebarCollapsed,
    toggleMobileMenu,
  } = useNavigation();

  const sidebarWidth = isSidebarCollapsed ? 72 : 280;

  return (
    <header
      id="shifa-topbar"
      role="banner"
      className="
        fixed top-0 left-0 right-0 z-30
        h-16 flex items-center justify-between
        bg-white/90 backdrop-blur-xl
        border-b border-[var(--color-border)]
        px-4 md:px-6
        transition-all duration-300
        shadow-sm
      "
      style={{ paddingInlineStart: `${sidebarWidth + 16}px` }}
    >
      {/* Breadcrumb side */}
      <div className="flex items-center gap-4">
        <button
          onClick={toggleMobileMenu}
          className="md:hidden flex items-center justify-center w-11 h-11 rounded-xl bg-[var(--color-bg-tertiary)] hover:bg-[var(--color-primary)]/8 text-[var(--color-text-primary)] transition-colors cursor-pointer touch-target"
          aria-label="فتح القائمة"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        <nav aria-label="مسار التنقل">
          <div className="flex items-center gap-2.5" aria-live="polite">
            <span className="text-xl" aria-hidden="true">{breadcrumbIcon}</span>
            <div>
              <p className="text-[var(--color-text-primary)] font-bold text-sm leading-tight">{breadcrumb}</p>
              <p className="text-[var(--color-text-muted)] text-[11px] hidden sm:block">
                SHIFA AI · لوحة التحكم
              </p>
            </div>
          </div>
        </nav>
      </div>

      {/* Actions side */}
      <div className="flex items-center gap-2" role="toolbar" aria-label="إجراءات سريعة">
        {/* Emergency */}
        <a
          href={`tel:${emergencyNumbers.ambulance.number}`}
          className="hidden sm:flex items-center gap-1.5 bg-red-50 hover:bg-red-100 border border-red-200 rounded-xl px-3 py-1.5 transition-all duration-200 group touch-target"
          aria-label={`اتصل بالإسعاف: ${emergencyNumbers.ambulance.number}`}
        >
          <span aria-hidden="true" className="text-sm">🚑</span>
          <span className="text-[var(--color-text-danger)] text-xs font-semibold">
            {emergencyNumbers.ambulance.number}
          </span>
        </a>

        {/* Status */}
        <div className="flex items-center gap-1.5 bg-emerald-50 border border-emerald-200 rounded-xl px-3 py-1.5" role="status" aria-label="حالة النظام: نشط">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" aria-hidden="true" />
          <span className="text-emerald-700 text-xs font-semibold hidden sm:inline">
            نشط
          </span>
        </div>

        {/* Deploy */}
        <button
          className="flex items-center gap-2 bg-[var(--color-primary)]/8 hover:bg-[var(--color-primary)]/15 border border-[var(--color-primary)]/15 rounded-xl px-3.5 py-1.5 transition-all duration-200 cursor-pointer group touch-target"
          aria-label="نشر التطبيق"
        >
          <svg className="w-3.5 h-3.5 text-[var(--color-primary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
          <span className="text-[var(--color-primary)] text-xs font-bold hidden sm:inline group-hover:text-[var(--color-text-primary)] transition-colors">
            نشر
          </span>
        </button>
      </div>
    </header>
  );
}
