import { useEffect, useRef } from 'react';
import { useNavigation } from '../context/NavigationContext';
import { mainNavLinks } from '../data/navigation';

/**
 * MobileDrawer — القائمة الجانبية المنزلقة للموبايل.
 * Light Medical Theme — White surface, blue accents.
 * WCAG: role="dialog", aria-labelledby, focus trap, Escape to close.
 */
export default function MobileDrawer() {
  const {
    currentPath,
    isMobileMenuOpen,
    toggleMobileMenu,
    navigate,
    user,
    logout,
  } = useNavigation();

  const drawerRef = useRef<HTMLDivElement>(null);

  // Focus trap + Escape handler
  useEffect(() => {
    if (!isMobileMenuOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        toggleMobileMenu();
        return;
      }
      // Focus trap
      if (e.key === 'Tab' && drawerRef.current) {
        const focusable = drawerRef.current.querySelectorAll<HTMLElement>(
          'button, a, input, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    // Focus the drawer on open
    const timer = setTimeout(() => {
      const firstBtn = drawerRef.current?.querySelector<HTMLElement>('button');
      firstBtn?.focus();
    }, 100);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      clearTimeout(timer);
    };
  }, [isMobileMenuOpen, toggleMobileMenu]);

  if (!isMobileMenuOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-[var(--color-text-primary)]/15 backdrop-blur-sm md:hidden"
        onClick={toggleMobileMenu}
        aria-hidden="true"
      />

      {/* Drawer panel */}
      <div
        ref={drawerRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="drawer-title"
        className="
          fixed top-0 bottom-0 z-50
          w-72 md:hidden
          bg-white/98 backdrop-blur-xl
          border-l border-[var(--color-border)]
          flex flex-col
          animate-slide-in-rtl
          shadow-xl
        "
        style={{ insetInlineStart: 0 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 h-16 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-2">
            <img src="/logo.png" alt="SHIFA AI" className="h-8" />
            <span id="drawer-title" className="text-[var(--color-text-primary)] font-extrabold text-base">
              SHIFA AI
            </span>
          </div>
          <button
            onClick={toggleMobileMenu}
            aria-label="إغلاق القائمة"
            className="w-10 h-10 rounded-lg flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer touch-target"
          >
            <span aria-hidden="true">✕</span>
          </button>
        </div>

        {/* User */}
        <div className="px-4 py-3 border-b border-[var(--color-border)]">
          <div className={`rounded-xl p-3 ${
            user.isAuthenticated
              ? 'bg-[var(--color-primary)]/5 border border-[var(--color-primary)]/10'
              : 'bg-amber-50 border border-amber-200'
          }`}>
            <p className="text-[var(--color-text-primary)] text-sm font-bold">{user.fullName}</p>
            <p className="text-[var(--color-text-muted)] text-xs mt-0.5">
              {user.isAuthenticated ? `@${user.username}` : 'زائر · وصول محدود'}
            </p>
          </div>
        </div>

        {/* Nav Links */}
        <nav className="flex-1 py-3 px-3 space-y-1" role="navigation" aria-label="التنقل الرئيسي">
          {mainNavLinks.map((link) => {
            const isActive = currentPath === link.path;
            return (
              <button
                key={link.id}
                onClick={() => { navigate(link.path); toggleMobileMenu(); }}
                aria-current={isActive ? 'page' : undefined}
                className={`
                  w-full flex items-center gap-3 px-3 py-3 rounded-xl cursor-pointer
                  transition-all duration-200 relative touch-target
                  ${isActive
                    ? 'bg-[var(--color-primary)]/8 text-[var(--color-primary)]'
                    : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)]'
                  }
                `}
              >
                {isActive && (
                  <div className="absolute top-1.5 bottom-1.5 w-[3px] rounded-full bg-[var(--color-primary)]"
                    style={{ insetInlineStart: 0 }} aria-hidden="true" />
                )}
                <span className="text-xl" aria-hidden="true">{link.icon}</span>
                <span className="text-sm font-semibold">{link.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Logout */}
        <div className="px-3 py-3 border-t border-[var(--color-border)]">
          <button
            onClick={logout}
            aria-label="تسجيل الخروج"
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-[var(--color-text-muted)] hover:text-[var(--color-text-danger)] hover:bg-red-50 transition-all cursor-pointer touch-target"
          >
            <span aria-hidden="true">🚪</span>
            <span className="text-sm font-medium">تسجيل الخروج</span>
          </button>
        </div>
      </div>
    </>
  );
}
