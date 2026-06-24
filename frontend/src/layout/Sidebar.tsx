import { useNavigation } from '../context/NavigationContext';
import { mainNavLinks } from '../data/navigation';

/**
 * Sidebar — القائمة الجانبية (280px desktop / 72px tablet / hidden mobile).
 * Light Medical Theme — White surface, blue accents.
 * WCAG 2.1 AA: role="navigation", aria-label, aria-current, aria-hidden decorative icons.
 */
export default function Sidebar() {
  const {
    currentPath,
    user,
    isSidebarCollapsed,
    toggleSidebar,
    navigate,
    logout,
  } = useNavigation();

  const sidebarWidth = isSidebarCollapsed ? 72 : 280;

  return (
    <aside
      id="shifa-sidebar"
      role="complementary"
      aria-label="القائمة الجانبية"
      className="
        fixed top-0 bottom-0 right-0 z-40
        hidden md:flex flex-col
        bg-white/95 backdrop-blur-xl
        border-l border-[var(--color-border)]
        transition-all duration-300 ease-in-out
        shadow-sm
      "
      style={{
        width: sidebarWidth,
        insetInlineStart: 0,
        insetInlineEnd: 'auto',
      }}
    >
      {/* ── Logo Section ── */}
      <div className="flex items-center gap-3 px-4 h-16 border-b border-[var(--color-border)]">
        <img
          src="/logo.png"
          alt="SHIFA AI — المنصة الطبية الذكية"
          className="h-9 w-auto flex-shrink-0"
        />
        {!isSidebarCollapsed && (
          <div className="overflow-hidden">
            <h2 className="text-base font-extrabold text-[var(--color-text-primary)] leading-tight truncate">
              SHIFA AI
            </h2>
            <p className="text-[10px] text-[var(--color-text-muted)] leading-none">
              المنصة الطبية الذكية
            </p>
          </div>
        )}
      </div>

      {/* ── User Badge ── */}
      <div className="px-3 py-3 border-b border-[var(--color-border)]" aria-label="حالة المستخدم" role="status">
        {isSidebarCollapsed ? (
          <div className="flex justify-center">
            <div
              className={`w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold ${
                user.isAuthenticated
                  ? 'bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                  : 'bg-amber-500/10 text-amber-600'
              }`}
              aria-label={user.isAuthenticated ? `مستخدم: ${user.fullName}` : 'زائر'}
            >
              <span aria-hidden="true">{user.isAuthenticated ? '✓' : '👤'}</span>
            </div>
          </div>
        ) : (
          <div className={`rounded-xl p-2.5 ${
            user.isAuthenticated
              ? 'bg-[var(--color-primary)]/5 border border-[var(--color-primary)]/10'
              : 'bg-amber-50 border border-amber-200'
          }`}>
            <div className="flex items-center gap-2.5">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                  user.isAuthenticated
                    ? 'bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                    : 'bg-amber-100 text-amber-700'
                }`}
                aria-hidden="true"
              >
                {user.isAuthenticated ? '✓' : '👤'}
              </div>
              <div className="min-w-0">
                <p className="text-[var(--color-text-primary)] text-sm font-bold truncate">
                  {user.fullName}
                </p>
                <p className="text-[var(--color-text-muted)] text-[11px]">
                  {user.isAuthenticated ? `@${user.username}` : 'زائر · وصول محدود'}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Navigation Links ── */}
      <nav className="flex-1 py-3 px-2 space-y-1 overflow-y-auto" role="navigation" aria-label="التنقل الرئيسي">
        {mainNavLinks.map((link) => {
          const isActive = currentPath === link.path;
          return (
            <button
              key={link.id}
              onClick={() => navigate(link.path)}
              aria-label={isSidebarCollapsed ? link.label : undefined}
              aria-current={isActive ? 'page' : undefined}
              className={`
                w-full flex items-center gap-3 rounded-xl cursor-pointer
                transition-all duration-200 group relative touch-target
                ${isSidebarCollapsed ? 'justify-center px-0 py-3' : 'px-3 py-2.5'}
                ${isActive
                  ? 'bg-[var(--color-primary)]/8 text-[var(--color-primary)]'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-text-primary)]'
                }
              `}
            >
              {isActive && (
                <div
                  className="absolute top-1 bottom-1 w-[3px] rounded-full bg-[var(--color-primary)]"
                  style={{ insetInlineStart: 0 }}
                  aria-hidden="true"
                />
              )}
              <span className={`text-xl flex-shrink-0 ${isActive ? 'drop-shadow-sm' : ''}`} aria-hidden="true">
                {link.icon}
              </span>
              {!isSidebarCollapsed && (
                <span className={`text-sm font-semibold truncate ${isActive ? 'text-[var(--color-primary)]' : ''}`}>
                  {link.label}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* ── Collapse Toggle ── */}
      <div className="px-2 py-2 border-t border-[var(--color-border)]">
        <button
          onClick={toggleSidebar}
          aria-label={isSidebarCollapsed ? 'توسيع القائمة الجانبية' : 'تصغير القائمة الجانبية'}
          aria-expanded={!isSidebarCollapsed}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-[var(--color-text-muted)] hover:text-[var(--color-primary)] hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer touch-target"
        >
          <svg
            className={`w-4 h-4 transition-transform duration-300 ${isSidebarCollapsed ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
          {!isSidebarCollapsed && <span className="text-xs">تصغير</span>}
        </button>
      </div>

      {/* ── Logout ── */}
      <div className="px-2 py-1.5 border-t border-[var(--color-border)]">
        <button
          onClick={logout}
          aria-label="تسجيل الخروج"
          className={`
            w-full flex items-center gap-2 rounded-lg cursor-pointer touch-target
            text-[var(--color-text-muted)] hover:text-[var(--color-text-danger)] hover:bg-red-50
            transition-all duration-200
            ${isSidebarCollapsed ? 'justify-center py-2.5' : 'px-3 py-2'}
          `}
        >
          <span aria-hidden="true" className="text-lg">🚪</span>
          {!isSidebarCollapsed && <span className="text-xs font-medium">تسجيل الخروج</span>}
        </button>
      </div>

      {/* ── Disclaimer ── */}
      {!isSidebarCollapsed && (
        <div className="px-3 py-3 border-t border-[var(--color-border)]">
          <div className="bg-blue-50 border-r-2 border-[var(--color-primary)]/30 rounded-lg rounded-r-none py-2 px-2.5" role="note">
            <p className="text-[var(--color-text-muted)] text-[10px] leading-relaxed">
              <strong className="text-[var(--color-text-secondary)]">
                <span aria-hidden="true">⚕️</span> إخلاء المسؤولية:
              </strong>{' '}
              لا تُغني عن استشارة الطبيب المختص.
            </p>
          </div>
        </div>
      )}
    </aside>
  );
}
