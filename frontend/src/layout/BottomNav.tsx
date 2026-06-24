import { useNavigation } from '../context/NavigationContext';
import { mainNavLinks } from '../data/navigation';

/**
 * BottomNav — شريط التنقل السفلي للموبايل (<768px).
 * Light Medical Theme — White surface, blue active state.
 * WCAG: role="navigation", aria-label, aria-current, touch-target 44px.
 */
export default function BottomNav() {
  const { currentPath, navigate } = useNavigation();

  return (
    <nav
      id="shifa-bottom-nav"
      role="navigation"
      aria-label="التنقل السريع"
      className="
        fixed bottom-0 left-0 right-0 z-40
        md:hidden
        bg-white/95 backdrop-blur-xl
        border-t border-[var(--color-border)]
        safe-area-bottom
        shadow-[0_-2px_8px_rgba(0,0,0,0.04)]
      "
    >
      <div className="flex items-center justify-around h-16 px-1">
        {mainNavLinks.map((link) => {
          const isActive = currentPath === link.path;
          return (
            <button
              key={link.id}
              onClick={() => navigate(link.path)}
              aria-label={link.label}
              aria-current={isActive ? 'page' : undefined}
              className={`
                flex flex-col items-center justify-center gap-0.5
                flex-1 h-full rounded-xl mx-0.5
                transition-all duration-200 cursor-pointer
                touch-target relative
                ${isActive
                  ? 'text-[var(--color-primary)]'
                  : 'text-[var(--color-text-muted)] active:text-[var(--color-text-primary)]'
                }
              `}
            >
              {isActive && (
                <div className="absolute top-1 w-6 h-0.5 rounded-full bg-[var(--color-primary)]" aria-hidden="true" />
              )}
              <span className={`text-xl transition-transform duration-200 ${isActive ? 'scale-110' : ''}`} aria-hidden="true">
                {link.icon}
              </span>
              <span className={`text-[10px] font-semibold leading-none ${isActive ? 'font-bold' : ''}`}>
                {link.label}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
