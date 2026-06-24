import { useNavigate } from 'react-router-dom';

/**
 * ServiceCard — بطاقة خدمة طبية قابلة لإعادة الاستخدام.
 * Light Medical Theme — White card, blue hover accents.
 */
export default function ServiceCard({
  title,
  icon,
  description,
  link = '#',
  isActive = true,
  gradient = '',
  size = 'default', // 'default' | 'compact'
  animationDelay = '',
}) {
  const navigate = useNavigate();

  const sizeClasses =
    size === 'compact'
      ? 'p-5'
      : 'p-7';

  const iconSize =
    size === 'compact'
      ? 'text-3xl'
      : 'text-4xl';

  const titleSize =
    size === 'compact'
      ? 'text-base'
      : 'text-xl';

  const handleClick = (e) => {
    e.preventDefault();
    if (!isActive) return;
    // Internal route or external URL
    if (link.startsWith('/')) {
      navigate(link);
    } else {
      window.open(link, '_blank');
    }
  };

  return (
    <div
      role="button"
      tabIndex={isActive ? 0 : -1}
      onClick={handleClick}
      onKeyDown={(e) => e.key === 'Enter' && handleClick(e)}
      className={`
        glass-card block relative overflow-hidden cursor-pointer
        ${sizeClasses}
        ${animationDelay}
        ${!isActive ? 'opacity-40 pointer-events-none grayscale' : ''}
      `}
    >
      {/* Background gradient overlay */}
      {gradient && (
        <div
          className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-30 pointer-events-none`}
        />
      )}

      {/* Content */}
      <div className="relative z-10">
        {/* Icon */}
        <div className={`${iconSize} mb-4`}>
          {icon}
        </div>

        {/* Title */}
        <h3 className={`${titleSize} font-bold text-[var(--color-text-primary)] mb-2 leading-snug`}>
          {title}
        </h3>

        {/* Description */}
        <p className="text-[var(--color-text-secondary)] text-sm leading-relaxed">
          {description}
        </p>

        {/* Inactive badge */}
        {!isActive && (
          <div className="absolute top-4 left-4 bg-slate-100 text-[var(--color-text-muted)] text-xs font-bold px-3 py-1 rounded-full border border-[var(--color-border)]">
            قريباً
          </div>
        )}

        {/* Arrow indicator */}
        {isActive && (
          <div className="mt-4 flex items-center gap-2 text-[var(--color-primary)] text-sm font-semibold opacity-0 group-hover:opacity-100 transition-opacity">
            <svg
              className="w-4 h-4 rotate-180 transition-transform group-hover:-translate-x-1"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 7l5 5m0 0l-5 5m5-5H6"
              />
            </svg>
            <span>فتح</span>
          </div>
        )}
      </div>

      {/* Hover border glow — bottom accent line */}
      <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-l from-transparent via-[var(--color-primary)]/40 to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
    </div>
  );
}
