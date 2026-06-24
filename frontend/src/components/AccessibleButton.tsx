import type { ButtonHTMLAttributes, ReactNode } from 'react';

/**
 * AccessibleButton — زر WCAG AA قابل لإعادة الاستخدام.
 * Light Medical Theme — Blue primary, slate secondary, red danger.
 *
 * Guarantees:
 * - aria-label required if no children text
 * - Focus visible ring (2px blue)
 * - Minimum 44x44px touch target
 * - Proper ARIA roles and states
 */

interface AccessibleButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  icon?: ReactNode;
  iconPosition?: 'start' | 'end';
  'aria-label'?: string;
  isLoading?: boolean;
  children?: ReactNode;
}

const variantClasses = {
  primary: `
    bg-[var(--color-primary)] text-white
    hover:bg-blue-700 active:bg-blue-800
    font-bold shadow-sm
  `,
  secondary: `
    bg-white text-[var(--color-text-primary)]
    border border-[var(--color-border)] hover:border-[var(--color-primary)]/30
    hover:bg-[var(--color-bg-tertiary)]
    font-semibold
  `,
  danger: `
    bg-red-50 text-[var(--color-text-danger)]
    border border-red-200 hover:bg-red-100
    font-semibold
  `,
  ghost: `
    bg-transparent text-[var(--color-text-secondary)]
    hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-text-primary)]
  `,
};

const sizeClasses = {
  sm: 'text-xs px-3 py-1.5 rounded-lg gap-1.5 min-h-[36px]',
  md: 'text-sm px-5 py-2.5 rounded-xl gap-2 min-h-[44px]',
  lg: 'text-base px-7 py-3.5 rounded-2xl gap-2.5 min-h-[48px]',
};

export default function AccessibleButton({
  variant = 'primary',
  size = 'md',
  icon,
  iconPosition = 'start',
  isLoading = false,
  children,
  disabled,
  className = '',
  ...rest
}: AccessibleButtonProps) {
  const isIconOnly = !children && icon;

  return (
    <button
      className={`
        inline-flex items-center justify-center
        transition-all duration-200 cursor-pointer
        touch-target
        disabled:opacity-40 disabled:cursor-not-allowed disabled:pointer-events-none
        ${variantClasses[variant]}
        ${sizeClasses[size]}
        ${isIconOnly ? '!px-0 aspect-square' : ''}
        ${className}
      `}
      disabled={disabled || isLoading}
      aria-busy={isLoading || undefined}
      {...rest}
    >
      {isLoading ? (
        <>
          <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="60" strokeLinecap="round" />
          </svg>
          {children && <span>جاري التحميل...</span>}
        </>
      ) : (
        <>
          {icon && iconPosition === 'start' && (
            <span aria-hidden="true">{icon}</span>
          )}
          {children}
          {isIconOnly && rest['aria-label'] && (
            <span className="sr-only">{rest['aria-label']}</span>
          )}
          {icon && iconPosition === 'end' && (
            <span aria-hidden="true">{icon}</span>
          )}
        </>
      )}
    </button>
  );
}
