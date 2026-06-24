import { type InputHTMLAttributes, type ReactNode, useId } from 'react';

/**
 * AccessibleInput — حقل إدخال WCAG AA قابل لإعادة الاستخدام.
 * Light Medical Theme — White card bg, blue focus border.
 *
 * Guarantees:
 * - <label> explicitly linked by htmlFor/id
 * - aria-describedby for helper text and errors
 * - aria-invalid + aria-errormessage for error state
 * - Focus visible ring (2px blue)
 * - Minimum 44px height
 */

interface AccessibleInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'id'> {
  label: string;
  helperText?: string;
  error?: string;
  icon?: ReactNode;
  iconPosition?: 'start' | 'end';
  wrapperClassName?: string;
}

export default function AccessibleInput({
  label,
  helperText,
  error,
  icon,
  iconPosition = 'start',
  wrapperClassName = '',
  className = '',
  required,
  ...rest
}: AccessibleInputProps) {
  const autoId = useId();
  const inputId = `input-${autoId}`;
  const helperId = helperText ? `helper-${autoId}` : undefined;
  const errorId = error ? `error-${autoId}` : undefined;

  const describedBy = [helperId, errorId].filter(Boolean).join(' ') || undefined;

  return (
    <div className={`w-full ${wrapperClassName}`}>
      {/* Label — always visible */}
      <label
        htmlFor={inputId}
        className="block text-[var(--color-text-secondary)] text-sm font-semibold mb-1.5"
      >
        {label}
        {required && (
          <span className="text-[var(--color-text-danger)] ms-1" aria-hidden="true">*</span>
        )}
      </label>

      {/* Input wrapper */}
      <div className={`
        relative flex items-center
        bg-white border rounded-xl
        transition-all duration-200
        shadow-sm
        ${error
          ? 'border-red-300 focus-within:border-red-500 focus-within:ring-1 focus-within:ring-red-200'
          : 'border-[var(--color-border)] focus-within:border-[var(--color-primary)] focus-within:ring-1 focus-within:ring-[var(--color-primary)]/20'
        }
      `}>
        {icon && iconPosition === 'start' && (
          <span className="ps-3 text-[var(--color-text-muted)] flex-shrink-0" aria-hidden="true">
            {icon}
          </span>
        )}

        <input
          id={inputId}
          className={`
            w-full bg-transparent
            text-[var(--color-text-primary)] text-sm
            py-3 px-3
            outline-none
            placeholder:text-slate-400
            min-h-[44px]
            ${className}
          `}
          dir="rtl"
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          aria-errormessage={errorId}
          aria-required={required || undefined}
          required={required}
          {...rest}
        />

        {icon && iconPosition === 'end' && (
          <span className="pe-3 text-[var(--color-text-muted)] flex-shrink-0" aria-hidden="true">
            {icon}
          </span>
        )}
      </div>

      {/* Helper text */}
      {helperText && !error && (
        <p id={helperId} className="mt-1 text-[var(--color-text-muted)] text-xs">
          {helperText}
        </p>
      )}

      {/* Error message */}
      {error && (
        <p id={errorId} className="mt-1 text-[var(--color-text-danger)] text-xs font-medium flex items-center gap-1" role="alert">
          <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
          </svg>
          <span>{error}</span>
        </p>
      )}
    </div>
  );
}
