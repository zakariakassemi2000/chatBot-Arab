import { useState } from 'react';
import ServiceCard from './ServiceCard';
import { advancedModules } from '../data/services';

/**
 * AdvancedModules — الموديولات المتقدمة (شبكة 2×3 قابلة للطي).
 * Light Medical Theme — Blue accents, white cards, primary text.
 */
export default function AdvancedModules() {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <section id="advanced-modules" className="section-gap max-w-6xl mx-auto px-6 sm:px-8">
      {/* Section Header */}
      <div className="text-center mb-10 animate-float-up">
        <div className="inline-flex items-center gap-2 bg-sky-50 text-sky-700 text-sm font-bold px-4 py-1.5 rounded-full border border-sky-200 mb-4">
          <span>🔬</span>
          <span>أدوات متقدمة</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-[var(--color-text-primary)] mb-3">
          الموديولات المتقدمة
        </h2>
        <p className="text-[var(--color-text-secondary)] text-lg max-w-xl mx-auto">
          أدوات متخصصة للتحليل العميق والرعاية الشاملة
        </p>
      </div>

      {/* Toggle Button */}
      <div className="flex justify-center mb-8">
        <button
          id="toggle-modules"
          onClick={() => setIsExpanded(!isExpanded)}
          className="group flex items-center gap-3 bg-white hover:bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] hover:border-[var(--color-primary)]/30 rounded-2xl px-8 py-4 text-[var(--color-text-primary)] font-bold text-lg transition-all duration-400 cursor-pointer shadow-sm"
        >
          <span>{isExpanded ? 'إخفاء الموديولات' : 'عرض المزيد'}</span>
          <svg
            className={`w-5 h-5 transition-transform duration-400 ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2.5}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>
      </div>

      {/* Collapsible Grid */}
      <div
        className={`
          grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5
          transition-all duration-700 ease-in-out overflow-hidden
          ${isExpanded ? 'max-h-[1200px] opacity-100 py-2' : 'max-h-0 opacity-0'}
        `}
      >
        {advancedModules.map((module) => (
          <div key={module.id} className="group">
            <ServiceCard
              title={module.title}
              icon={module.icon}
              description={module.description}
              link={module.link}
              isActive={module.isActive}
              size="compact"
            />
          </div>
        ))}
      </div>

      {/* Collapsed preview hint */}
      {!isExpanded && (
        <div className="flex justify-center gap-3 mt-2">
          {advancedModules.slice(0, 4).map((m) => (
            <div
              key={m.id}
              className="w-12 h-12 rounded-xl bg-white border border-[var(--color-border)] flex items-center justify-center text-xl opacity-60 shadow-sm"
              title={m.title}
            >
              {m.icon}
            </div>
          ))}
          <div className="w-12 h-12 rounded-xl bg-white border border-[var(--color-border)] flex items-center justify-center text-[var(--color-text-muted)] text-sm font-bold opacity-60 shadow-sm">
            +{advancedModules.length - 4}
          </div>
        </div>
      )}
    </section>
  );
}
