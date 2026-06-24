import ServiceCard from './ServiceCard';
import { quickServices } from '../data/services';

/**
 * QuickServices — الخدمات الأساسية (3 بطاقات).
 * Light Medical Theme — Blue section badge, dark primary text titles.
 */
export default function QuickServices() {
  return (
    <section id="quick-services" className="section-gap max-w-6xl mx-auto px-6 sm:px-8">
      {/* Section Header */}
      <div className="text-center mb-10 animate-float-up">
        <div className="inline-flex items-center gap-2 bg-[var(--color-primary)]/8 text-[var(--color-primary)] text-sm font-bold px-4 py-1.5 rounded-full border border-[var(--color-primary)]/15 mb-4">
          <span>🏥</span>
          <span>الأكثر استخداماً</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-[var(--color-text-primary)] mb-3">
          الخدمات الأساسية
        </h2>
        <p className="text-[var(--color-text-secondary)] text-lg max-w-xl mx-auto">
          ابدأ باختيار الخدمة المناسبة لحالتك الصحية
        </p>
      </div>

      {/* 3 Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {quickServices.map((service, index) => (
          <div key={service.id} className={`animate-float-up-delay-${index + 1} group`}>
            <ServiceCard
              title={service.title}
              icon={service.icon}
              description={service.description}
              link={service.link}
              isActive={service.isActive}
              gradient={service.gradient}
              size="default"
            />
          </div>
        ))}
      </div>
    </section>
  );
}
