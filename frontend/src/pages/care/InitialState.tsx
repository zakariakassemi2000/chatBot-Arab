/**
 * InitialState — الحالة الأولى: بدون موقع.
 * Light Medical Theme — Blue primary CTA, white cards.
 */

import { useState } from 'react';

interface InitialStateProps {
  onRequestGPS: () => void;
  onManualSearch: (query: string) => void;
}

export default function InitialState({ onRequestGPS, onManualSearch }: InitialStateProps) {
  const [showSearch, setShowSearch] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearchSubmit = () => {
    if (searchQuery.trim().length > 2) {
      onManualSearch(searchQuery.trim());
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] px-4 relative">
      {/* Background animated map illustration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-15">
        <svg viewBox="0 0 800 500" className="w-full h-full animate-map-float" fill="none">
          {/* Grid lines */}
          {Array.from({ length: 12 }).map((_, i) => (
            <line key={`h${i}`} x1="0" y1={i * 45} x2="800" y2={i * 45}
              stroke="var(--color-primary)" strokeWidth="0.5" opacity="0.2" />
          ))}
          {Array.from({ length: 18 }).map((_, i) => (
            <line key={`v${i}`} x1={i * 50} y1="0" x2={i * 50} y2="500"
              stroke="var(--color-primary)" strokeWidth="0.5" opacity="0.2" />
          ))}
          {/* Roads */}
          <path d="M0 250 Q200 200 400 250 T800 250" stroke="var(--color-text-muted)" strokeWidth="3" opacity="0.3" />
          <path d="M400 0 Q380 150 400 250 Q420 350 400 500" stroke="var(--color-text-muted)" strokeWidth="3" opacity="0.3" />
          <path d="M100 0 Q150 200 200 500" stroke="var(--color-text-muted)" strokeWidth="2" opacity="0.15" />
          <path d="M600 0 Q650 250 700 500" stroke="var(--color-text-muted)" strokeWidth="2" opacity="0.15" />
          {/* Location markers */}
          <circle cx="400" cy="250" r="8" fill="var(--color-primary)" opacity="0.4">
            <animate attributeName="r" values="6;12;6" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.4;0.15;0.4" dur="2s" repeatCount="indefinite" />
          </circle>
          <circle cx="400" cy="250" r="4" fill="var(--color-primary)" />
          <circle cx="250" cy="180" r="5" fill="#ef4444" opacity="0.4" />
          <circle cx="550" cy="300" r="5" fill="#10b981" opacity="0.4" />
          <circle cx="320" cy="350" r="5" fill="#f59e0b" opacity="0.4" />
          <circle cx="500" cy="150" r="5" fill="#2563eb" opacity="0.4" />
        </svg>
      </div>

      {/* Content */}
      <div className="relative z-10 text-center max-w-md animate-float-up">
        {/* Icon */}
        <div className="mb-6">
          <div className="w-20 h-20 mx-auto rounded-full bg-[var(--color-primary)]/8 border border-[var(--color-primary)]/15 flex items-center justify-center">
            <svg className="w-10 h-10 text-[var(--color-primary)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
            </svg>
          </div>
        </div>

        {/* Title */}
        <h2 className="text-2xl sm:text-3xl font-extrabold text-[var(--color-text-primary)] mb-3">
          الرعاية الصحية القريبة منك
        </h2>
        <p className="text-[var(--color-text-secondary)] text-base mb-8 leading-relaxed">
          اكتشف المستشفيات والعيادات والصيدليات القريبة من موقعك الحالي
        </p>

        {/* Primary CTA — GPS */}
        <button
          onClick={onRequestGPS}
          className="
            w-full flex items-center justify-center gap-3
            bg-[var(--color-primary)] hover:bg-blue-700
            text-white font-bold text-lg
            px-8 py-4 rounded-2xl
            transition-all duration-300 cursor-pointer
            hover:scale-[1.02] active:scale-[0.98]
            shadow-lg shadow-blue-500/20
            hover:shadow-xl hover:shadow-blue-500/30
            mb-3
          "
        >
          {/* GPS pulse icon */}
          <div className="relative">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 2C12 2 12 2 12 2M12 22C12 22 12 22 12 22M2 12H4M20 12H22M12 2V4M12 20V22M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-white animate-ping" />
            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-white" />
          </div>
          <span>تحديد موقعي تلقائياً</span>
        </button>

        {/* Secondary — Manual search */}
        {!showSearch ? (
          <button
            onClick={() => setShowSearch(true)}
            className="
              w-full flex items-center justify-center gap-2
              bg-white hover:bg-[var(--color-bg-tertiary)]
              border border-[var(--color-border)] hover:border-[var(--color-primary)]/30
              text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] font-medium text-base
              px-6 py-3 rounded-2xl
              transition-all duration-300 cursor-pointer
              shadow-sm
            "
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
            </svg>
            <span>أدخل عنوانك يدوياً</span>
          </button>
        ) : (
          <div className="animate-float-up">
            <div className="flex items-center gap-2 bg-white border border-[var(--color-border)] rounded-2xl px-4 py-2 shadow-sm focus-within:border-[var(--color-primary)] focus-within:ring-1 focus-within:ring-[var(--color-primary)]/20">
              <svg className="w-5 h-5 text-[var(--color-text-muted)] flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
              </svg>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearchSubmit()}
                placeholder="مثال: مراكش، جيليز..."
                className="flex-1 bg-transparent text-[var(--color-text-primary)] text-sm py-2 outline-none placeholder:text-slate-400"
                autoFocus
                dir="rtl"
              />
              <button
                onClick={handleSearchSubmit}
                disabled={searchQuery.trim().length < 3}
                className="bg-[var(--color-primary)]/10 hover:bg-[var(--color-primary)]/20 text-[var(--color-primary)] text-sm font-bold px-4 py-1.5 rounded-xl transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
              >
                بحث
              </button>
            </div>
          </div>
        )}

        {/* Privacy message */}
        <div className="mt-6 flex items-center justify-center gap-2 text-[var(--color-text-muted)] text-xs">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
          </svg>
          <span>موقعك محمي ولا يتم مشاركته مع أي طرف ثالث</span>
        </div>
      </div>
    </div>
  );
}
