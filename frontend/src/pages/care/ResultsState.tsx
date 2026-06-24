/**
 * ResultsState — الحالة 3: الخريطة التفاعلية + قائمة المرافق.
 * Light Medical Theme — Light tile layer, white panel, blue accents.
 */
import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import type { Facility, Filters, FacilityType } from './careData';
import { facilityMeta } from './careData';

interface ResultsStateProps {
  lat: number;
  lng: number;
  facilities: Facility[];
}

const filterOptions: { value: FacilityType | 'all'; label: string }[] = [
  { value: 'all', label: 'الكل' },
  { value: 'hospital', label: '🏥 مستشفى' },
  { value: 'clinic', label: '🩺 عيادة' },
  { value: 'pharmacy', label: '💊 صيدلية' },
  { value: 'doctor', label: '👨‍⚕️ طبيب' },
];

function createMarkerIcon(color: string): L.DivIcon {
  return L.divIcon({
    className: '',
    html: `<div style="width:28px;height:28px;border-radius:50%;background:${color};border:3px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

function createUserIcon(): L.DivIcon {
  return L.divIcon({
    className: '',
    html: `<div style="width:18px;height:18px;border-radius:50%;background:#2563eb;border:3px solid white;box-shadow:0 0 0 6px rgba(37,99,235,0.2),0 2px 8px rgba(0,0,0,0.15);"></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
}

export default function ResultsState({ lat, lng, facilities }: ResultsStateProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<L.Map | null>(null);
  const markersRef = useRef<L.LayerGroup | null>(null);

  const [filters, setFilters] = useState<Filters>({ type: 'all', maxDistance: 5, openNow: false });
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const filtered = facilities.filter(f => {
    if (filters.type !== 'all' && f.type !== filters.type) return false;
    if (filters.openNow && !f.isOpenNow) return false;
    if (f.distance > filters.maxDistance) return false;
    return true;
  }).sort((a, b) => a.distance - b.distance);

  // Init map
  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;
    const map = L.map(mapRef.current, { zoomControl: false }).setView([lat, lng], 14);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CARTO',
    }).addTo(map);
    L.control.zoom({ position: 'bottomleft' }).addTo(map);
    L.marker([lat, lng], { icon: createUserIcon() }).addTo(map).bindPopup('📍 موقعك');
    markersRef.current = L.layerGroup().addTo(map);
    mapInstance.current = map;
    return () => { map.remove(); mapInstance.current = null; };
  }, [lat, lng]);

  // Update markers
  useEffect(() => {
    if (!markersRef.current) return;
    markersRef.current.clearLayers();
    filtered.forEach(f => {
      const meta = facilityMeta[f.type];
      const marker = L.marker([f.lat, f.lng], { icon: createMarkerIcon(meta.markerColor) });
      marker.bindPopup(`<b>${f.nameAr}</b><br/>${meta.label} · ${f.distance} كم`);
      marker.on('click', () => setSelectedId(f.id));
      markersRef.current!.addLayer(marker);
    });
  }, [filtered]);

  // Pan to selected
  useEffect(() => {
    if (!selectedId || !mapInstance.current) return;
    const f = facilities.find(x => x.id === selectedId);
    if (f) mapInstance.current.flyTo([f.lat, f.lng], 16, { duration: 0.5 });
  }, [selectedId, facilities]);

  return (
    <div className="flex flex-col lg:flex-row gap-0 h-[calc(100vh-80px)] animate-float-up">
      {/* Map */}
      <div className="flex-1 relative min-h-[300px]">
        <div ref={mapRef} className="absolute inset-0" id="care-map" />
      </div>

      {/* Results panel */}
      <div className="w-full lg:w-96 bg-white/95 backdrop-blur-xl border-r border-[var(--color-border)] flex flex-col max-h-[calc(100vh-80px)] shadow-sm">
        {/* Filters */}
        <div className="p-3 border-b border-[var(--color-border)] space-y-2">
          {/* Type pills */}
          <div className="flex gap-1.5 overflow-x-auto pb-1 hide-scrollbar">
            {filterOptions.map(opt => (
              <button key={opt.value} onClick={() => setFilters(p => ({...p, type: opt.value}))}
                className={`flex-shrink-0 px-3 py-1.5 rounded-xl text-xs font-bold cursor-pointer transition-all
                  ${filters.type === opt.value
                    ? 'bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20'
                    : 'bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] border border-transparent hover:bg-slate-200'}`}>
                {opt.label}
              </button>
            ))}
          </div>
          {/* Open now toggle */}
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={filters.openNow} onChange={e => setFilters(p => ({...p, openNow: e.target.checked}))}
                className="w-4 h-4 accent-[var(--color-primary)] rounded cursor-pointer" />
              <span className="text-xs text-[var(--color-text-secondary)]">مفتوح الآن فقط</span>
            </label>
            <span className="text-xs text-[var(--color-text-muted)]">{filtered.length} نتيجة</span>
          </div>
        </div>

        {/* Facility list */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {filtered.length === 0 ? (
            <div className="text-center py-10 text-[var(--color-text-muted)]">
              <p className="text-3xl mb-2">🔍</p>
              <p className="text-sm">لا توجد نتائج تطابق المعايير</p>
            </div>
          ) : filtered.map(f => {
            const meta = facilityMeta[f.type];
            const isSelected = selectedId === f.id;
            return (
              <button key={f.id} onClick={() => setSelectedId(f.id)}
                className={`w-full text-right rounded-2xl p-3.5 transition-all duration-200 cursor-pointer border
                  ${isSelected
                    ? 'bg-[var(--color-primary)]/5 border-[var(--color-primary)]/20 shadow-sm'
                    : 'bg-white border-[var(--color-border)] hover:border-[var(--color-primary)]/20 hover:shadow-sm'}`}>
                <div className="flex items-start gap-3">
                  {/* Icon */}
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center text-xl flex-shrink-0" style={{background: `${meta.color}12`}}>
                    {meta.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-[var(--color-text-primary)] font-bold text-sm truncate">{f.nameAr}</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-[var(--color-text-muted)] text-xs">{meta.label}</span>
                      <span className="text-[var(--color-text-muted)] text-xs">·</span>
                      <span className="text-[var(--color-primary)] text-xs font-bold">{f.distance} كم</span>
                      {f.isOpenNow ? (
                        <span className="flex items-center gap-1 text-emerald-600 text-[10px]">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />مفتوح
                        </span>
                      ) : (
                        <span className="text-red-500 text-[10px]">مغلق</span>
                      )}
                    </div>
                    <p className="text-[var(--color-text-muted)] text-[11px] mt-1 truncate">{f.hours} · {f.address}</p>
                  </div>
                </div>
                {/* Actions */}
                <div className="flex gap-2 mt-2.5">
                  <a href={`https://www.google.com/maps/dir/?api=1&destination=${f.lat},${f.lng}`} target="_blank" rel="noopener"
                    onClick={e => e.stopPropagation()}
                    className="flex-1 flex items-center justify-center gap-1.5 bg-[var(--color-primary)]/8 hover:bg-[var(--color-primary)]/15 text-[var(--color-primary)] text-xs font-bold py-2 rounded-xl transition-colors">
                    <span>🧭</span><span>اتجاهات</span>
                  </a>
                  <a href={`tel:${f.phone}`} onClick={e => e.stopPropagation()}
                    className="flex-1 flex items-center justify-center gap-1.5 bg-sky-50 hover:bg-sky-100 text-sky-700 text-xs font-bold py-2 rounded-xl transition-colors">
                    <span>📞</span><span>اتصال</span>
                  </a>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
