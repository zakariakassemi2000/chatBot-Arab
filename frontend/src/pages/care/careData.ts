/**
 * Types & mock data for nearby care facilities.
 */

export type FacilityType = 'hospital' | 'clinic' | 'pharmacy' | 'doctor';

export interface Facility {
  id: string;
  name: string;
  nameAr: string;
  type: FacilityType;
  lat: number;
  lng: number;
  distance: number;        // km
  address: string;
  phone: string;
  isOpenNow: boolean;
  hours: string;
  rating: number;          // 1-5
}

export interface Filters {
  type: FacilityType | 'all';
  maxDistance: number;      // km
  openNow: boolean;
}

export const facilityMeta: Record<FacilityType, { label: string; icon: string; color: string; markerColor: string }> = {
  hospital:  { label: 'مستشفى',   icon: '🏥', color: '#ef4444', markerColor: '#ef4444' },
  clinic:    { label: 'عيادة',    icon: '🩺', color: '#f97316', markerColor: '#f97316' },
  pharmacy:  { label: 'صيدلية',   icon: '💊', color: '#22c55e', markerColor: '#22c55e' },
  doctor:    { label: 'طبيب',     icon: '👨‍⚕️', color: '#3b82f6', markerColor: '#3b82f6' },
};

/**
 * Generate realistic mock facilities around a given center point.
 */
export function generateMockFacilities(centerLat: number, centerLng: number): Facility[] {
  const facilities: Facility[] = [
    {
      id: 'h1', name: 'CHU Mohammed VI', nameAr: 'المركز الاستشفائي الجامعي محمد السادس',
      type: 'hospital', lat: centerLat + 0.008, lng: centerLng - 0.005,
      distance: 0.9, address: 'شارع عبد الكريم الخطابي، مراكش',
      phone: '+212524431818', isOpenNow: true, hours: '24/7', rating: 4.2,
    },
    {
      id: 'h2', name: 'Hôpital Ibn Tofail', nameAr: 'مستشفى ابن طفيل',
      type: 'hospital', lat: centerLat - 0.012, lng: centerLng + 0.008,
      distance: 1.6, address: 'حي المحمدي، مراكش',
      phone: '+212524338746', isOpenNow: true, hours: '24/7', rating: 3.8,
    },
    {
      id: 'c1', name: 'Clinique Al Kawtar', nameAr: 'عيادة الكوثر',
      type: 'clinic', lat: centerLat + 0.003, lng: centerLng + 0.006,
      distance: 0.5, address: 'شارع الزرقطوني، جيليز',
      phone: '+212524436789', isOpenNow: true, hours: '08:00 - 20:00', rating: 4.5,
    },
    {
      id: 'c2', name: 'Clinique Internationale', nameAr: 'العيادة الدولية',
      type: 'clinic', lat: centerLat - 0.005, lng: centerLng - 0.009,
      distance: 1.1, address: 'شارع محمد الخامس، المدينة',
      phone: '+212524445566', isOpenNow: false, hours: '09:00 - 18:00', rating: 4.0,
    },
    {
      id: 'p1', name: 'Pharmacie du Centre', nameAr: 'صيدلية المركز',
      type: 'pharmacy', lat: centerLat + 0.001, lng: centerLng + 0.002,
      distance: 0.2, address: 'شارع الحرية، جيليز',
      phone: '+212524321000', isOpenNow: true, hours: '08:00 - 23:00', rating: 4.6,
    },
    {
      id: 'p2', name: 'Pharmacie de Nuit', nameAr: 'صيدلية المناوبة الليلية',
      type: 'pharmacy', lat: centerLat - 0.003, lng: centerLng + 0.004,
      distance: 0.4, address: 'ساحة جامع الفنا',
      phone: '+212524387654', isOpenNow: true, hours: '24/7', rating: 4.3,
    },
    {
      id: 'p3', name: 'Pharmacie Atlas', nameAr: 'صيدلية الأطلس',
      type: 'pharmacy', lat: centerLat + 0.006, lng: centerLng - 0.003,
      distance: 0.7, address: 'حي الداوديات',
      phone: '+212524567890', isOpenNow: false, hours: '08:30 - 21:00', rating: 3.9,
    },
    {
      id: 'd1', name: 'Dr. Hassan Alaoui', nameAr: 'د. حسن العلوي — طب عام',
      type: 'doctor', lat: centerLat + 0.004, lng: centerLng - 0.001,
      distance: 0.3, address: 'عمارة البدر، الطابق 3، جيليز',
      phone: '+212661234567', isOpenNow: true, hours: '09:00 - 17:00', rating: 4.7,
    },
    {
      id: 'd2', name: 'Dr. Fatima Zahra Bennani', nameAr: 'د. فاطمة الزهراء بناني — طب أطفال',
      type: 'doctor', lat: centerLat - 0.007, lng: centerLng - 0.006,
      distance: 1.0, address: 'شارع المسيرة الخضراء',
      phone: '+212662345678', isOpenNow: true, hours: '08:00 - 16:00', rating: 4.8,
    },
    {
      id: 'd3', name: 'Dr. Ahmed Tazi', nameAr: 'د. أحمد التازي — أمراض القلب',
      type: 'doctor', lat: centerLat + 0.010, lng: centerLng + 0.010,
      distance: 1.8, address: 'حي السعادة، مراكش',
      phone: '+212663456789', isOpenNow: false, hours: '10:00 - 15:00', rating: 4.4,
    },
  ];

  return facilities;
}
