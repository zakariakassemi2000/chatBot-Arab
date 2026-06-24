import type { NavLink, UserInfo } from '../types/navigation';

/**
 * SHIFA AI — Navigation Configuration
 * Centralized navigation links and route-to-breadcrumb mapping.
 */

/** 5 liens principaux de la sidebar */
export const mainNavLinks: NavLink[] = [
  {
    id: 'home',
    label: 'الرئيسية',
    icon: '🏠',
    path: '/',
    isActive: true,
  },
  {
    id: 'checkup',
    label: 'فحص مبدئي',
    icon: '🩺',
    path: '/checkup',
    isActive: true,
  },
  {
    id: 'assistant',
    label: 'المساعد الطبي',
    icon: '💬',
    path: '/assistant',
    isActive: true,
  },
  {
    id: 'care',
    label: 'الرعاية',
    icon: '🏥',
    path: '/care',
    isActive: true,
  },
  {
    id: 'mental',
    label: 'الصحة النفسية',
    icon: '🧠',
    path: '/mental',
    isActive: true,
  },
];

/** Mapping route → breadcrumb */
export const breadcrumbMap: Record<string, { label: string; icon: string }> = {
  '/': { label: 'الرئيسية', icon: '🏠' },
  '/checkup': { label: 'فحص مبدئي', icon: '🩺' },
  '/assistant': { label: 'المساعد الطبي', icon: '💬' },
  '/care': { label: 'الرعاية', icon: '🏥' },
  '/mental': { label: 'الصحة النفسية', icon: '🧠' },
};

/** Utilisateur invité par défaut */
export const defaultGuestUser: UserInfo = {
  username: 'زائر',
  fullName: 'مستخدم زائر',
  role: 'guest',
  isAuthenticated: false,
};
