import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from 'react';
import { useLocation, useNavigate as useRouterNavigate } from 'react-router-dom';
import type { NavigationContextType, UserInfo } from '../types/navigation';
import { breadcrumbMap, defaultGuestUser } from '../data/navigation';

/**
 * NavigationContext — Context API pour l'état de navigation global.
 * Gère : page courante, breadcrumb, user, sidebar collapse, mobile menu.
 */

const NavigationContext = createContext<NavigationContextType | null>(null);

export function NavigationProvider({ children }: { children: ReactNode }) {
  const location = useLocation();
  const routerNavigate = useRouterNavigate();

  const [user, setUser] = useState<UserInfo>(defaultGuestUser);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [breadcrumb, setBreadcrumbState] = useState('الرئيسية');
  const [breadcrumbIcon, setBreadcrumbIconState] = useState('🏠');

  // Update breadcrumb on route change
  useEffect(() => {
    const entry = breadcrumbMap[location.pathname];
    if (entry) {
      setBreadcrumbState(entry.label);
      setBreadcrumbIconState(entry.icon);
    }
  }, [location.pathname]);

  // Close mobile menu on route change
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location.pathname]);

  // Handle responsive sidebar collapse
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setIsSidebarCollapsed(true);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const navigate = useCallback((path: string) => {
    routerNavigate(path);
  }, [routerNavigate]);

  const toggleSidebar = useCallback(() => {
    setIsSidebarCollapsed((prev) => !prev);
  }, []);

  const toggleMobileMenu = useCallback(() => {
    setIsMobileMenuOpen((prev) => !prev);
  }, []);

  const logout = useCallback(() => {
    setUser(defaultGuestUser);
    routerNavigate('/');
  }, [routerNavigate]);

  const setBreadcrumb = useCallback((label: string, icon: string) => {
    setBreadcrumbState(label);
    setBreadcrumbIconState(icon);
  }, []);

  const value: NavigationContextType = {
    currentPath: location.pathname,
    breadcrumb,
    breadcrumbIcon,
    user,
    isSidebarCollapsed,
    isMobileMenuOpen,
    navigate,
    toggleSidebar,
    toggleMobileMenu,
    logout,
    setBreadcrumb,
  };

  return (
    <NavigationContext.Provider value={value}>
      {children}
    </NavigationContext.Provider>
  );
}

/** Hook pour consommer le NavigationContext */
export function useNavigation(): NavigationContextType {
  const ctx = useContext(NavigationContext);
  if (!ctx) {
    throw new Error('useNavigation must be used within a NavigationProvider');
  }
  return ctx;
}
