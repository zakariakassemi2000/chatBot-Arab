/**
 * SHIFA AI — Navigation Type Definitions
 * TypeScript interfaces for the ShifaShell navigation system.
 */

/** Statut d'authentification utilisateur */
export type UserRole = 'guest' | 'user' | 'doctor';

/** Informations utilisateur */
export interface UserInfo {
  username: string;
  fullName: string;
  role: UserRole;
  isAuthenticated: boolean;
}

/** Définition d'un lien de navigation */
export interface NavLink {
  id: string;
  label: string;
  icon: string;
  path: string;
  isActive: boolean;
}

/** Props du sidebar */
export interface SidebarProps {
  user: UserInfo;
  navLinks: NavLink[];
  currentPath: string;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  onLogout: () => void;
  onNavigate: (path: string) => void;
}

/** Props du topbar */
export interface TopBarProps {
  breadcrumb: string;
  breadcrumbIcon: string;
  sidebarWidth: number;
  onMenuToggle: () => void;
  isMobileMenuOpen: boolean;
}

/** Props d'un item de navigation */
export interface NavItemProps {
  link: NavLink;
  isCurrentPage: boolean;
  isCollapsed: boolean;
  onClick: () => void;
}

/** Props du BottomNav mobile */
export interface BottomNavProps {
  navLinks: NavLink[];
  currentPath: string;
  onNavigate: (path: string) => void;
}

/** État du contexte de navigation */
export interface NavigationState {
  currentPath: string;
  breadcrumb: string;
  breadcrumbIcon: string;
  user: UserInfo;
  isSidebarCollapsed: boolean;
  isMobileMenuOpen: boolean;
}

/** Actions du contexte de navigation */
export interface NavigationContextType extends NavigationState {
  navigate: (path: string) => void;
  toggleSidebar: () => void;
  toggleMobileMenu: () => void;
  logout: () => void;
  setBreadcrumb: (label: string, icon: string) => void;
}
