import { useState, useEffect, type ReactNode } from 'react';
import { useNavigation } from '../context/NavigationContext';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import BottomNav from './BottomNav';
import MobileDrawer from './MobileDrawer';

/**
 * ShifaShell — Layout universel parent.
 * Structure : TopBar fixe (64px) + Sidebar droite fixe (280px) + Zone contenu centrale.
 * Sur TOUS les écrans, sidebar et topbar sont IDENTIQUES — seul le contenu central change.
 *
 * Responsive :
 * - Desktop (>1024px)  : sidebar expanded (280px) + topbar
 * - Tablette (768-1024) : sidebar collapsed (72px icons only) + topbar
 * - Mobile (<768px)     : no sidebar, bottom nav bar + mobile drawer
 */

interface ShifaShellProps {
  children: ReactNode;
}

function useWindowWidth() {
  const [width, setWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1200);
  useEffect(() => {
    const handle = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handle);
    return () => window.removeEventListener('resize', handle);
  }, []);
  return width;
}

export default function ShifaShell({ children }: ShifaShellProps) {
  const { isSidebarCollapsed } = useNavigation();
  const windowWidth = useWindowWidth();
  const isDesktop = windowWidth >= 768;

  // Sidebar width controls content offset
  const sidebarWidth = isSidebarCollapsed ? 72 : 280;

  return (
    <div className="min-h-screen" dir="rtl">
      {/* ── Fixed Sidebar (desktop/tablet) ── */}
      <Sidebar />

      {/* ── Fixed TopBar ── */}
      <TopBar />

      {/* ── Mobile Drawer (overlay) ── */}
      <MobileDrawer />

      {/* ── Main Content Area ── */}
      <main
        id="shifa-content"
        className="
          pt-16 min-h-screen
          transition-all duration-300
          pb-20 md:pb-0
        "
        style={{
          paddingInlineStart: isDesktop ? `${sidebarWidth}px` : '0',
        }}
      >
        {/* Page transition wrapper */}
        <div className="page-transition">
          {children}
        </div>
      </main>

      {/* ── Mobile Bottom Nav ── */}
      <BottomNav />
    </div>
  );
}
