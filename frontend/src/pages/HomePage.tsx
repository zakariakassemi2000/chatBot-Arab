import EmergencyBanner from '../components/EmergencyBanner';
import HeroSection from '../components/HeroSection';
import QuickServices from '../components/QuickServices';
import AdvancedModules from '../components/AdvancedModules';
import Footer from '../components/Footer';

/**
 * HomePage — Page d'accueil (الرئيسية).
 * Réutilise les composants existants du sprint précédent.
 */
export default function HomePage() {
  return (
    <>
      <EmergencyBanner />
      <HeroSection />
      <QuickServices />
      <AdvancedModules />
      <Footer />
    </>
  );
}
