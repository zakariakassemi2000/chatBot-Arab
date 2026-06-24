import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { NavigationProvider } from './context/NavigationContext';
import ShifaShell from './layout/ShifaShell';
import HomePage from './pages/HomePage';
import CheckupPage from './pages/CheckupPage';
import AssistantPage from './pages/AssistantPage';
import CarePage from './pages/CarePage';
import MentalPage from './pages/MentalPage';

/**
 * App — Root component avec React Router v6.
 * ShifaShell wraps ALL routes — sidebar & topbar sont identiques partout.
 * Seul le contenu central change selon la route.
 */
export default function App() {
  return (
    <BrowserRouter>
      <NavigationProvider>
        <ShifaShell>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/checkup" element={<CheckupPage />} />
            <Route path="/assistant" element={<AssistantPage />} />
            <Route path="/care" element={<CarePage />} />
            <Route path="/mental" element={<MentalPage />} />
          </Routes>
        </ShifaShell>
      </NavigationProvider>
    </BrowserRouter>
  );
}
