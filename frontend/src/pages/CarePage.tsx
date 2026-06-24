/**
 * CarePage — الرعاية الصحية القريبة منك.
 * Gère 4 états : initial → loading → results/error.
 * Intègre Geolocation API + localStorage pour la dernière position.
 */
import { useState, useCallback, useEffect } from 'react';
import InitialState from './care/InitialState';
import LoadingState from './care/LoadingState';
import ResultsState from './care/ResultsState';
import ErrorState from './care/ErrorState';
import { generateMockFacilities, type Facility } from './care/careData';

type PageState = 'initial' | 'loading' | 'results' | 'error';
type ErrorType = 'denied' | 'unavailable' | 'timeout' | 'no_results';

const STORAGE_KEY = 'shifa_last_position';

function getSavedPosition(): { lat: number; lng: number } | null {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch { return null; }
}

function savePosition(lat: number, lng: number) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ lat, lng })); } catch {}
}

export default function CarePage() {
  const [state, setState] = useState<PageState>('initial');
  const [errorType, setErrorType] = useState<ErrorType>('unavailable');
  const [userLat, setUserLat] = useState(0);
  const [userLng, setUserLng] = useState(0);
  const [facilities, setFacilities] = useState<Facility[]>([]);

  // Check for saved position on mount
  useEffect(() => {
    const saved = getSavedPosition();
    if (saved) {
      setUserLat(saved.lat);
      setUserLng(saved.lng);
      setFacilities(generateMockFacilities(saved.lat, saved.lng));
      setState('results');
    }
  }, []);

  const handleGPSRequest = useCallback(() => {
    setState('loading');

    if (!navigator.geolocation) {
      setErrorType('unavailable');
      setState('error');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        setUserLat(latitude);
        setUserLng(longitude);
        savePosition(latitude, longitude);
        const data = generateMockFacilities(latitude, longitude);
        setFacilities(data);
        setState('results');
      },
      (err) => {
        switch (err.code) {
          case err.PERMISSION_DENIED:
            setErrorType('denied'); break;
          case err.POSITION_UNAVAILABLE:
            setErrorType('unavailable'); break;
          case err.TIMEOUT:
            setErrorType('timeout'); break;
          default:
            setErrorType('unavailable');
        }
        setState('error');
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 }
    );
  }, []);

  const handleManualSearch = useCallback((query: string) => {
    setState('loading');
    // Simulate geocoding delay (in prod, call Nominatim or Google Geocoding)
    setTimeout(() => {
      // Default to Marrakech center for demo
      const lat = 31.6295 + (Math.random() - 0.5) * 0.02;
      const lng = -7.9811 + (Math.random() - 0.5) * 0.02;
      setUserLat(lat);
      setUserLng(lng);
      savePosition(lat, lng);
      setFacilities(generateMockFacilities(lat, lng));
      setState('results');
    }, 1500);
  }, []);

  const handleRetry = useCallback(() => {
    setState('initial');
  }, []);

  return (
    <div className="h-full">
      {state === 'initial' && (
        <InitialState onRequestGPS={handleGPSRequest} onManualSearch={handleManualSearch} />
      )}
      {state === 'loading' && <LoadingState />}
      {state === 'results' && (
        <ResultsState lat={userLat} lng={userLng} facilities={facilities} />
      )}
      {state === 'error' && (
        <ErrorState errorType={errorType} onRetry={handleRetry} onManualSearch={handleManualSearch} />
      )}
    </div>
  );
}
