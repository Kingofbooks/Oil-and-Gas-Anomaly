import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import './App.css';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import AlertsPage from './pages/Alerts';
import Anomalies from './pages/Anomalies';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import WellDetails from './pages/WellDetails';
import Wells from './pages/Wells';

function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar apiOnline={true} modelOnline={true} />

        <div className="content-shell">
          <Sidebar />

          <main className="main-panel">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/wells" element={<Wells />} />
              <Route path="/wells/:wellId" element={<WellDetails />} />
              <Route path="/anomalies" element={<Anomalies />} />
              <Route path="/alerts" element={<AlertsPage />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
