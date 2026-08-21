import { Activity, BellRing, Gauge, ShieldAlert } from 'lucide-react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Dashboard', icon: Gauge },
  { to: '/wells', label: 'Wells', icon: Activity },
  { to: '/anomalies', label: 'Anomalies', icon: ShieldAlert },
  { to: '/alerts', label: 'Alerts', icon: BellRing },
];

export default function Navbar({ apiOnline, modelOnline }) {
  return (
    <header className="topbar">
      <div className="brand-block">
        <div className="brand-mark">OG</div>
        <div>
          <h1>OilGuard AI</h1>
          <p>AI-Powered Oil &amp; Gas Anomaly Detection</p>
        </div>
      </div>

      <nav className="main-nav">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="status-cluster">
        <div className={`status-pill ${apiOnline ? 'ok' : 'bad'}`}>
          <span className="dot" />
          {apiOnline ? 'API ONLINE' : 'API OFFLINE'}
        </div>
        <div className={`status-pill ${modelOnline ? 'ok' : 'bad'}`}>
          <span className="dot" />
          {modelOnline ? 'TranAD ONLINE' : 'TranAD OFFLINE'}
        </div>
      </div>
    </header>
  );
}
