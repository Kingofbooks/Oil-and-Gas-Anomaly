import { Activity, BellRing, Gauge, ShieldAlert } from 'lucide-react';
import { NavLink } from 'react-router-dom';

const items = [
  { to: '/', label: 'Dashboard', icon: Gauge },
  { to: '/wells', label: 'Wells', icon: Activity },
  { to: '/anomalies', label: 'Anomalies', icon: ShieldAlert },
  { to: '/alerts', label: 'Alerts', icon: BellRing },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      {items.map(({ to, label, icon: Icon }) => (
        <NavLink key={to} to={to} className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <Icon size={16} />
          <span>{label}</span>
        </NavLink>
      ))}
    </aside>
  );
}
