import { AlertTriangle, CheckCircle2, Clock3 } from 'lucide-react';

export default function AlertCard({ alert }) {
  const isOpen = alert.status === 'OPEN';
  const severity = String(alert.severity || 'INFO').toUpperCase();

  return (
    <div className={`alert-card alert-${severity.toLowerCase()}`}>
      <div className="alert-card__top">
        <div className="alert-card__severity">
          {isOpen ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
          <span>{severity}</span>
        </div>
        <span className={`alert-status ${isOpen ? 'open' : 'resolved'}`}>{alert.status}</span>
      </div>

      <div className="alert-card__body">
        <h4>{alert.well_id || alert.wellId || 'Unknown well'}</h4>
        <p>{alert.message || 'No message available.'}</p>
      </div>

      <div className="alert-card__meta">
        <span>
          <Clock3 size={12} />
          {alert.created_at ? new Date(alert.created_at).toLocaleString() : 'Unknown time'}
        </span>
      </div>
    </div>
  );
}
