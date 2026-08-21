import { Activity, AlertTriangle, CheckCircle2, WifiOff } from 'lucide-react';

const STATUS_STYLES = {
  online: {
    icon: CheckCircle2,
    label: 'ONLINE',
    className: 'status-online',
  },
  offline: {
    icon: WifiOff,
    label: 'OFFLINE',
    className: 'status-offline',
  },
  warning: {
    icon: AlertTriangle,
    label: 'WARNING',
    className: 'status-warning',
  },
};

export default function StatusIndicator({ status = 'online', label, compact = false }) {
  const config = STATUS_STYLES[status] || STATUS_STYLES.online;
  const Icon = config.icon;

  return (
    <div className={`status-indicator ${compact ? 'compact' : ''}`}>
      <span className={`status-dot ${config.className}`}>
        <Icon size={compact ? 12 : 14} />
      </span>
      <span>{label || config.label}</span>
    </div>
  );
}
