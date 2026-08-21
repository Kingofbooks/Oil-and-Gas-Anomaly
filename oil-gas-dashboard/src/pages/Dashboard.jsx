import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Activity, BellRing, Database, Gauge, ShieldAlert, Waves } from 'lucide-react';
import { getAlerts } from '../api/alerts';
import { getAnomalies, getLatestAnomalies } from '../api/anomalies';
import { getDashboardSummary, getHealth } from '../api/dashboard';
import { getWells } from '../api/wells';
import AlertCard from '../components/AlertCard';
import AnomalyChart from '../components/AnomalyChart';
import LoadingState from '../components/LoadingState';
import StatCard from '../components/StatCard';
import StatusIndicator from '../components/StatusIndicator';

const REFRESH_MS = 5000;

const systemStatus = [
  { name: 'MQTT', status: 'warning', label: 'No direct health endpoint' },
  { name: 'PostgreSQL', status: 'online', label: 'Live via DB-backed API' },
  { name: 'FastAPI', status: 'online', label: 'Health endpoint active' },
  { name: 'TranAD', status: 'online', label: 'Model endpoint available' },
];

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [wellList, setWellList] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [recentAnomalies, setRecentAnomalies] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(true);

  const loadDashboard = async () => {
    try {
      const [health, dashboardSummary, wells, alertsData, anomalies] = await Promise.all([
        getHealth().catch(() => ({ status: 'offline' })),
        getDashboardSummary(),
        getWells(),
        getAlerts({ limit: 4 }),
        getLatestAnomalies(12),
      ]);

      setApiOnline(health?.status === 'ok' || health?.status === 'OK');
      setSummary(dashboardSummary);
      setWellList(Array.isArray(wells) ? wells : []);
      setAlerts(Array.isArray(alertsData) ? alertsData : []);
      setRecentAnomalies(Array.isArray(anomalies) ? anomalies : []);
    } catch (error) {
      setApiOnline(false);
      setSummary(null);
      setWellList([]);
      setAlerts([]);
      setRecentAnomalies([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDashboard();
    const interval = setInterval(loadDashboard, REFRESH_MS);
    return () => clearInterval(interval);
  }, []);

  const chartData = useMemo(
    () =>
      recentAnomalies
        .slice()
        .reverse()
        .map((item) => ({
          timestamp: item.timestamp,
          anomaly_score: Number(item.anomaly_score ?? 0),
        })),
    [recentAnomalies]
  );

  if (isLoading) {
    return <LoadingState label="Loading dashboard telemetry..." />;
  }

  return (
    <div className="page-shell">
      <div className="page-header-row">
        <div>
          <p className="page-kicker">Operations overview</p>
          <h2>Control room dashboard</h2>
        </div>
      </div>

      <div className="stat-grid">
        <StatCard title="Total Wells" value={summary?.total_wells ?? 0} hint="Active monitored wells" icon={Activity} tone="blue" />
        <StatCard title="Total Sensor Readings" value={summary?.total_readings ?? 0} hint="Captured data points" icon={Waves} tone="purple" />
        <StatCard title="Total Anomalies" value={summary?.total_anomalies ?? 0} hint="Detected by TranAD" icon={ShieldAlert} tone="red" />
        <StatCard title="Active Alerts" value={summary?.active_alerts ?? 0} hint="Open issues" icon={BellRing} tone="amber" />
      </div>

      <div className="top-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">System state</p>
              <h3>Runtime health</h3>
            </div>
          </div>

          <div className="status-list">
            {systemStatus.map((item) => (
              <div key={item.name} className="status-row">
                <div className="status-row__meta">
                  <span className="status-name">{item.name}</span>
                  <small>{item.label}</small>
                </div>
                <StatusIndicator status={item.status} label={item.status.toUpperCase()} compact />
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Coverage</p>
              <h3>Monitoring scope</h3>
            </div>
          </div>

          <div className="mini-metrics">
            <div className="mini-metric">
              <Database size={18} />
              <div>
                <strong>{wellList.length}</strong>
                <span>Wells tracked</span>
              </div>
            </div>
            <div className="mini-metric">
              <Gauge size={18} />
              <div>
                <strong>{recentAnomalies.length}</strong>
                <span>Recent results</span>
              </div>
            </div>
            <div className="mini-metric">
              <AlertTriangle size={18} />
              <div>
                <strong>{alerts.filter((alert) => alert.status === 'OPEN').length}</strong>
                <span>Open alerts</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <AnomalyChart data={chartData} />

      <div className="split-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Events</p>
              <h3>Recent anomalies</h3>
            </div>
          </div>

          {recentAnomalies.length === 0 ? (
            <div className="empty-state">No anomaly events available.</div>
          ) : (
            <div className="table-wrap compact-table">
              <table>
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Well</th>
                    <th>Score</th>
                    <th>Status</th>
                    <th>Model</th>
                    <th>Version</th>
                  </tr>
                </thead>
                <tbody>
                  {recentAnomalies.slice(0, 8).map((item) => (
                    <tr key={item.id}>
                      <td>{new Date(item.timestamp).toLocaleString()}</td>
                      <td>{'N/A'}</td>
                      <td>{Number(item.anomaly_score ?? 0).toFixed(4)}</td>
                      <td>
                        <span className={`prediction-pill ${item.is_anomaly ? 'anomaly' : 'normal'}`}>
                          {item.is_anomaly ? 'ANOMALY' : 'NORMAL'}
                        </span>
                      </td>
                      <td>{item.model_name || 'TranAD'}</td>
                      <td>{item.model_version || 'demo-v1'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Actionable issues</p>
              <h3>Active alerts</h3>
            </div>
          </div>

          <div className="alert-stack">
            {alerts.length === 0 ? (
              <div className="empty-state">No active alerts.</div>
            ) : (
              alerts.slice(0, 4).map((alert) => <AlertCard key={alert.id} alert={alert} />)
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
