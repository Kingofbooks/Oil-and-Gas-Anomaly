import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getAlerts } from '../api/alerts';
import { getAnomalies } from '../api/anomalies';
import { getReadingsByWell } from '../api/wells';
import LoadingState from '../components/LoadingState';

const features = [
  'ABER_CKGL',
  'ABER_CKP',
  'ESTADO_DHSV',
  'ESTADO_M1',
  'ESTADO_M2',
  'ESTADO_PXO',
  'ESTADO_SDV_GL',
  'ESTADO_SDV_P',
  'ESTADO_W1',
  'ESTADO_W2',
  'ESTADO_XO',
  'P_ANULAR',
  'P_JUS_CKGL',
  'P_JUS_CKP',
  'P_MON_CKP',
  'P_PDG',
  'P_TPT',
  'QGL',
  'T_JUS_CKP',
  'T_MON_CKP',
  'T_PDG',
  'T_TPT',
];

export default function WellDetails() {
  const { wellId } = useParams();
  const [readings, setReadings] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const [readingsData, alertData, anomalyData] = await Promise.all([
          getReadingsByWell(wellId, 20),
          getAlerts({ well_id: wellId, limit: 10 }),
          getAnomalies({ well_id: wellId, limit: 10 }),
        ]);

        setReadings(Array.isArray(readingsData) ? readingsData : []);
        setAlerts(Array.isArray(alertData) ? alertData : []);
        setAnomalies(Array.isArray(anomalyData) ? anomalyData : []);
      } catch (err) {
        setError(err.message || 'Unable to load well detail data.');
      } finally {
        setLoading(false);
      }
    };

    if (wellId) load();
  }, [wellId]);

  if (loading) return <LoadingState label={`Loading ${wellId || 'well'} details...`} />;
  if (error) return <div className="panel error-panel">{error}</div>;

  const latestReading = readings[0] || null;

  return (
    <div className="page-shell">
      <div className="page-header-row">
        <div>
          <p className="page-kicker">Well detail</p>
          <h2>{wellId}</h2>
        </div>
      </div>

      <div className="well-summary-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Status</p>
              <h3>Current well health</h3>
            </div>
          </div>
          <div className="well-summary-box">
            <span className="well-summary-box__label">Current status</span>
            <strong>{anomalies.some((a) => a.is_anomaly) ? 'ANOMALY DETECTED' : 'NORMAL'}</strong>
            <small>{alerts.length ? `${alerts.filter((a) => a.status === 'OPEN').length} open alert(s)` : 'No active alerts'}</small>
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Latest reading</p>
              <h3>Telemetry snapshot</h3>
            </div>
          </div>
          {latestReading ? (
            <div className="sensor-metrics">
              {features.slice(0, 10).map((feature) => (
                <div key={feature} className="sensor-metric">
                  <label>{feature}</label>
                  <strong>{Number(latestReading[feature] ?? 0).toFixed(3)}</strong>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">No sensor readings available for this well.</div>
          )}
        </div>
      </div>

      <div className="split-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Timeline</p>
              <h3>Recent readings</h3>
            </div>
          </div>
          {readings.length === 0 ? (
            <div className="empty-state">No readings for this well.</div>
          ) : (
            <div className="table-wrap compact-table">
              <table>
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Source</th>
                    <th>QGL</th>
                    <th>P_TPT</th>
                  </tr>
                </thead>
                <tbody>
                  {readings.map((reading) => (
                    <tr key={reading.id}>
                      <td>{new Date(reading.timestamp).toLocaleString()}</td>
                      <td>{reading.source}</td>
                      <td>{Number(reading.QGL ?? 0).toFixed(3)}</td>
                      <td>{Number(reading.P_TPT ?? 0).toFixed(3)}</td>
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
              <p className="panel-kicker">Events</p>
              <h3>Recent anomalies</h3>
            </div>
          </div>
          {anomalies.length === 0 ? (
            <div className="empty-state">No anomaly history available.</div>
          ) : (
            <div className="stack-list">
              {anomalies.map((item) => (
                <div key={item.id} className="stack-item">
                  <strong>{new Date(item.timestamp).toLocaleString()}</strong>
                  <span className={`prediction-pill ${item.is_anomaly ? 'anomaly' : 'normal'}`}>
                    {item.is_anomaly ? 'ANOMALY' : 'NORMAL'}
                  </span>
                  <small>{Number(item.anomaly_score ?? 0).toFixed(4)}</small>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <p className="panel-kicker">Notifications</p>
            <h3>Recent alerts</h3>
          </div>
        </div>
        {alerts.length === 0 ? (
          <div className="empty-state">No alerts for this well.</div>
        ) : (
          <div className="alert-stack">
            {alerts.map((alert) => (
              <div key={alert.id} className="alert-mini">
                <div className="alert-mini__head">
                  <span className={`alert-status ${alert.status === 'OPEN' ? 'open' : 'resolved'}`}>{alert.status}</span>
                  <span className="alert-severity">{alert.severity}</span>
                </div>
                <p>{alert.message}</p>
                <small>{new Date(alert.created_at).toLocaleString()}</small>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
