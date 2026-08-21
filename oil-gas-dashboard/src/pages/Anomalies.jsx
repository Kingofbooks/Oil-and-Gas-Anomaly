import { useEffect, useMemo, useState } from 'react';
import { getAnomalies } from '../api/anomalies';
import LoadingState from '../components/LoadingState';

export default function Anomalies() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('all');
  const [sortDirection, setSortDirection] = useState('desc');

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getAnomalies({ limit: 200 });
        setItems(Array.isArray(data) ? data : []);
      } catch (err) {
        setError(err.message || 'Unable to load anomalies.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  const filteredData = useMemo(() => {
    const next = [...items].filter((item) => {
      if (filter === 'anomaly_only' && !item.is_anomaly) return false;
      if (filter === 'normal_only' && item.is_anomaly) return false;
      return true;
    });

    next.sort((a, b) => {
      const left = Number(a.anomaly_score ?? 0);
      const right = Number(b.anomaly_score ?? 0);
      return sortDirection === 'desc' ? right - left : left - right;
    });

    return next;
  }, [items, filter, sortDirection]);

  if (loading) return <LoadingState label="Loading anomaly events..." />;
  if (error) return <div className="panel error-panel">{error}</div>;

  return (
    <div className="page-shell">
      <div className="page-header-row">
        <div>
          <p className="page-kicker">Detection log</p>
          <h2>Anomalies</h2>
        </div>
        <div className="toolbar">
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All</option>
            <option value="anomaly_only">Anomaly only</option>
            <option value="normal_only">Normal only</option>
          </select>

          <select value={sortDirection} onChange={(e) => setSortDirection(e.target.value)}>
            <option value="desc">Score ↓</option>
            <option value="asc">Score ↑</option>
          </select>
        </div>
      </div>

      <div className="panel">
        {filteredData.length === 0 ? (
          <div className="empty-state">No anomaly data available.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Reading ID</th>
                  <th>Well</th>
                  <th>Score</th>
                  <th>Prediction</th>
                  <th>Model</th>
                  <th>Version</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map((item) => (
                  <tr key={item.id}>
                    <td>{new Date(item.timestamp).toLocaleString()}</td>
                    <td>{item.reading_id}</td>
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
    </div>
  );
}
