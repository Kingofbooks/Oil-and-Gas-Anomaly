import { useEffect, useMemo, useState } from 'react';
import { getAlerts } from '../api/alerts';
import LoadingState from '../components/LoadingState';

const severityOptions = ['ALL', 'CRITICAL', 'WARNING', 'INFO'];

export default function AlertsPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [severityFilter, setSeverityFilter] = useState('ALL');

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getAlerts({ limit: 200 });
        setItems(Array.isArray(data) ? data : []);
      } catch (err) {
        setError(err.message || 'Unable to load alerts.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  const filteredData = useMemo(() => {
    return items.filter((item) => {
      const statusMatch = statusFilter === 'ALL' || item.status === statusFilter;
      const severityMatch = severityFilter === 'ALL' || String(item.severity).toUpperCase() === severityFilter;
      return statusMatch && severityMatch;
    });
  }, [items, statusFilter, severityFilter]);

  if (loading) return <LoadingState label="Loading alerts..." />;
  if (error) return <div className="panel error-panel">{error}</div>;

  return (
    <div className="page-shell">
      <div className="page-header-row">
        <div>
          <p className="page-kicker">Alert operations</p>
          <h2>Alerts</h2>
        </div>

        <div className="toolbar">
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="ALL">All</option>
            <option value="OPEN">OPEN</option>
            <option value="RESOLVED">RESOLVED</option>
          </select>

          <select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}>
            {severityOptions.map((item) => (
              <option key={item} value={item}>{item === 'ALL' ? 'All severities' : item}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="panel">
        {filteredData.length === 0 ? (
          <div className="empty-state">No alerts match the selected filters.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Well</th>
                  <th>Message</th>
                  <th>Status</th>
                  <th>Created At</th>
                  <th>Resolved At</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <span className={`severity-pill ${String(item.severity).toLowerCase()}`}>
                        {String(item.severity || 'INFO').toUpperCase()}
                      </span>
                    </td>
                    <td>{item.well_id}</td>
                    <td>{item.message}</td>
                    <td>
                      <span className={`alert-status ${item.status === 'OPEN' ? 'open' : 'resolved'}`}>{item.status}</span>
                    </td>
                    <td>{new Date(item.created_at).toLocaleString()}</td>
                    <td>{item.resolved_at ? new Date(item.resolved_at).toLocaleString() : '—'}</td>
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
