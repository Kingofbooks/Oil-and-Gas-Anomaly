import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getWells } from '../api/wells';
import LoadingState from '../components/LoadingState';

export default function Wells() {
  const [wells, setWells] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getWells();
        setWells(Array.isArray(data) ? data : []);
      } catch (err) {
        setError(err.message || 'Unable to load wells.');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  if (loading) return <LoadingState label="Loading wells..." />;
  if (error) return <div className="panel error-panel">{error}</div>;

  return (
    <div className="page-shell">
      <div className="page-header-row">
        <div>
          <p className="page-kicker">Asset roster</p>
          <h2>Wells</h2>
        </div>
      </div>

      <div className="well-grid">
        {wells.length === 0 ? (
          <div className="empty-state">No wells available.</div>
        ) : (
          wells.map((well) => (
            <Link key={well} to={`/wells/${encodeURIComponent(well)}`} className="well-card">
              <div className="well-card__top">
                <span className="well-token">Well</span>
                <span className="well-status">ONLINE</span>
              </div>
              <h3>{well}</h3>
              <p>Open the well detail view for recent readings and anomaly history.</p>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}
