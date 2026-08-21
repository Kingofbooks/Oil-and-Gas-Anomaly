import { useMemo, useState } from 'react';
import axios from 'axios';

const FEATURE_COUNT = 22;
const WINDOW_SIZE = 120;

const defaultWindow = Array.from({ length: WINDOW_SIZE }, () => Array(FEATURE_COUNT).fill(0));

export default function Predict() {
  const [jsonText, setJsonText] = useState(JSON.stringify(defaultWindow, null, 2));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const validation = useMemo(() => {
    try {
      const parsed = JSON.parse(jsonText);
      if (!Array.isArray(parsed)) {
        return 'Input must be a 2D array.';
      }
      if (parsed.length !== WINDOW_SIZE) {
        return `Exactly ${WINDOW_SIZE} rows are required. Received ${parsed.length}.`;
      }

      for (let i = 0; i < parsed.length; i += 1) {
        if (!Array.isArray(parsed[i])) {
          return `Row ${i} must be an array.`;
        }
        if (parsed[i].length !== FEATURE_COUNT) {
          return `Row ${i} must contain exactly ${FEATURE_COUNT} values. Received ${parsed[i].length}.`;
        }
      }

      return '';
    } catch (e) {
      return 'Invalid JSON. Paste a valid 2D numeric array.';
    }
  }, [jsonText]);

  const handleSubmit = async () => {
    setError('');
    setResult(null);

    if (validation) {
      setError(validation);
      return;
    }

    try {
      setLoading(true);
      const payload = JSON.parse(jsonText);
      const response = await axios.post('http://127.0.0.1:8000/predict', { readings: payload });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-shell">
      <div className="page-header-row">
        <div>
          <p className="page-kicker">Model test</p>
          <h2>Manual AI Prediction</h2>
        </div>
      </div>

      <div className="panel predict-panel">
        <div className="predict-grid">
          <div>
            <label className="field-label">JSON input</label>
            <textarea
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              rows={16}
              className="predict-textarea"
            />
          </div>

          <div className="predict-side">
            <div className="predict-info">
              <h3>Requirements</h3>
              <ul>
                <li>120 rows</li>
                <li>22 features per row</li>
                <li>Numeric values only</li>
              </ul>
            </div>

            <button className="primary-button" onClick={handleSubmit} disabled={loading}>
              {loading ? 'Running prediction...' : 'Run TranAD Prediction'}
            </button>

            {validation && <div className="validation-box">{validation}</div>}
            {error && <div className="error-box">{error}</div>}

            {result && (
              <div className={`prediction-result ${result.is_anomaly ? 'anomaly' : 'normal'}`}>
                <span>{result.is_anomaly ? 'ANOMALY DETECTED' : 'NORMAL'}</span>
                <strong>{result.anomaly_score.toFixed(6)}</strong>
                <small>
                  Model: {result.model} | Version: {result.version} | Threshold: {result.threshold}
                </small>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
