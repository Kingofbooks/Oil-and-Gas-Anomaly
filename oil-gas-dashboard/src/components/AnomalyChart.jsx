import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const threshold = 0.005;

export default function AnomalyChart({ data = [] }) {
  const chartData = data.map((item, index) => ({
    ...item,
    label: item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : `#${index + 1}`,
  }));

  return (
    <div className="chart-panel">
      <div className="panel-heading">
        <div>
          <p className="panel-kicker">Anomaly score</p>
          <h3>TranAD trend</h3>
        </div>
        <span className="threshold-badge">Threshold 0.005</span>
      </div>

      {chartData.length === 0 ? (
        <div className="empty-state">No anomaly score history available.</div>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData} margin={{ top: 12, right: 12, left: 0, bottom: 18 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.14)" strokeDasharray="3 3" />
            <XAxis dataKey="label" tick={{ fill: '#9ca3af', fontSize: 10 }} minTickGap={20} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} domain={[0, 'auto']} />
            <Tooltip
              contentStyle={{
                background: '#0f172a',
                border: '1px solid rgba(148,163,184,0.3)',
                borderRadius: 12,
                color: '#e5eefb',
              }}
            />
            <ReferenceLine y={threshold} stroke="#fbbf24" strokeDasharray="6 6" label={{ value: 'Threshold', position: 'insideTopRight', fill: '#fbbf24' }} />
            <Line
              type="monotone"
              dataKey="anomaly_score"
              stroke="#38bdf8"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
