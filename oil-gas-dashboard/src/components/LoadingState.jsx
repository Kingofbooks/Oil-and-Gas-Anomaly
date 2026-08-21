export default function LoadingState({ label = 'Loading data...' }) {
  return (
    <div className="loading-state">
      <div className="spinner" />
      <span>{label}</span>
    </div>
  );
}
