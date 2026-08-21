import { ArrowUpRight } from 'lucide-react';

export default function StatCard({ title, value, hint, icon: Icon, tone = 'neutral' }) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <div className="stat-card__header">
        <div className="stat-card__title-wrap">
          <span className="stat-card__eyebrow">{title}</span>
          <h3>{value}</h3>
        </div>
        {Icon ? (
          <div className="stat-card__icon">
            <Icon size={18} />
          </div>
        ) : (
          <div className="stat-card__icon muted">
            <ArrowUpRight size={16} />
          </div>
        )}
      </div>
      {hint && <p className="stat-card__hint">{hint}</p>}
    </div>
  );
}
