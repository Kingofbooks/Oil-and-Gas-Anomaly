import { api, withErrorHandling } from './client';

export const getAlerts = (params = {}) =>
  withErrorHandling(() =>
    api.get('/alerts', { params }).then((res) => res.data)
  );

export const resolveAlert = (alertId, status = 'RESOLVED') =>
  withErrorHandling(() =>
    api.patch(`/alerts/${alertId}`, { status }).then((res) => res.data)
  );
