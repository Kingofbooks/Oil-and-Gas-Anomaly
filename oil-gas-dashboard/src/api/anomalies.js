import { api, withErrorHandling } from './client';

export const getAnomalies = (params = {}) =>
  withErrorHandling(() =>
    api.get('/anomalies', { params }).then((res) => res.data)
  );

export const getLatestAnomalies = (limit = 20) =>
  withErrorHandling(() =>
    api.get('/anomalies/latest', { params: { limit } }).then((res) => res.data)
  );
