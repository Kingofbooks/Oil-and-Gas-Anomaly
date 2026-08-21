import { api, withErrorHandling } from './client';

export const getDashboardSummary = () =>
  withErrorHandling(() => api.get('/dashboard/summary').then((res) => res.data));

export const getHealth = () =>
  withErrorHandling(() => api.get('/health').then((res) => res.data));
