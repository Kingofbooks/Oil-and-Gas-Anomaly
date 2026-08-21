import { api, withErrorHandling } from './client';

export const getWells = () =>
  withErrorHandling(() => api.get('/wells').then((res) => res.data));

export const getReadingsByWell = (wellId, limit = 50) =>
  withErrorHandling(() =>
    api.get('/readings', { params: { well_id: wellId, limit } }).then((res) => res.data)
  );
