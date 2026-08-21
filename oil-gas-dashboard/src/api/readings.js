import { api, withErrorHandling } from './client';

export const getReadings = (params = {}) =>
  withErrorHandling(() =>
    api.get('/readings', { params }).then((res) => res.data)
  );

export const createReading = (payload) =>
  withErrorHandling(() =>
    api.post('/readings', payload).then((res) => res.data)
  );
