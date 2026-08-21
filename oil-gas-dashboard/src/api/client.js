import axios from 'axios';

export const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 15000,
});

export const withErrorHandling = async (request) => {
  try {
    return await request();
  } catch (error) {
    if (error.response) {
      throw new Error(
        error.response.data?.detail || error.response.data?.message || 'API request failed'
      );
    }
    if (error.request) {
      throw new Error('API unavailable. Backend is offline or unreachable.');
    }
    throw new Error(error.message || 'Request failed');
  }
};
