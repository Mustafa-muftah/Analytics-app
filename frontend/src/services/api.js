import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add error interceptor
api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Camera endpoints
export const getCameras = async () => {
  const response = await api.get('/api/cameras');
  return response.data;
};

export const getCamera = async (cameraId) => {
  const response = await api.get(`/api/cameras/${cameraId}`);
  return response.data;
};

// Count endpoints
export const getCurrentCount = async (cameraId = null) => {
  const params = cameraId ? { camera_id: cameraId } : {};
  const response = await api.get('/api/count', { params });
  return response.data;
};

export const getPeakHours = async (cameraId = null) => {
  const params = cameraId ? { camera_id: cameraId } : {};
  const response = await api.get('/api/peak-hours', { params });
  return response.data;
};

// Gender endpoints
export const getGenderStats = async (cameraId = null) => {
  const params = cameraId ? { camera_id: cameraId } : {};
  const response = await api.get('/api/gender-stats', { params });
  return response.data;
};

// Zone endpoints
export const getZoneStats = async () => {
  const response = await api.get('/api/zone-stats');
  return response.data;
};

// Visitor tracking endpoints
export const getVisitorStats = async (cameraId = null) => {
  const params = cameraId ? { camera_id: cameraId } : {};
  const response = await api.get('/api/visitor-stats', { params });
  return response.data;
};

export const getDwellTime = async (cameraId = null) => {
  const params = cameraId ? { camera_id: cameraId } : {};
  const response = await api.get('/api/dwell-time', { params });
  return response.data;
};

// Heatmap endpoints
export const getLatestHeatmap = async (cameraId = null) => {
  const params = cameraId ? { camera_id: cameraId } : {};
  const response = await api.get('/api/heatmap', { params });
  return response.data;
};

// Batch processing endpoints
export const startBatchProcessing = async (cameraId) => {
  const response = await api.post(`/api/batch/start/${cameraId}`);
  return response.data;
};

export const getBatchProgress = async (cameraId) => {
  const response = await api.get(`/api/batch/progress/${cameraId}`);
  return response.data;
};

export const getProcessingJobs = async (cameraId = null, status = null) => {
  const params = {};
  if (cameraId) params.camera_id = cameraId;
  if (status) params.status = status;
  const response = await api.get('/api/batch/jobs', { params });
  return response.data;
};

// Stats endpoint
export const getStats = async () => {
  const response = await api.get('/api/stats');
  return response.data;
};

// Report endpoint
export const getReportSummary = async (cameraId = null) => {
  const params = cameraId ? { camera_id: cameraId } : {};
  const response = await api.get('/api/report/summary', { params });
  return response.data;
};

export default api;
