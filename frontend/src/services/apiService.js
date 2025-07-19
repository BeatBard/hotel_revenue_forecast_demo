import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error);
    const message = error.response?.data?.error || error.message || 'An error occurred';
    throw new Error(message);
  }
);

export const apiService = {
  // Health check
  healthCheck: () => apiClient.get('/health'),

  // Data loading
  loadData: () => apiClient.post('/load-data'),

  // EDA endpoints
  getDataOverview: () => apiClient.get('/eda/overview'),
  getRevenuePatterns: () => apiClient.get('/eda/revenue-patterns'),
  getSeasonalAnalysis: () => apiClient.get('/eda/seasonal-analysis'),
  getEDAVisualization: (type) => apiClient.get(`/eda/visualizations?type=${type}`),

  // Feature engineering endpoints
  createTemporalFeatures: (config = {}) => apiClient.post('/feature-engineering/create-features', config),
  createLagFeatures: (config = {}) => apiClient.post('/feature-engineering/lag-features', config),
  createRollingFeatures: (config = {}) => apiClient.post('/feature-engineering/rolling-features', config),
  getLeakageDemo: () => apiClient.get('/feature-engineering/leakage-prevention'),

  // Model training endpoints
  trainIndividualModels: (config = {}) => apiClient.post('/models/individual-training', config),
  trainEnsembleModels: (config = {}) => apiClient.post('/models/ensemble-training', config),
  getModelComparison: () => apiClient.get('/models/performance-comparison'),
  generatePredictions: (config = {}) => apiClient.post('/models/predictions', config),
  getFeatureImportance: (model = 'xgboost') => apiClient.get(`/models/feature-importance?model=${model}`),
  
  // Original data analysis
  getOriginalDataFeatures: () => apiClient.get('/analysis/original-data-features'),

  // Evaluation endpoints
  getEvaluationMetrics: () => apiClient.get('/evaluation/metrics'),
  getEvaluationVisualization: (type) => apiClient.get(`/evaluation/visualizations?type=${type}`),

  // Full demonstration
  runFullDemonstration: () => apiClient.post('/demonstration/full-pipeline'),
}; 