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
  edaAnalysis: (endpoint) => apiClient.get(`/eda/${endpoint}`),
  getDataOverview: () => apiClient.get('/eda/data-overview'),

  getRevenueDistributions: () => apiClient.get('/eda/revenue-distributions'),
  getTimeSeriesAnalysis: () => apiClient.get('/eda/time-series-analysis'),
  getCorrelationAnalysis: () => apiClient.get('/eda/correlation-analysis'),
  getCategoricalAnalysis: () => apiClient.get('/eda/categorical-analysis'),
  getOutlierAnalysis: () => apiClient.get('/eda/outlier-analysis'),

  // Feature engineering endpoints
  createTemporalFeatures: (config = {}) => apiClient.post('/feature-engineering/create-features', config),
  createLagFeatures: (config = {}) => apiClient.post('/feature-engineering/lag-features', config),
  createRollingFeatures: (config = {}) => apiClient.post('/feature-engineering/rolling-features', config),
  getLeakageDemo: () => apiClient.get('/feature-engineering/leakage-prevention'),
  analyzeFeatureImportance: () => apiClient.get('/feature-engineering/feature-importance'),
  analyzeFeatureCorrelations: () => apiClient.get('/feature-engineering/correlation-analysis'),

  // Model training endpoints
  trainIndividualModels: (config = {}) => apiClient.post('/models/individual-training', config),
  trainEnsembleModels: (config = {}) => apiClient.post('/models/ensemble-training', config),
  getModelComparison: () => apiClient.get('/models/performance-comparison'),
  generatePredictions: (config = {}) => apiClient.post('/models/predictions', config),
  getModelFeatureImportance: (model = 'xgboost') => apiClient.get(`/models/feature-importance?model=${model}`),
  
  // Original data analysis
  getOriginalDataFeatures: () => apiClient.get('/analysis/original-data-features'),

  // Evaluation endpoints
  getEvaluationMetrics: () => apiClient.get('/evaluation/metrics'),
  getEvaluationVisualization: (type) => apiClient.get(`/evaluation/visualizations?type=${type}`),

  // Full demonstration
  runFullDemonstration: () => apiClient.post('/demonstration/full-pipeline'),
}; 