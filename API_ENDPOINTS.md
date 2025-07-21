# Hotel Revenue Demo - API Endpoints

## 🚀 Flask Backend API Documentation

**Base URL**: `http://localhost:5000/api`

## 📋 Available Endpoints

### Health Check
```http
GET /api/health
```
**Description**: Check if the backend service is running  
**Response**: Service status and timestamp

---

### Data Management

#### Load Data
```http
POST /api/load-data
```
**Description**: Load sample hotel revenue data  
**Response**: Data information and statistics

---

### EDA (Exploratory Data Analysis)

#### Get Data Overview
```http
GET /api/eda/data-overview
```
**Description**: Get comprehensive data overview and basic statistics  
**Response**: Dataset shape, missing values, data types, date ranges, revenue stats


#### Get Revenue Distributions
```http
GET /api/eda/revenue-distributions
```
**Description**: Generate revenue distribution plots and analysis  
**Response**: Distribution plots by meal period, day of week, revenue center + statistics

#### Get Time Series Analysis
```http
GET /api/eda/time-series-analysis
```
**Description**: Analyze revenue patterns over time  
**Response**: Daily/monthly trends, seasonal patterns, time series plots

#### Get Correlation Analysis
```http
GET /api/eda/correlation-analysis
```
**Description**: Analyze correlations between numerical variables  
**Response**: Correlation matrix heatmap, strong correlations detection

#### Get Categorical Analysis
```http
GET /api/eda/categorical-analysis
```
**Description**: Analyze categorical variables and their impact on revenue  
**Response**: Revenue by categories, event impact analysis, categorical plots

#### Get Outlier Analysis
```http
GET /api/eda/outlier-analysis
```
**Description**: Detect and analyze outliers in revenue data  
**Response**: Outlier detection using IQR method, visualizations, outlier details

#### Get EDA Visualizations
```http
GET /api/eda/visualizations?type={viz_type}
```
**Description**: Generate EDA visualizations  
**Parameters**:
- `type`: `revenue_distribution`, `revenue_by_meal_period`, `daily_revenue_trend`, `monthly_comparison`, `revenue_heatmap`
**Response**: Plotly JSON visualization data

---

### Feature Engineering

#### Create Temporal Features
```http
POST /api/feature-engineering/create-features
```
**Description**: Create temporal features with cyclical encoding  
**Body**: Configuration options (optional)  
**Response**: Feature creation details and samples

#### Create Lag Features
```http
POST /api/feature-engineering/lag-features
```
**Description**: Create lag features with proper time shifting  
**Body**: 
```json
{
  "lags": [1, 7, 14, 30]
}
```
**Response**: Lag feature details and leakage prevention info

#### Create Rolling Features
```http
POST /api/feature-engineering/rolling-features
```
**Description**: Create rolling window features  
**Body**: 
```json
{
  "windows": [7, 14, 30]
}
```
**Response**: Rolling feature details and safe implementation info

#### Get Leakage Prevention Demo
```http
GET /api/feature-engineering/leakage-prevention
```
**Description**: Demonstrate data leakage prevention  
**Response**: Examples of leaky vs safe features

#### Get Feature Importance Analysis
```http
GET /api/feature-engineering/feature-importance
```
**Description**: Analyze feature importance for predicting CheckTotal using Random Forest model with real revenue data  
**Response**: Feature importance plot, rankings, model performance metrics, feature categories

#### Get Feature Correlation Analysis
```http
GET /api/feature-engineering/correlation-analysis
```
**Description**: Analyze correlation between features and revenue to identify low-correlation features for removal  
**Response**: Correlation distribution plot, high/low correlation features, removal recommendations

---

### Model Training

#### Train Individual Models
```http
POST /api/models/individual-training
```
**Description**: Train 5 base models  
**Body**: 
```json
{
  "models": ["ridge", "random_forest", "xgboost", "lightgbm", "gradient_boosting"]
}
```
**Response**: Training results and performance metrics

#### Train Ensemble Models
```http
POST /api/models/ensemble-training
```
**Description**: Create ensemble strategies  
**Body**: 
```json
{
  "ensembles": ["simple", "weighted", "top3", "median"]
}
```
**Response**: Ensemble performance and strategy details

#### Get Model Performance Comparison
```http
GET /api/models/performance-comparison
```
**Description**: Compare all model performances  
**Response**: Comprehensive model comparison

#### Generate Predictions
```http
POST /api/models/predictions
```
**Description**: Generate predictions with trained models  
**Body**: 
```json
{
  "model": "ensemble"
}
```
**Response**: Predictions and performance metrics

#### Get Feature Importance
```http
GET /api/models/feature-importance?model={model_name}
```
**Description**: Get feature importance from models  
**Parameters**:
- `model`: `xgboost`, `random_forest`, `gradient_boosting`, etc.
**Response**: Feature importance rankings

---

### Evaluation

#### Get Evaluation Metrics
```http
GET /api/evaluation/metrics
```
**Description**: Get comprehensive evaluation metrics  
**Response**: All model metrics and comparison

#### Get Evaluation Visualizations
```http
GET /api/evaluation/visualizations?type={viz_type}
```
**Description**: Generate evaluation visualizations  
**Parameters**:
- `type`: `predictions_vs_actual`, `residuals_plot`, `feature_importance`, `model_comparison`
**Response**: Evaluation visualization data

---

### Full Demonstration

#### Run Complete Pipeline
```http
POST /api/demonstration/full-pipeline
```
**Description**: Execute the complete ML pipeline  
**Response**: End-to-end results including EDA, feature engineering, training, and evaluation

---

## 🧪 Testing the Endpoints

### Using curl:
```bash
# Health check
curl http://localhost:5000/api/health

# Load data
curl -X POST http://localhost:5000/api/load-data

# Get data overview
curl http://localhost:5000/api/eda/overview

# Train models
curl -X POST http://localhost:5000/api/models/individual-training \
  -H "Content-Type: application/json" \
  -d '{"models": ["ridge", "xgboost"]}'
```

### Using browser:
Navigate to any GET endpoint directly:
- http://localhost:5000/api/health
- http://localhost:5000/api/eda/overview
- http://localhost:5000/api/models/performance-comparison

### Using the React Frontend:
The React app automatically calls these endpoints through the user interface.

---

## 🔧 Response Format

All endpoints return JSON responses with this structure:

### Success Response:
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully"
}
```

### Error Response:
```json
{
  "success": false,
  "error": "Error description"
}
```

---

## 📊 Typical Workflow

1. **Health Check**: `GET /api/health`
2. **Load Data**: `POST /api/load-data`
3. **EDA Analysis**: `GET /api/eda/overview`
4. **Feature Engineering**: `POST /api/feature-engineering/create-features`
5. **Train Models**: `POST /api/models/individual-training`
6. **Create Ensembles**: `POST /api/models/ensemble-training`
7. **Evaluate**: `GET /api/evaluation/metrics`
8. **Full Demo**: `POST /api/demonstration/full-pipeline` 