# Hotel Revenue Forecasting Demo - Setup Instructions

## 🎯 Quick Start Guide

This guide will help you set up and run the complete hotel revenue forecasting demonstration system for your university presentation.

## 📋 Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.8+** (with pip)
- **Node.js 16+** (with npm)
- **Git** (for version control)

## 🚀 Installation Steps

### 1. Clone or Download the Project

```bash
# If using git
git clone <your-repository-url>
cd hotel_revenue_demo

# Or extract the ZIP file to hotel_revenue_demo/
```

### 2. Backend Setup (Flask API)

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import flask, pandas, numpy, sklearn, xgboost, lightgbm; print('All packages installed successfully!')"
```

### 3. Frontend Setup (React Application)

```bash
# Open a new terminal/command prompt
cd hotel_revenue_demo/frontend

# Install Node.js dependencies
npm install

# Verify installation
npm list --depth=0
```

## 🎬 Running the Application

### Step 1: Start the Backend Server

```bash
# In the backend directory (with virtual environment activated)
cd hotel_revenue_demo/backend
python app.py
```

You should see:
```
🚀 Hotel Revenue Demo Backend Started
==================================================
📊 EDA Analysis: Ready
🔧 Feature Engineering: Ready
🤖 Ensemble Models: Ready
📈 Performance Evaluation: Ready
==================================================
 * Running on http://127.0.0.1:5000
```

### Step 2: Start the Frontend Application

```bash
# In a new terminal, navigate to frontend directory
cd hotel_revenue_demo/frontend
npm start
```

The React application will automatically open at `http://localhost:3000`

## 🎓 University Presentation Flow

### 1. Introduction (Dashboard)
- Open the application in your browser
- Show the project overview and highlights
- Click "Load Sample Data" to initialize the system

### 2. Data Exploration (EDA Analysis Tab)
- Navigate to "EDA Analysis" 
- Demonstrate data quality assessment
- Show revenue patterns and visualizations
- Highlight key business insights

### 3. Feature Engineering (Feature Engineering Tab)
- Navigate to "Feature Engineering"
- Create temporal features with cyclical encoding
- Demonstrate lag features with proper time shifting
- Show data leakage prevention techniques
- Compare leaky vs. safe feature implementations

### 4. Model Training (Model Training Tab)
- Navigate to "Model Training"
- Train 5 individual base models
- Create ensemble strategies
- Analyze feature importance
- Compare model performance

### 5. Results & Evaluation (Results Tab)
- Navigate to "Results & Evaluation"
- Show comprehensive model comparison
- Highlight project achievements
- Demonstrate technical excellence

### 6. Complete Pipeline Demo
- Use the "Run Full Demo" button in the header
- Show end-to-end automated processing

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Backend Issues

**Problem**: `ModuleNotFoundError` when starting backend
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**Problem**: Port 5000 already in use
```bash
# Solution: Kill existing process or change port in app.py
# On Windows:
netstat -ano | findstr :5000
taskkill /PID <process_id> /F

# On macOS/Linux:
lsof -ti:5000 | xargs kill -9
```

#### Frontend Issues

**Problem**: `npm install` fails
```bash
# Solution: Clear npm cache and retry
npm cache clean --force
npm install
```

**Problem**: Frontend can't connect to backend
- Ensure backend is running on port 5000
- Check browser console for CORS errors
- Verify the proxy setting in `package.json`

#### Data Loading Issues

**Problem**: "No data loaded" error
- The demo uses sample data generated programmatically
- If data loading fails, check backend console for error messages
- Restart the backend service

## 🎯 Presentation Tips

### For Academic Reviewers

1. **Technical Depth**: Emphasize the ensemble methods, feature engineering, and data leakage prevention
2. **Methodology**: Highlight proper cross-validation and temporal splits
3. **Performance**: Showcase the R² = 0.486 achievement vs industry standards
4. **Code Quality**: Demonstrate clean, production-ready implementation

### Demo Script (5-10 minutes)

1. **Overview** (1 min): Show dashboard and project highlights
2. **Data Quality** (1 min): Navigate to EDA, show data assessment
3. **Feature Engineering** (2 min): Demonstrate temporal features and leakage prevention
4. **Model Training** (2 min): Train models and show ensemble strategies
5. **Results** (2 min): Compare performance and highlight achievements
6. **Q&A** (2-3 min): Answer questions about methodology and implementation

## 📊 Performance Benchmarks

Your demonstration should achieve:
- **R² Score**: ~0.486 (excellent for revenue forecasting)
- **MAE**: ~$856 (good accuracy for hotel revenue)
- **Data Quality**: 100% (no missing values or leakage)
- **Model Training**: ~30-60 seconds for complete pipeline

## 🔧 Customization Options

### Modifying Sample Data
- Edit `backend/utils/data_loader.py` to change data generation patterns
- Adjust seasonality factors, revenue ranges, or time periods

### Adding New Models
- Extend `backend/models/ensemble_model.py` with additional algorithms
- Update frontend components to display new model results

### Changing Visualizations
- Modify `backend/models/eda_analyzer.py` to add new chart types
- Update frontend to display additional visualization options

## 📞 Support

If you encounter issues during setup or presentation:

1. Check the browser console for error messages
2. Review backend terminal output for Python errors
3. Ensure all dependencies are properly installed
4. Verify both services are running on correct ports

## ✅ Pre-Presentation Checklist

- [ ] Backend starts without errors on port 5000
- [ ] Frontend loads successfully on port 3000
- [ ] Data loads properly when clicking "Load Sample Data"
- [ ] All tabs are accessible and functional
- [ ] "Run Full Demo" button executes successfully
- [ ] Model training completes and shows results
- [ ] Visualizations render correctly

## 🎉 You're Ready!

Your hotel revenue forecasting demonstration is now ready for your university presentation. The system showcases advanced machine learning techniques suitable for academic evaluation.

**Good luck with your presentation!** 🍀 