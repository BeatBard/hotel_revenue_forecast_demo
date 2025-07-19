import React, { useState, useEffect } from 'react';
import { 
  Card, 
  DatePicker, 
  Select, 
  Button, 
  Row, 
  Col, 
  Statistic, 
  Table, 
  message, 
  Spin, 
  Typography,
  Tag,
  Space,
  Tabs,
  Alert
} from 'antd';
import { 
  CalendarOutlined, 
  DollarCircleOutlined, 
  TrophyOutlined, 
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined
} from '@ant-design/icons';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import dayjs from 'dayjs';
import axios from 'axios';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const { Title: AntTitle, Text, Paragraph } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const Predictions = () => {
  // Single Prediction State - Start from where CSV data ends (2024-04-30)
  const [selectedDate, setSelectedDate] = useState(dayjs('2024-05-01'));
  const [selectedMeal, setSelectedMeal] = useState('Breakfast');
  const [singlePrediction, setSinglePrediction] = useState(null);
  const [singleLoading, setSingleLoading] = useState(false);

  // 90-Day Forecast State - Start from where CSV data ends (2024-04-30)
  const [forecastStartDate, setForecastStartDate] = useState(dayjs('2024-05-01'));
  const [forecast90Days, setForecast90Days] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);

  // Accuracy Plots State
  const [accuracyPlots, setAccuracyPlots] = useState(null);
  const [plotsLoading, setPlotsLoading] = useState(false);

  // Time Series Plots State
  const [timeSeriesPlots, setTimeSeriesPlots] = useState(null);
  const [timeSeriesLoading, setTimeSeriesLoading] = useState(false);

  // Single Prediction Function
  const handleSinglePrediction = async () => {
    setSingleLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/predictions/single`, {
        date: selectedDate.format('YYYY-MM-DD'),
        meal_period: selectedMeal,
        model_type: 'best_ensemble'
      });

      if (response.data.success) {
        setSinglePrediction(response.data);
        message.success('Prediction generated successfully!');
      } else {
        message.error(response.data.error || 'Prediction failed');
      }
    } catch (error) {
      console.error('Single prediction error:', error);
      message.error('Failed to generate prediction');
    } finally {
      setSingleLoading(false);
    }
  };

  // 90-Day Forecast Function
  const handle90DayForecast = async () => {
    setForecastLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/predictions/90-days`, {
        start_date: forecastStartDate.format('YYYY-MM-DD')
      });

      if (response.data.success) {
        setForecast90Days(response.data);
        message.success('90-day forecast generated successfully!');
      } else {
        message.error(response.data.error || 'Forecast failed');
      }
    } catch (error) {
      console.error('90-day forecast error:', error);
      message.error('Failed to generate forecast');
    } finally {
      setForecastLoading(false);
    }
  };

  // Accuracy Plots Function
  const handleAccuracyPlots = async () => {
    setPlotsLoading(true);
    try {
      console.log('🚀 Starting accuracy plots request...');
      
             const response = await axios.get(`${API_BASE_URL}/api/models/accuracy-plots`, {
         timeout: 60000, // 60 second timeout for large image
         maxContentLength: 100 * 1024 * 1024, // 100MB limit
         maxBodyLength: 100 * 1024 * 1024,
         responseType: 'json', // Explicitly set JSON response type
         headers: {
           'Accept': 'application/json',
           'Content-Type': 'application/json'
         }
       });

             console.log('✅ Response received');
       console.log('📡 Response status:', response.status);
       console.log('📦 Response headers:', response.headers);
       console.log('📊 Raw response data type:', typeof response.data);
       
               // Handle potential string response that needs parsing
        let parsedData = response.data;
        if (typeof response.data === 'string') {
          console.log('⚠️ Response is string, attempting to parse as JSON...');
          try {
            // Replace Infinity values with null before parsing (fallback safety)
            const cleanedResponse = response.data
              .replace(/:\s*Infinity/g, ': null')
              .replace(/:\s*-Infinity/g, ': null')
              .replace(/:\s*NaN/g, ': null');
            
            parsedData = JSON.parse(cleanedResponse);
            console.log('✅ Successfully parsed string response to object');
          } catch (e) {
            console.error('❌ Failed to parse string response:', e);
            console.error('❌ Raw response causing error:', response.data);
            message.error('Invalid JSON response from server - check console for details');
            return;
          }
        }
       
       console.log('📊 Parsed data type:', typeof parsedData);
       console.log('📏 Response data size (approx):', JSON.stringify(parsedData).length);
       console.log('📈 Response data keys:', parsedData && typeof parsedData === 'object' ? Object.keys(parsedData) : 'No keys');
       console.log('🎯 Success field:', parsedData?.success);
       console.log('🖼️ Has plot_image field:', parsedData && typeof parsedData === 'object' && parsedData.hasOwnProperty('plot_image'));
       console.log('🖼️ Plot_image value type:', parsedData?.plot_image ? typeof parsedData.plot_image : 'undefined');
       console.log('📏 Plot image length:', parsedData?.plot_image?.length || 0);
       console.log('🔍 Plot image first 100 chars:', parsedData?.plot_image?.substring(0, 100) || 'N/A');
       console.log('🔍 Plot image last 100 chars:', parsedData?.plot_image?.substring(-100) || 'N/A');

             // Use the already-parsed data
       const data = parsedData;
       
       console.log('🔍 Final data analysis:');
       console.log('   Data type:', typeof data);
       console.log('   Is object:', data && typeof data === 'object');
       console.log('   Has keys:', data && typeof data === 'object' ? Object.keys(data) : 'No keys');
       
       // Safe property checks using hasOwnProperty
       const hasPlotImage = data && 
                           typeof data === 'object' &&
                           data.hasOwnProperty('plot_image') &&
                           typeof data.plot_image === 'string' && 
                           data.plot_image.length > 100;
                           
       const hasMetrics = data && 
                         typeof data === 'object' &&
                         data.hasOwnProperty('metrics_summary') &&
                         typeof data.metrics_summary === 'object' &&
                         data.metrics_summary !== null &&
                         Object.keys(data.metrics_summary).length > 0;
       
       console.log('   hasPlotImage:', hasPlotImage);
       console.log('   hasMetrics:', hasMetrics);
       console.log('   plot_image exists:', data && typeof data === 'object' && data.hasOwnProperty('plot_image'));
       console.log('   plot_image type:', data?.plot_image ? typeof data.plot_image : 'undefined');
       console.log('   plot_image length:', data?.plot_image?.length || 0);
       console.log('   metrics_summary exists:', data && typeof data === 'object' && data.hasOwnProperty('metrics_summary'));
       console.log('   metrics_summary type:', data?.metrics_summary ? typeof data.metrics_summary : 'undefined');
       
       // Try to display if we have essential data
       if (hasPlotImage && hasMetrics) {
         console.log('✅ Essential data present - displaying plots');
         setAccuracyPlots(data);
         message.success('Accuracy plots generated successfully!');
       } else if (data && data.error) {
         console.error('❌ Backend error:', data.error);
         message.error(`Backend error: ${data.error}`);
       } else {
         console.error('❌ Missing essential data:');
         console.error('   plot_image missing:', !hasPlotImage);
         console.error('   metrics_summary missing:', !hasMetrics);
         console.error('   Full response:', data);
         
         // Try to extract more info about what's wrong
         if (data) {
           if (!data.plot_image) {
             message.error('Plot generation failed - No image data received');
           } else if (!data.metrics_summary) {
             message.error('Plot generation failed - No metrics data received');
           } else {
             message.error('Plot generation failed - Data format issue');
           }
         } else {
           message.error('Plot generation failed - Empty response');
         }
       }
    } catch (error) {
      console.error('❌ Request failed:', error);
      
      if (error.code === 'ECONNABORTED') {
        message.error('Request timeout - Plot generation took too long');
      } else if (error.response) {
        console.error('📤 Error response status:', error.response.status);
        console.error('📥 Error response data:', error.response.data);
        message.error(`Server error: ${error.response.data?.error || error.response.statusText}`);
      } else if (error.request) {
        console.error('📡 Network error:', error.request);
        message.error('Network error - Could not reach server');
      } else {
        console.error('⚙️ Request setup error:', error.message);
        message.error(`Request error: ${error.message}`);
      }
    } finally {
      setPlotsLoading(false);
    }
  };

  // Time Series Plots Function
  const handleTimeSeriesPlots = async () => {
    setTimeSeriesLoading(true);
    try {
      console.log('📈 Starting time series plots request...');
      
      const response = await axios.get(`${API_BASE_URL}/api/predictions/time-series-plots`, {
        timeout: 60000,
        maxContentLength: 100 * 1024 * 1024,
        maxBodyLength: 100 * 1024 * 1024,
        responseType: 'json',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });

      console.log('✅ Time series response received');
      console.log('📊 Response status:', response.status);
      console.log('📈 Response data keys:', response.data && typeof response.data === 'object' ? Object.keys(response.data) : 'No keys');

      // Handle potential string response that needs parsing
      let parsedData = response.data;
      if (typeof response.data === 'string') {
        console.log('⚠️ Response is string, attempting to parse as JSON...');
        try {
          const cleanedResponse = response.data
            .replace(/:\s*Infinity/g, ': null')
            .replace(/:\s*-Infinity/g, ': null')
            .replace(/:\s*NaN/g, ': null');
          
          parsedData = JSON.parse(cleanedResponse);
          console.log('✅ Successfully parsed string response to object');
        } catch (e) {
          console.error('❌ Failed to parse string response:', e);
          message.error('Invalid JSON response from server - check console for details');
          return;
        }
      }

      // Use the already-parsed data
      const data = parsedData;
      
      // Safe property checks using hasOwnProperty
      const hasPlotImage = data && 
                          typeof data === 'object' &&
                          data.hasOwnProperty('plot_image') &&
                          typeof data.plot_image === 'string' && 
                          data.plot_image.length > 100;
                          
      const hasMetrics = data && 
                        typeof data === 'object' &&
                        data.hasOwnProperty('metrics') &&
                        typeof data.metrics === 'object' &&
                        data.metrics !== null;

      console.log('🔍 Time series data analysis:');
      console.log('   hasPlotImage:', hasPlotImage);
      console.log('   hasMetrics:', hasMetrics);
      console.log('   plot_image length:', data?.plot_image?.length || 0);

      // Try to display if we have essential data
      if (hasPlotImage && hasMetrics) {
        console.log('✅ Time series data present - displaying plots');
        setTimeSeriesPlots(data);
        message.success('Time series plots generated successfully!');
      } else if (data && data.error) {
        console.error('❌ Backend error:', data.error);
        message.error(`Backend error: ${data.error}`);
      } else {
        console.error('❌ Missing essential time series data:');
        console.error('   plot_image missing:', !hasPlotImage);
        console.error('   metrics missing:', !hasMetrics);
        
        if (data) {
          if (!data.plot_image) {
            message.error('Time series plot generation failed - No image data received');
          } else if (!data.metrics) {
            message.error('Time series plot generation failed - No metrics data received');
          } else {
            message.error('Time series plot generation failed - Data format issue');
          }
        } else {
          message.error('Time series plot generation failed - Empty response');
        }
      }
    } catch (error) {
      console.error('❌ Time series request failed:', error);
      
      if (error.code === 'ECONNABORTED') {
        message.error('Request timeout - Time series plot generation took too long');
      } else if (error.response) {
        console.error('📤 Error response status:', error.response.status);
        console.error('📥 Error response data:', error.response.data);
        message.error(`Server error: ${error.response.data?.error || error.response.statusText}`);
      } else if (error.request) {
        console.error('📡 Network error:', error.request);
        message.error('Network error - Could not reach server');
      } else {
        console.error('⚙️ Request setup error:', error.message);
        message.error(`Request error: ${error.message}`);
      }
    } finally {
      setTimeSeriesLoading(false);
    }
  };

  // Create forecast chart data
  const createForecastChartData = () => {
    if (!forecast90Days?.detailed_forecast) return null;

    const forecastData = forecast90Days.detailed_forecast;
    
    // Group by meal period
    const breakfastData = forecastData.filter(item => item.meal_period === 'Breakfast');
    const lunchData = forecastData.filter(item => item.meal_period === 'Lunch');
    const dinnerData = forecastData.filter(item => item.meal_period === 'Dinner');

    // Get labels (dates) - using every 7th day to avoid crowding
    const labels = breakfastData
      .filter((_, index) => index % 7 === 0)
      .map(item => dayjs(item.date).format('MMM DD'));

    return {
      labels,
      datasets: [
        {
          label: 'Breakfast',
          data: breakfastData.filter((_, index) => index % 7 === 0).map(item => item.predicted_revenue),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.1
        },
        {
          label: 'Lunch',
          data: lunchData.filter((_, index) => index % 7 === 0).map(item => item.predicted_revenue),
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          tension: 0.1
        },
        {
          label: 'Dinner',
          data: dinnerData.filter((_, index) => index % 7 === 0).map(item => item.predicted_revenue),
          borderColor: 'rgb(255, 205, 86)',
          backgroundColor: 'rgba(255, 205, 86, 0.2)',
          tension: 0.1
        }
      ]
    };
  };

  // Create meal period statistics chart
  const createMealStatsChart = () => {
    if (!forecast90Days?.meal_period_statistics) return null;

    const stats = forecast90Days.meal_period_statistics;
    const mealPeriods = Object.keys(stats);
    
    return {
      labels: mealPeriods,
      datasets: [
        {
          label: 'Average Revenue',
          data: mealPeriods.map(meal => stats[meal].average),
          backgroundColor: [
            'rgba(255, 99, 132, 0.6)',
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 205, 86, 0.6)'
          ],
          borderColor: [
            'rgb(255, 99, 132)',
            'rgb(54, 162, 235)',
            'rgb(255, 205, 86)'
          ],
          borderWidth: 1
        }
      ]
    };
  };

  // Accuracy metrics table columns
  const accuracyColumns = [
    {
      title: 'Model',
      dataIndex: 'model',
      key: 'model',
      render: (text) => <Text strong>{text}</Text>
    },
    {
      title: 'R² Score',
      dataIndex: 'r2_score',
      key: 'r2_score',
      render: (value) => <Text>{(value * 100).toFixed(1)}%</Text>,
      sorter: (a, b) => a.r2_score - b.r2_score
    },
    {
      title: 'MAE',
      dataIndex: 'mae',
      key: 'mae',
      render: (value) => <Text>${value.toFixed(2)}</Text>,
      sorter: (a, b) => a.mae - b.mae
    },
    {
      title: 'RMSE',
      dataIndex: 'rmse',
      key: 'rmse',
      render: (value) => <Text>${value.toFixed(2)}</Text>,
      sorter: (a, b) => a.rmse - b.rmse
    },
    {
      title: 'MAPE',
      dataIndex: 'mape',
      key: 'mape',
      render: (value) => <Text>{value.toFixed(1)}%</Text>,
      sorter: (a, b) => a.mape - b.mape
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <AntTitle level={2}>
        <LineChartOutlined /> Revenue Predictions & Forecasting
      </AntTitle>
      <Paragraph>
        Generate revenue predictions for specific dates, create 90-day forecasts, and analyze model accuracy.
      </Paragraph>

      <Tabs defaultActiveKey="single" size="large">
        {/* Single Prediction Tab */}
        <TabPane tab={<span><CalendarOutlined />Single Prediction</span>} key="single">
          <Row gutter={[24, 24]}>
            <Col span={12}>
              <Card title="🔮 Single Revenue Prediction" bordered={false}>
                <Space direction="vertical" size="large" style={{ width: '100%' }}>
                  <div>
                    <Text strong>Select Date:</Text>
                    <DatePicker
                      value={selectedDate}
                      onChange={setSelectedDate}
                      style={{ width: '100%', marginTop: 8 }}
                      format="YYYY-MM-DD"
                      disabledDate={(current) => current && current < dayjs('2024-05-01')}
                      placeholder="Select date from 2024-05-01 onwards"
                    />
                  </div>
                  
                  <div>
                    <Text strong>Select Meal Period:</Text>
                    <Select
                      value={selectedMeal}
                      onChange={setSelectedMeal}
                      style={{ width: '100%', marginTop: 8 }}
                    >
                      <Option value="Breakfast">🌅 Breakfast</Option>
                      <Option value="Lunch">🌞 Lunch</Option>
                      <Option value="Dinner">🌙 Dinner</Option>
                    </Select>
                  </div>

                  <Button
                    type="primary"
                    size="large"
                    loading={singleLoading}
                    onClick={handleSinglePrediction}
                    icon={<DollarCircleOutlined />}
                    block
                  >
                    Generate Prediction
                  </Button>
                </Space>
              </Card>
            </Col>

            <Col span={12}>
              {singlePrediction ? (
                <Card title="💰 Prediction Results" bordered={false}>
                  <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                    <Statistic
                      title="Predicted Revenue"
                      value={singlePrediction.predicted_revenue}
                      prefix="$"
                      precision={2}
                      valueStyle={{ color: '#3f8600', fontSize: '28px' }}
                    />
                    
                    <Row gutter={16}>
                      <Col span={12}>
                        <Statistic
                          title="Date"
                          value={dayjs(singlePrediction.date).format('MMM DD, YYYY')}
                          valueStyle={{ fontSize: '16px' }}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="Meal Period"
                          value={singlePrediction.meal_period}
                          valueStyle={{ fontSize: '16px' }}
                        />
                      </Col>
                    </Row>

                    <div>
                      <Text strong>Best Model: </Text>
                      <Tag color="blue" icon={<TrophyOutlined />}>
                        {singlePrediction.best_model}
                      </Tag>
                    </div>

                    <div>
                      <Text strong>Islamic Period: </Text>
                      <Tag color="purple">{singlePrediction.islamic_period}</Tag>
                    </div>

                    <div>
                      <Text strong>Tourism Intensity: </Text>
                      <Tag color={
                        singlePrediction.tourism_intensity === 'High' ? 'red' :
                        singlePrediction.tourism_intensity === 'Medium' ? 'orange' : 'green'
                      }>
                        {singlePrediction.tourism_intensity}
                      </Tag>
                    </div>

                    {singlePrediction.individual_predictions && (
                      <div>
                        <Text strong>Individual Model Predictions:</Text>
                        <div style={{ marginTop: 8 }}>
                          {Object.entries(singlePrediction.individual_predictions).map(([model, value]) => (
                            <div key={model} style={{ marginBottom: 4 }}>
                              <Text>{model}: <Text strong>${Number(value).toFixed(2)}</Text></Text>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </Space>
                </Card>
              ) : (
                <Card>
                  <div style={{ textAlign: 'center', padding: '40px' }}>
                    <CalendarOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                    <div style={{ marginTop: 16 }}>
                      <Text type="secondary">Select a date and meal period to generate prediction</Text>
                    </div>
                  </div>
                </Card>
              )}
            </Col>
          </Row>
        </TabPane>

        {/* 90-Day Forecast Tab */}
        <TabPane tab={<span><BarChartOutlined />90-Day Forecast</span>} key="forecast">
          <Row gutter={[24, 24]}>
            <Col span={24}>
              <Card title="📈 90-Day Revenue Forecast" bordered={false}>
                <Row gutter={16} style={{ marginBottom: 24 }}>
                  <Col span={8}>
                    <Text strong>Start Date:</Text>
                    <DatePicker
                      value={forecastStartDate}
                      onChange={setForecastStartDate}
                      style={{ width: '100%', marginTop: 8 }}
                      format="YYYY-MM-DD"
                      disabledDate={(current) => current && current < dayjs('2024-05-01')}
                      placeholder="Select start date from 2024-05-01 onwards"
                    />
                  </Col>
                  <Col span={8}>
                    <div style={{ marginTop: 32 }}>
                      <Button
                        type="primary"
                        size="large"
                        loading={forecastLoading}
                        onClick={handle90DayForecast}
                        icon={<LineChartOutlined />}
                      >
                        Generate 90-Day Forecast
                      </Button>
                    </div>
                  </Col>
                </Row>

                {forecast90Days && (
                  <div>
                    <Row gutter={16} style={{ marginBottom: 24 }}>
                      <Col span={6}>
                        <Statistic
                          title="Total Forecast Revenue"
                          value={forecast90Days.total_forecast_revenue}
                          prefix="$"
                          precision={2}
                          valueStyle={{ color: '#3f8600' }}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="Daily Average"
                          value={forecast90Days.daily_average_revenue}
                          prefix="$"
                          precision={2}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="Total Predictions"
                          value={forecast90Days.total_predictions}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="Forecast Period"
                          value="90 Days"
                        />
                      </Col>
                    </Row>

                    <Row gutter={[24, 24]}>
                      <Col span={16}>
                        <Card title="📊 Revenue Forecast by Meal Period" size="small">
                          {createForecastChartData() && (
                            <Line
                              data={createForecastChartData()}
                              options={{
                                responsive: true,
                                plugins: {
                                  legend: {
                                    position: 'top',
                                  },
                                  title: {
                                    display: true,
                                    text: '90-Day Revenue Forecast'
                                  }
                                },
                                scales: {
                                  y: {
                                    beginAtZero: true,
                                    ticks: {
                                      callback: function(value) {
                                        return '$' + value.toLocaleString();
                                      }
                                    }
                                  }
                                }
                              }}
                            />
                          )}
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card title="📈 Meal Period Statistics" size="small">
                          {createMealStatsChart() && (
                            <Bar
                              data={createMealStatsChart()}
                              options={{
                                responsive: true,
                                plugins: {
                                  legend: {
                                    display: false
                                  },
                                  title: {
                                    display: true,
                                    text: 'Average Revenue by Meal'
                                  }
                                },
                                scales: {
                                  y: {
                                    beginAtZero: true,
                                    ticks: {
                                      callback: function(value) {
                                        return '$' + value.toLocaleString();
                                      }
                                    }
                                  }
                                }
                              }}
                            />
                          )}
                        </Card>
                      </Col>
                    </Row>
                  </div>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        {/* Model Accuracy Tab */}
        <TabPane tab={<span><PieChartOutlined />Model Accuracy</span>} key="accuracy">
          <Row gutter={[24, 24]}>
            <Col span={24}>
              <Card title="🎯 Model Accuracy Analysis" bordered={false}>
                <Row gutter={16} style={{ marginBottom: 24 }}>
                  <Col span={12}>
                    <Button
                      type="primary"
                      size="large"
                      loading={plotsLoading}
                      onClick={handleAccuracyPlots}
                      icon={<BarChartOutlined />}
                      block
                    >
                      Generate Accuracy Plots
                    </Button>
                  </Col>
                  <Col span={12}>
                    <Button
                      type="default"
                      size="large"
                      loading={timeSeriesLoading}
                      onClick={handleTimeSeriesPlots}
                      icon={<LineChartOutlined />}
                      block
                    >
                      Generate Time Series Plots
                    </Button>
                  </Col>
                </Row>

                {accuracyPlots && (
                  <div>
                    <Alert
                      message={`Best performing model: ${accuracyPlots.best_model}`}
                      description={`Analysis based on ${accuracyPlots.test_samples} test samples`}
                      type="success"
                      showIcon
                      style={{ marginBottom: 24 }}
                    />

                    <Row gutter={[24, 24]}>
                      <Col span={16}>
                        <Card title="📈 Predicted vs Actual Values" size="small">
                          {accuracyPlots.plot_image ? (
                            <div style={{ textAlign: 'center' }}>
                              <img
                                src={`data:image/png;base64,${accuracyPlots.plot_image}`}
                                alt="Model Accuracy Plots"
                                style={{ 
                                  width: '100%', 
                                  maxWidth: '100%', 
                                  height: 'auto',
                                  border: '1px solid #d9d9d9',
                                  borderRadius: '6px'
                                }}
                                onError={(e) => {
                                  console.error('Image load error:', e);
                                  e.target.style.display = 'none';
                                  e.target.nextSibling.style.display = 'block';
                                }}
                              />
                              <div style={{ display: 'none', padding: '20px', color: '#ff4d4f' }}>
                                ❌ Error loading plot image. Check console for details.
                              </div>
                            </div>
                          ) : (
                            <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
                              📊 Plot image not available
                            </div>
                          )}
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card title="📊 Performance Metrics" size="small">
                          <Table
                            columns={accuracyColumns}
                            dataSource={Object.entries(accuracyPlots.metrics_summary).map(([model, metrics]) => ({
                              key: model,
                              model,
                              ...metrics
                            }))}
                            pagination={false}
                            size="small"
                          />
                        </Card>
                      </Col>
                    </Row>
                  </div>
                )}

                {timeSeriesPlots && (
                  <div style={{ marginTop: 24 }}>
                    <Alert
                      message={`Time Series Analysis: ${timeSeriesPlots.metrics?.best_model || 'Best Model'}`}
                      description={`Analysis period: ${timeSeriesPlots.date_range} | Test samples: ${timeSeriesPlots.test_samples} | R² Score: ${(timeSeriesPlots.metrics?.r2_score * 100)?.toFixed(1)}%`}
                      type="info"
                      showIcon
                      style={{ marginBottom: 24 }}
                    />

                    <Row gutter={[24, 24]}>
                      <Col span={24}>
                        <Card title="📈 Time Series: Actual vs Predicted Revenue" size="small">
                          {timeSeriesPlots.plot_image ? (
                            <div style={{ textAlign: 'center' }}>
                              <img
                                src={`data:image/png;base64,${timeSeriesPlots.plot_image}`}
                                alt="Time Series Plots"
                                style={{ 
                                  width: '100%', 
                                  maxWidth: '100%', 
                                  height: 'auto',
                                  border: '1px solid #d9d9d9',
                                  borderRadius: '6px'
                                }}
                                onError={(e) => {
                                  console.error('Time series image load error:', e);
                                  e.target.style.display = 'none';
                                  e.target.nextSibling.style.display = 'block';
                                }}
                              />
                              <div style={{ display: 'none', padding: '20px', color: '#ff4d4f' }}>
                                ❌ Error loading time series plot. Check console for details.
                              </div>
                            </div>
                          ) : (
                            <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
                              📈 Time series plot not available
                            </div>
                          )}
                        </Card>
                      </Col>
                    </Row>

                    {timeSeriesPlots.metrics && (
                      <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
                        <Col span={6}>
                          <Statistic
                            title="Best Model"
                            value={timeSeriesPlots.metrics.best_model?.replace('_', ' ')?.toUpperCase()}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="R² Score"
                            value={(timeSeriesPlots.metrics.r2_score * 100).toFixed(1)}
                            suffix="%"
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="Mean Actual"
                            value={timeSeriesPlots.metrics.mean_actual}
                            prefix="$"
                            precision={0}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="Mean Predicted"
                            value={timeSeriesPlots.metrics.mean_predicted}
                            prefix="$"
                            precision={0}
                          />
                        </Col>
                      </Row>
                    )}
                  </div>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default Predictions; 