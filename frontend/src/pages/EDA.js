import React, { useState, useEffect } from 'react';
import { Card, Tabs, Spin, Alert, Row, Col, Statistic, Table, Typography, Image, Space, Tag } from 'antd';
import { 
  BarChartOutlined, 
  LineChartOutlined, 
  PieChartOutlined, 
  DashboardOutlined,
  ExclamationCircleOutlined,
  LinkOutlined 
} from '@ant-design/icons';
import styled from 'styled-components';
import { apiService } from '../services/apiService';

const { TabPane } = Tabs;
const { Title, Text } = Typography;

const StyledCard = styled(Card)`
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  
  .ant-card-head {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px 8px 0 0;
    
    .ant-card-head-title {
      color: white;
      font-weight: 600;
    }
  }
`;

const StyledTabs = styled(Tabs)`
  .ant-tabs-tab {
    font-weight: 500;
  }
  
  .ant-tabs-tab-active {
    .ant-tabs-tab-btn {
      color: #667eea !important;
    }
  }
`;

const StatsCard = styled(Card)`
  text-align: center;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  
  .ant-statistic-title {
    font-size: 14px;
    color: #666;
  }
  
  .ant-statistic-content {
    color: #1890ff;
  }
`;

const ImageContainer = styled.div`
  text-align: center;
  margin: 20px 0;
  
  img {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
`;

const EDA = () => {
  const [loading, setLoading] = useState(false);
  const [dataOverview, setDataOverview] = useState(null);

  const [revenueDistributions, setRevenueDistributions] = useState(null);
  const [timeSeriesAnalysis, setTimeSeriesAnalysis] = useState(null);
  const [correlationAnalysis, setCorrelationAnalysis] = useState(null);
  const [categoricalAnalysis, setCategoricalAnalysis] = useState(null);
  const [outlierAnalysis, setOutlierAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  const loadEDAData = async (endpoint, setter) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.edaAnalysis(endpoint);
      
      // Validate response structure
      if (!response || !response.success) {
        throw new Error(response?.error || 'Invalid response from server');
      }
      
      setter(response);
    } catch (err) {
      console.error(`Error loading EDA data for ${endpoint}:`, err);
      setError(`Failed to load ${endpoint.replace('-', ' ')}: ${err.message}`);
      setter(null); // Reset the state on error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Load data overview by default
    loadEDAData('data-overview', setDataOverview);
  }, []);

  const handleTabChange = (key) => {
    setActiveTab(key);
    
    // Load data for the selected tab if not already loaded
    switch (key) {
      case 'overview':
        if (!dataOverview) loadEDAData('data-overview', setDataOverview);
        break;

      case 'distributions':
        if (!revenueDistributions) loadEDAData('revenue-distributions', setRevenueDistributions);
        break;
      case 'timeseries':
        if (!timeSeriesAnalysis) loadEDAData('time-series-analysis', setTimeSeriesAnalysis);
        break;
      case 'correlations':
        if (!correlationAnalysis) loadEDAData('correlation-analysis', setCorrelationAnalysis);
        break;
      case 'categorical':
        if (!categoricalAnalysis) loadEDAData('categorical-analysis', setCategoricalAnalysis);
        break;
      case 'outliers':
        if (!outlierAnalysis) loadEDAData('outlier-analysis', setOutlierAnalysis);
        break;
      default:
        break;
    }
  };

  const renderDataOverview = () => (
    <div>
      {dataOverview && (
        <>
          {/* Basic Stats */}
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <StatsCard>
                <Statistic 
                  title="Total Records" 
                  value={dataOverview.data_shape[0].toLocaleString()} 
                />
              </StatsCard>
            </Col>
            <Col span={6}>
              <StatsCard>
                <Statistic 
                  title="Total Columns" 
                  value={dataOverview.data_shape[1]} 
                />
              </StatsCard>
            </Col>
            <Col span={6}>
              <StatsCard>
                <Statistic 
                  title="Revenue Centers" 
                  value={dataOverview.revenue_centers.length} 
                />
              </StatsCard>
            </Col>
            <Col span={6}>
              <StatsCard>
                <Statistic 
                  title="Date Range (Days)" 
                  value={dataOverview.date_range.total_days} 
                />
              </StatsCard>
            </Col>
          </Row>

          {/* Revenue Statistics */}
          <StyledCard title="Revenue Statistics" style={{ marginBottom: 24 }}>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Statistic
                  title="Total Revenue"
                  value={dataOverview.revenue_stats.total_revenue}
                  precision={2}
                  prefix="$"
                  valueStyle={{ color: '#3f8600' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Average Revenue"
                  value={dataOverview.revenue_stats.mean_revenue}
                  precision={2}
                  prefix="$"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Median Revenue"
                  value={dataOverview.revenue_stats.median_revenue}
                  precision={2}
                  prefix="$"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
            </Row>
          </StyledCard>

          {/* Date Range */}
          <StyledCard title="Data Coverage" style={{ marginBottom: 24 }}>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Text strong>Start Date: </Text>
                <Tag color="green">{dataOverview.date_range.start_date}</Tag>
              </Col>
              <Col span={8}>
                <Text strong>End Date: </Text>
                <Tag color="red">{dataOverview.date_range.end_date}</Tag>
              </Col>
              <Col span={8}>
                <Text strong>Total Days: </Text>
                <Tag color="blue">{dataOverview.date_range.total_days}</Tag>
              </Col>
            </Row>
          </StyledCard>

          {/* Revenue Centers and Meal Periods */}
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <StyledCard title="Revenue Centers">
                <Space wrap>
                  {dataOverview.revenue_centers.map((center, index) => (
                    <Tag key={index} color="processing">{center}</Tag>
                  ))}
                </Space>
              </StyledCard>
            </Col>
            <Col span={12}>
              <StyledCard title="Meal Periods">
                <Space wrap>
                  {dataOverview.meal_periods.map((period, index) => (
                    <Tag key={index} color="warning">{period}</Tag>
                  ))}
                </Space>
              </StyledCard>
            </Col>
          </Row>
        </>
      )}
    </div>
  );



  const renderRevenueDistributions = () => (
    <div>
      {revenueDistributions && revenueDistributions.plots && (
        <>
          <StyledCard title="Revenue Distribution Analysis">
            <ImageContainer>
              <Image
                src={`data:image/png;base64,${revenueDistributions.plots.revenue_distributions}`}
                alt="Revenue Distributions"
                preview={{
                  mask: 'Click to view full size'
                }}
              />
            </ImageContainer>
          </StyledCard>

          {revenueDistributions.meal_period_stats && revenueDistributions.center_stats && (
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <StyledCard title="Revenue by Meal Period">
                  <Table
                    size="small"
                    dataSource={Object.entries(revenueDistributions.meal_period_stats.mean || {}).map(([period, mean]) => ({
                      key: period,
                      period,
                      mean: mean?.toFixed(2) || 'N/A',
                      median: revenueDistributions.meal_period_stats.median?.[period]?.toFixed(2) || 'N/A',
                      std: revenueDistributions.meal_period_stats.std?.[period]?.toFixed(2) || 'N/A'
                    }))}
                    columns={[
                      { title: 'Meal Period', dataIndex: 'period', key: 'period' },
                      { title: 'Mean ($)', dataIndex: 'mean', key: 'mean' },
                      { title: 'Median ($)', dataIndex: 'median', key: 'median' },
                      { title: 'Std Dev ($)', dataIndex: 'std', key: 'std' }
                    ]}
                    pagination={false}
                  />
                </StyledCard>
              </Col>
              <Col span={12}>
                <StyledCard title="Revenue by Center">
                  <Table
                    size="small"
                    dataSource={Object.entries(revenueDistributions.center_stats.mean || {}).map(([center, mean]) => ({
                      key: center,
                      center,
                      mean: mean?.toFixed(2) || 'N/A',
                      median: revenueDistributions.center_stats.median?.[center]?.toFixed(2) || 'N/A',
                      std: revenueDistributions.center_stats.std?.[center]?.toFixed(2) || 'N/A'
                    }))}
                    columns={[
                      { title: 'Revenue Center', dataIndex: 'center', key: 'center' },
                      { title: 'Mean ($)', dataIndex: 'mean', key: 'mean' },
                      { title: 'Median ($)', dataIndex: 'median', key: 'median' },
                      { title: 'Std Dev ($)', dataIndex: 'std', key: 'std' }
                    ]}
                    pagination={false}
                  />
                </StyledCard>
              </Col>
            </Row>
          )}
        </>
      )}
    </div>
  );

  const renderTimeSeriesAnalysis = () => (
    <div>
      {timeSeriesAnalysis && timeSeriesAnalysis.plots && (
        <>
          <StyledCard title="Time Series Analysis">
            <ImageContainer>
              <Image
                src={`data:image/png;base64,${timeSeriesAnalysis.plots.time_series}`}
                alt="Time Series Analysis"
                preview={{
                  mask: 'Click to view full size'
                }}
              />
            </ImageContainer>
          </StyledCard>

          <Row gutter={[16, 16]}>
            <Col span={8}>
              <StyledCard title="Daily Revenue Stats">
                <Table
                  size="small"
                  showHeader={false}
                  dataSource={Object.entries(timeSeriesAnalysis.seasonal_stats?.daily_stats || {}).map(([stat, value]) => ({
                    key: stat,
                    stat: stat.charAt(0).toUpperCase() + stat.slice(1),
                    value: typeof value === 'number' ? value.toFixed(2) : value
                  }))}
                  columns={[
                    { dataIndex: 'stat', key: 'stat' },
                    { dataIndex: 'value', key: 'value' }
                  ]}
                  pagination={false}
                />
              </StyledCard>
            </Col>
            <Col span={8}>
              <StyledCard title="Monthly Revenue Stats">
                <Table
                  size="small"
                  showHeader={false}
                  dataSource={Object.entries(timeSeriesAnalysis.seasonal_stats?.monthly_stats || {}).map(([stat, value]) => ({
                    key: stat,
                    stat: stat.charAt(0).toUpperCase() + stat.slice(1),
                    value: typeof value === 'number' ? value.toFixed(2) : value
                  }))}
                  columns={[
                    { dataIndex: 'stat', key: 'stat' },
                    { dataIndex: 'value', key: 'value' }
                  ]}
                  pagination={false}
                />
              </StyledCard>
            </Col>
            <Col span={8}>
              <StyledCard title="Meal Period Averages">
                <Table
                  size="small"
                  dataSource={Object.entries(timeSeriesAnalysis.seasonal_stats?.meal_period_stats || {}).map(([period, avg]) => ({
                    key: period,
                    period,
                    average: avg?.toFixed(2) || 'N/A'
                  }))}
                  columns={[
                    { title: 'Period', dataIndex: 'period', key: 'period' },
                    { title: 'Avg ($)', dataIndex: 'average', key: 'average' }
                  ]}
                  pagination={false}
                />
              </StyledCard>
            </Col>
          </Row>
        </>
      )}
    </div>
  );

  const renderCorrelationAnalysis = () => (
    <div>
      {correlationAnalysis && correlationAnalysis.correlation_plot && (
        <>
          <StyledCard title="Correlation Analysis with Revenue (CheckTotal)">
            <ImageContainer>
              <Image
                src={`data:image/png;base64,${correlationAnalysis.correlation_plot}`}
                alt="Revenue Correlation Analysis"
                preview={{
                  mask: 'Click to view full size'
                }}
              />
            </ImageContainer>
          </StyledCard>

          <StyledCard title="Revenue Correlations">
            <Table
              dataSource={Object.entries(correlationAnalysis.revenue_correlations || {}).map(([variable, correlation], index) => ({
                key: index,
                variable,
                correlation
              }))}
              columns={[
                { title: 'Variable', dataIndex: 'variable', key: 'variable' },
                { 
                  title: 'Correlation', 
                  dataIndex: 'correlation', 
                  key: 'correlation',
                  render: (value) => (
                    <Tag color={value > 0.5 ? 'green' : value < -0.5 ? 'red' : value > 0 ? 'blue' : 'orange'}>
                      {value > 0 ? '+' : ''}{value.toFixed(3)}
                    </Tag>
                  )
                }
              ]}
              pagination={{ pageSize: 15, showSizeChanger: true }}
            />
          </StyledCard>

          <StyledCard title="Analysis Summary">
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Statistic
                  title="Variables Analyzed"
                  value={correlationAnalysis.total_variables || 0}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Positive Correlations"
                  value={Object.values(correlationAnalysis.revenue_correlations || {}).filter(v => v > 0).length}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Strongest Correlation"
                  value={Object.values(correlationAnalysis.revenue_correlations || {}).length > 0 ? 
                    Math.max(...Object.values(correlationAnalysis.revenue_correlations).map(Math.abs)).toFixed(3) : '0.000'}
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
            </Row>
          </StyledCard>
        </>
      )}
    </div>
  );

  const renderCategoricalAnalysis = () => (
    <div>
      {categoricalAnalysis && categoricalAnalysis.plots && (
        <>
          <StyledCard title="Categorical Variables Analysis">
            <ImageContainer>
              <Image
                src={`data:image/png;base64,${categoricalAnalysis.plots.categorical_analysis}`}
                alt="Categorical Analysis"
                preview={{
                  mask: 'Click to view full size'
                }}
              />
            </ImageContainer>
          </StyledCard>

          {categoricalAnalysis.event_impact && Object.keys(categoricalAnalysis.event_impact).length > 0 && (
            <StyledCard title="Event Impact on Revenue">
              <Table
                dataSource={Object.entries(categoricalAnalysis.event_impact || {}).map(([event, impact]) => ({
                  key: event,
                  event: event.replace('Is', ''),
                  ...impact
                }))}
                columns={[
                  { title: 'Event', dataIndex: 'event', key: 'event' },
                  { 
                    title: 'With Event ($)', 
                    dataIndex: 'with_event', 
                    key: 'with_event',
                    render: (value) => value?.toFixed(2) || 'N/A'
                  },
                  { 
                    title: 'Without Event ($)', 
                    dataIndex: 'without_event', 
                    key: 'without_event',
                    render: (value) => value?.toFixed(2) || 'N/A'
                  },
                  { 
                    title: 'Impact ($)', 
                    dataIndex: 'impact', 
                    key: 'impact',
                    render: (value) => value !== undefined && value !== null ? (
                      <Tag color={value > 0 ? 'green' : value < 0 ? 'red' : 'default'}>
                        {value > 0 ? '+' : ''}{value.toFixed(2)}
                      </Tag>
                    ) : 'N/A'
                  }
                ]}
                pagination={false}
              />
            </StyledCard>
          )}
        </>
      )}
    </div>
  );

  const renderOutlierAnalysis = () => (
    <div>
      {outlierAnalysis && outlierAnalysis.outlier_plot && (
        <>
          <StyledCard title="Outlier Analysis">
            <ImageContainer>
              <Image
                src={`data:image/png;base64,${outlierAnalysis.outlier_plot}`}
                alt="Outlier Analysis"
                preview={{
                  mask: 'Click to view full size'
                }}
              />
            </ImageContainer>
          </StyledCard>

          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <StatsCard>
                <Statistic 
                  title="Total Outliers" 
                  value={outlierAnalysis.outlier_stats.total_outliers} 
                  valueStyle={{ color: '#f5222d' }}
                />
              </StatsCard>
            </Col>
            <Col span={6}>
              <StatsCard>
                <Statistic 
                  title="Outlier Percentage" 
                  value={outlierAnalysis.outlier_stats.outlier_percentage} 
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: '#fa8c16' }}
                />
              </StatsCard>
            </Col>
            <Col span={6}>
              <StatsCard>
                <Statistic 
                  title="Upper Bound" 
                  value={outlierAnalysis.outlier_stats.upper_bound} 
                  precision={2}
                  prefix="$"
                  valueStyle={{ color: '#1890ff' }}
                />
              </StatsCard>
            </Col>
            <Col span={6}>
              <StatsCard>
                <Statistic 
                  title="Lower Bound" 
                  value={outlierAnalysis.outlier_stats.lower_bound} 
                  precision={2}
                  prefix="$"
                  valueStyle={{ color: '#52c41a' }}
                />
              </StatsCard>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col span={12}>
              <StyledCard title="Top 10 Highest Revenue Outliers">
                <Table
                  size="small"
                  dataSource={outlierAnalysis.top_outliers || []}
                  columns={[
                    { title: 'Date', dataIndex: 'Date', key: 'date', render: (date) => date ? new Date(date).toLocaleDateString() : 'N/A' },
                    { title: 'Meal Period', dataIndex: 'MealPeriod', key: 'meal' },
                    { title: 'Revenue Center', dataIndex: 'RevenueCenterName', key: 'center' },
                    { title: 'Revenue ($)', dataIndex: 'CheckTotal', key: 'revenue', render: (val) => val?.toFixed(2) || 'N/A' }
                  ]}
                  pagination={false}
                />
              </StyledCard>
            </Col>
            <Col span={12}>
              <StyledCard title="Top 10 Lowest Revenue Outliers">
                <Table
                  size="small"
                  dataSource={outlierAnalysis.bottom_outliers || []}
                  columns={[
                    { title: 'Date', dataIndex: 'Date', key: 'date', render: (date) => date ? new Date(date).toLocaleDateString() : 'N/A' },
                    { title: 'Meal Period', dataIndex: 'MealPeriod', key: 'meal' },
                    { title: 'Revenue Center', dataIndex: 'RevenueCenterName', key: 'center' },
                    { title: 'Revenue ($)', dataIndex: 'CheckTotal', key: 'revenue', render: (val) => val?.toFixed(2) || 'N/A' }
                  ]}
                  pagination={false}
                />
              </StyledCard>
            </Col>
          </Row>
        </>
      )}
    </div>
  );

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2} style={{ marginBottom: '24px', color: '#1890ff' }}>
        <BarChartOutlined /> Exploratory Data Analysis
      </Title>
      
      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Spin spinning={loading} size="large">
        <StyledTabs activeKey={activeTab} onChange={handleTabChange} type="card">
          <TabPane 
            tab={<span><DashboardOutlined />Data Overview</span>} 
            key="overview"
          >
            {renderDataOverview()}
          </TabPane>
          

          
          <TabPane 
            tab={<span><PieChartOutlined />Distributions</span>} 
            key="distributions"
          >
            {renderRevenueDistributions()}
          </TabPane>
          
          <TabPane 
            tab={<span><LineChartOutlined />Time Series</span>} 
            key="timeseries"
          >
            {renderTimeSeriesAnalysis()}
          </TabPane>
          
          <TabPane 
            tab={<span><LinkOutlined />Correlations</span>} 
            key="correlations"
          >
            {renderCorrelationAnalysis()}
          </TabPane>
          
          <TabPane 
            tab={<span><BarChartOutlined />Categorical</span>} 
            key="categorical"
          >
            {renderCategoricalAnalysis()}
          </TabPane>
          
          <TabPane 
            tab={<span><ExclamationCircleOutlined />Outliers</span>} 
            key="outliers"
          >
            {renderOutlierAnalysis()}
          </TabPane>
        </StyledTabs>
      </Spin>
    </div>
  );
};

export default EDA; 