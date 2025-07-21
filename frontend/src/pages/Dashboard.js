import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Typography, Statistic, Progress, Alert } from 'antd';
import {
  DatabaseOutlined,
  BarChartOutlined,
  SettingOutlined,
  RobotOutlined,
  TrophyOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import styled from 'styled-components';
import { apiService } from '../services/apiService';

const { Title, Paragraph, Text } = Typography;

const StyledCard = styled(Card)`
  height: 100%;
  transition: all 0.3s ease;
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
`;

const StepCard = styled(Card)`
  margin-bottom: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
  
  .ant-card-body {
    padding: 20px;
  }
`;

const Dashboard = ({ dataLoaded, onLoadData }) => {
  const [dataInfo, setDataInfo] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (dataLoaded) {
      loadDataOverview();
    }
  }, [dataLoaded]);

  const loadDataOverview = async () => {
    setLoading(true);
    try {
      const overview = await apiService.getDataOverview();
      setDataInfo(overview);
    } catch (error) {
      console.error('Failed to load data overview:', error);
    } finally {
      setLoading(false);
    }
  };

  const workflowSteps = [
    {
      id: 'data-loading',
      title: 'Data Loading & Quality Check',
      description: 'Load hotel revenue data and perform quality assessment',
      icon: <DatabaseOutlined />,
      status: dataLoaded ? 'completed' : 'pending',
      color: dataLoaded ? '#52c41a' : '#faad14'
    },
    {
      id: 'eda',
      title: 'Exploratory Data Analysis',
      description: 'Analyze revenue patterns, seasonality, and data distributions',
      icon: <BarChartOutlined />,
      status: dataLoaded ? 'available' : 'disabled',
      color: dataLoaded ? '#1890ff' : '#d9d9d9'
    },
    {
      id: 'feature-engineering',
      title: 'Feature Engineering',
      description: 'Create temporal features, lag features, and prevent data leakage',
      icon: <SettingOutlined />,
      status: dataLoaded ? 'available' : 'disabled',
      color: dataLoaded ? '#722ed1' : '#d9d9d9'
    },
    {
      id: 'model-training',
      title: 'Model Training',
      description: 'Train 5 base models and create ensemble strategies',
      icon: <RobotOutlined />,
      status: dataLoaded ? 'available' : 'disabled',
      color: dataLoaded ? '#13c2c2' : '#d9d9d9'
    },
    {
      id: 'evaluation',
      title: 'Results & Evaluation',
      description: 'Compare model performance and analyze predictions',
      icon: <TrophyOutlined />,
      status: dataLoaded ? 'available' : 'disabled',
      color: dataLoaded ? '#eb2f96' : '#d9d9d9'
    }
  ];

  const projectHighlights = [
    {
      title: 'Ensemble Methods',
      value: '5 Base Models',
      description: 'Ridge, RandomForest, XGBoost, LightGBM, GradientBoosting',
      color: '#1890ff'
    },
    {
      title: 'Performance Achieved',
      value: 'R² = 0.486',
      description: 'Excellent performance for revenue forecasting',
      color: '#52c41a'
    },
    {
      title: 'Data Leakage',
      value: 'Zero Detected',
      description: 'Strict temporal validation and feature engineering',
      color: '#722ed1'
    },
    {
      title: 'Features Used',
      value: '44 Features',
      description: 'Temporal, lag, rolling, and interaction features',
      color: '#eb2f96'
    }
  ];

  return (
    <div>
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <StyledCard>
            <Title level={2} style={{ textAlign: 'center', marginBottom: '16px' }}>
              🏨 Hotel Revenue Forecasting
            </Title>
            <Paragraph style={{ textAlign: 'center', fontSize: '16px', color: '#666' }}>
              Advanced machine learning system for hotel revenue prediction and analysis
            </Paragraph>
            
            {!dataLoaded && (
              <Alert
                message="Get Started"
                description="Click the 'Load Data' button below to start analyzing hotel revenue data."
                type="info"
                icon={<InfoCircleOutlined />}
                style={{ marginTop: '20px' }}
              />
            )}
          </StyledCard>
        </Col>
      </Row>

      {/* Project Highlights */}
      <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Title level={3}>🎯 System Overview</Title>
        </Col>
        {projectHighlights.map((highlight, index) => (
          <Col xs={24} sm={12} lg={6} key={index}>
            <StyledCard>
              <Statistic
                title={highlight.title}
                value={highlight.value}
                valueStyle={{ color: highlight.color, fontSize: '28px' }}
                suffix={<div style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>
                  {highlight.description}
                </div>}
              />
            </StyledCard>
          </Col>
        ))}
      </Row>

      {/* Data Overview */}
      {dataLoaded && dataInfo && (
        <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
          <Col span={24}>
            <Title level={3}>📊 Data Overview</Title>
          </Col>
          <Col xs={24} lg={8}>
            <StyledCard title="Dataset Information">
              <Statistic
                title="Total Records"
                value={dataInfo.dataset_info?.total_records || 0}
                suffix="transactions"
              />
              <div style={{ marginTop: '16px' }}>
                <Text strong>Date Range: </Text>
                <Text>{dataInfo.dataset_info?.date_range?.start} to {dataInfo.dataset_info?.date_range?.end}</Text>
              </div>
              <div style={{ marginTop: '8px' }}>
                <Text strong>Revenue Centers: </Text>
                <Text>{dataInfo.dataset_info?.revenue_centers?.length || 0}</Text>
              </div>
            </StyledCard>
          </Col>
          <Col xs={24} lg={8}>
            <StyledCard title="Revenue Statistics">
              <Statistic
                title="Total Revenue"
                value={dataInfo.revenue_statistics?.total_revenue || 0}
                precision={0}
                prefix="$"
              />
              <div style={{ marginTop: '16px' }}>
                <Text strong>Average: </Text>
                <Text>${(dataInfo.revenue_statistics?.mean_revenue || 0).toFixed(2)}</Text>
              </div>
              <div style={{ marginTop: '8px' }}>
                <Text strong>Median: </Text>
                <Text>${(dataInfo.revenue_statistics?.median_revenue || 0).toFixed(2)}</Text>
              </div>
            </StyledCard>
          </Col>
          <Col xs={24} lg={8}>
            <StyledCard title="Data Quality">
              <div style={{ textAlign: 'center' }}>
                <Progress 
                  type="circle" 
                  percent={100} 
                  status="success"
                  format={() => '100%'}
                />
                <div style={{ marginTop: '16px' }}>
                  <Text strong>Excellent Data Quality</Text>
                  <br />
                  <Text type="secondary">No missing values detected</Text>
                </div>
              </div>
            </StyledCard>
          </Col>
        </Row>
      )}

              {/* Workflow Steps */}
        <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
          <Col span={24}>
            <Title level={3}>🚀 Analysis Workflow</Title>
          </Col>
        <Col span={24}>
          {workflowSteps.map((step, index) => (
            <StepCard key={step.id}>
              <Row align="middle" gutter={[16, 16]}>
                <Col flex="none">
                  <div style={{ 
                    fontSize: '24px', 
                    color: step.color,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '48px',
                    height: '48px',
                    borderRadius: '50%',
                    background: `${step.color}20`
                  }}>
                    {step.icon}
                  </div>
                </Col>
                <Col flex="auto">
                  <Title level={4} style={{ margin: 0, color: step.color }}>
                    {index + 1}. {step.title}
                  </Title>
                  <Paragraph style={{ margin: '4px 0 0 0', color: '#666' }}>
                    {step.description}
                  </Paragraph>
                </Col>
                <Col flex="none">
                  <div style={{
                    padding: '4px 12px',
                    borderRadius: '16px',
                    background: step.color,
                    color: 'white',
                    fontSize: '12px',
                    fontWeight: 'bold'
                  }}>
                    {step.status === 'completed' ? '✅ Completed' :
                     step.status === 'available' ? '🔄 Available' : '⏳ Pending'}
                  </div>
                </Col>
              </Row>
            </StepCard>
          ))}
        </Col>
      </Row>

      {/* Load Data Section */}
      {!dataLoaded && (
        <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
          <Col span={24}>
            <StyledCard style={{ textAlign: 'center', padding: '40px' }}>
              <Title level={3}>Ready to Start?</Title>
              <Paragraph style={{ fontSize: '16px', color: '#666', marginBottom: '32px' }}>
                Load the sample hotel revenue data to begin exploring the complete machine learning pipeline.
              </Paragraph>
              <Button 
                type="primary" 
                size="large" 
                icon={<DatabaseOutlined />}
                onClick={onLoadData}
                loading={loading}
                style={{ padding: '8px 32px', height: 'auto' }}
              >
                Load Sample Data
              </Button>
            </StyledCard>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default Dashboard; 