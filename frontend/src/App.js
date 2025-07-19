import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout, Menu, Button, message, Card, Spin } from 'antd';
import {
  DashboardOutlined,
  BarChartOutlined,
  SettingOutlined,
  RobotOutlined,
  TrophyOutlined,
  PlayCircleOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import styled from 'styled-components';

// Import pages
import Dashboard from './pages/Dashboard';

import FeatureEngineering from './pages/FeatureEngineering';
import ModelTraining from './pages/ModelTraining';
import Results from './pages/Results';
import Predictions from './pages/Predictions';
import { apiService } from './services/apiService';

const { Header, Sider, Content } = Layout;

const StyledLayout = styled(Layout)`
  min-height: 100vh;
`;

const StyledHeader = styled(Header)`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
`;

const StyledSider = styled(Sider)`
  background: white;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
`;

const StyledContent = styled(Content)`
  margin: 24px;
  padding: 24px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  min-height: calc(100vh - 96px);
`;

const Logo = styled.div`
  color: white;
  font-size: 20px;
  font-weight: bold;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const StatusCard = styled(Card)`
  margin: 16px;
  .ant-card-body {
    padding: 16px;
  }
`;

const App = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [dataLoaded, setDataLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await apiService.healthCheck();
      setBackendStatus('connected');
      console.log('Backend connected:', response.message);
    } catch (error) {
      setBackendStatus('disconnected');
      console.error('Backend connection failed:', error);
      message.error('Backend connection failed. Please ensure the Flask server is running on port 5000.');
    }
  };

  const loadData = async () => {
    setLoading(true);
    try {
      const response = await apiService.loadData();
      setDataLoaded(true);
      message.success('Data loaded successfully!');
      console.log('Data info:', response.data_info);
    } catch (error) {
      message.error('Failed to load data: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const runFullDemo = async () => {
    if (!dataLoaded) {
      message.warning('Please load data first');
      return;
    }
    
    setLoading(true);
    try {
      const response = await apiService.runFullDemonstration();
      message.success('Complete demonstration pipeline executed successfully!');
      console.log('Demo results:', response.results);
    } catch (error) {
      message.error('Demo execution failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },

    {
      key: 'feature-engineering',
      icon: <SettingOutlined />,
      label: 'Feature Engineering',
    },
    {
      key: 'model-training',
      icon: <RobotOutlined />,
      label: 'Model Training',
    },
    {
      key: 'results',
      icon: <TrophyOutlined />,
      label: 'Results & Evaluation',
    },
    {
      key: 'predictions',
      icon: <LineChartOutlined />,
      label: 'Predictions & Forecast',
    },
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected':
        return '#52c41a';
      case 'disconnected':
        return '#ff4d4f';
      default:
        return '#faad14';
    }
  };

  const renderContent = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard dataLoaded={dataLoaded} onLoadData={loadData} />;

      case 'feature-engineering':
        return <FeatureEngineering dataLoaded={dataLoaded} />;
      case 'model-training':
        return <ModelTraining dataLoaded={dataLoaded} />;
      case 'results':
        return <Results dataLoaded={dataLoaded} />;
      case 'predictions':
        return <Predictions dataLoaded={dataLoaded} />;
      default:
        return <Dashboard dataLoaded={dataLoaded} onLoadData={loadData} />;
    }
  };

  return (
    <StyledLayout>
      <StyledHeader>
        <Logo>
          <RobotOutlined />
          Hotel Revenue Forecasting Demo
        </Logo>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={runFullDemo}
            loading={loading}
            disabled={!dataLoaded || backendStatus !== 'connected'}
            size="large"
          >
            Run Full Demo
          </Button>
          <div style={{ 
            color: getStatusColor(backendStatus),
            fontSize: '14px',
            fontWeight: 'bold'
          }}>
            Backend: {backendStatus}
          </div>
        </div>
      </StyledHeader>
      
      <Layout>
        <StyledSider
          trigger={null}
          collapsible
          collapsed={collapsed}
          width={250}
        >
          <StatusCard size="small" style={{ margin: collapsed ? '8px' : '16px' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                color: dataLoaded ? '#52c41a' : '#faad14',
                fontWeight: 'bold',
                fontSize: collapsed ? '12px' : '14px'
              }}>
                {dataLoaded ? '✅ Data Ready' : '⏳ Load Data'}
              </div>
              {!collapsed && !dataLoaded && (
                <Button 
                  size="small" 
                  type="primary" 
                  onClick={loadData}
                  loading={loading}
                  style={{ marginTop: '8px' }}
                >
                  Load Data
                </Button>
              )}
            </div>
          </StatusCard>
          
          <Menu
            mode="inline"
            selectedKeys={[currentPage]}
            items={menuItems}
            onClick={({ key }) => setCurrentPage(key)}
            style={{ border: 'none' }}
          />
        </StyledSider>
        
        <Layout>
          <StyledContent>
            <Spin spinning={loading} tip="Processing...">
              {renderContent()}
            </Spin>
          </StyledContent>
        </Layout>
      </Layout>
    </StyledLayout>
  );
};

export default App; 