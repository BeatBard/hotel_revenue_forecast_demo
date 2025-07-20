import React, { useState } from 'react';
import { Card, Row, Col, Button, Typography, Alert, Tabs, Tag, List, Divider, Table, Statistic, Image } from 'antd';
import { SettingOutlined, SafetyOutlined, ExperimentOutlined, BarChartOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import { apiService } from '../services/apiService';

const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;

const ImageContainer = styled.div`
  text-align: center;
  margin: 20px 0;
  
  img {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
`;

const StyledCard = styled(Card)`
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
`;

const FeatureEngineering = ({ dataLoaded }) => {
  const [loading, setLoading] = useState(false);
  const [temporalFeatures, setTemporalFeatures] = useState(null);
  const [lagFeatures, setLagFeatures] = useState(null);
  const [rollingFeatures, setRollingFeatures] = useState(null);
  const [leakageDemo, setLeakageDemo] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);

  const createTemporalFeatures = async () => {
    setLoading(true);
    try {
      const response = await apiService.createTemporalFeatures();
      setTemporalFeatures(response);
    } catch (error) {
      console.error('Failed to create temporal features:', error);
    } finally {
      setLoading(false);
    }
  };

  const createLagFeatures = async () => {
    setLoading(true);
    try {
      const response = await apiService.createLagFeatures({ lags: [1, 7, 14, 30] });
      setLagFeatures(response);
    } catch (error) {
      console.error('Failed to create lag features:', error);
    } finally {
      setLoading(false);
    }
  };

  const createRollingFeatures = async () => {
    setLoading(true);
    try {
      const response = await apiService.createRollingFeatures({ windows: [7, 14, 30] });
      setRollingFeatures(response);
    } catch (error) {
      console.error('Failed to create rolling features:', error);
    } finally {
      setLoading(false);
    }
  };

  const demonstrateLeakagePrevention = async () => {
    setLoading(true);
    try {
      const response = await apiService.getLeakageDemo();
      setLeakageDemo(response);
    } catch (error) {
      console.error('Failed to load leakage demonstration:', error);
    } finally {
      setLoading(false);
    }
  };

  const analyzeFeatureImportance = async () => {
    setLoading(true);
    try {
      const response = await apiService.analyzeFeatureImportance();
      setFeatureImportance(response);
    } catch (error) {
      console.error('Failed to analyze feature importance:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!dataLoaded) {
    return (
      <Alert
        message="Data Required"
        description="Please load the data first from the Dashboard to perform feature engineering."
        type="warning"
        showIcon
      />
    );
  }

  return (
    <div>
      <Title level={2}>🔧 Feature Engineering</Title>
      <Paragraph>
        Advanced feature engineering techniques for hotel revenue forecasting with strict data leakage prevention.
      </Paragraph>

      <Tabs defaultActiveKey="temporal">
        <TabPane tab="Temporal Features" key="temporal">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<ExperimentOutlined />}
                onClick={createTemporalFeatures}
                loading={loading}
              >
                Create Temporal Features
              </Button>
            </div>
            
            {temporalFeatures && (
              <Row gutter={[24, 24]}>
                <Col xs={24} lg={12}>
                  <Card title="Features Created" size="small">
                    <p><strong>Total Features:</strong> {temporalFeatures.feature_count}</p>
                    <Divider />
                    <div style={{ marginBottom: '16px' }}>
                      <Text strong>Cyclical Encoding:</Text>
                      <div>
                        {temporalFeatures.encoding_techniques?.cyclical_encoding?.map(feature => (
                          <Tag key={feature} color="blue">{feature}</Tag>
                        ))}
                      </div>
                    </div>
                    <div style={{ marginBottom: '16px' }}>
                      <Text strong>Binary Encoding:</Text>
                      <div>
                        {temporalFeatures.encoding_techniques?.binary_encoding?.map(feature => (
                          <Tag key={feature} color="green">{feature}</Tag>
                        ))}
                      </div>
                    </div>
                    <div>
                      <Text strong>Categorical Encoding:</Text>
                      <div>
                        {temporalFeatures.encoding_techniques?.categorical_encoding?.map(feature => (
                          <Tag key={feature} color="orange">{feature}</Tag>
                        ))}
                      </div>
                    </div>
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card title="Sample Feature Values" size="small">
                    <List size="small">
                      <List.Item>
                        <Text strong>Month Sin Sample:</Text> {temporalFeatures.sample_features?.Month_sin_sample?.slice(0, 3).join(', ')}...
                      </List.Item>
                      <List.Item>
                        <Text strong>DayOfWeek Cos Sample:</Text> {temporalFeatures.sample_features?.DayOfWeek_cos_sample?.slice(0, 3).join(', ')}...
                      </List.Item>
                      <List.Item>
                        <Text strong>IsWeekend Sample:</Text> {temporalFeatures.sample_features?.IsWeekend_sample?.slice(0, 5).join(', ')}
                      </List.Item>
                    </List>
                  </Card>
                </Col>
                <Col span={24}>
                  <Alert
                    message="Cyclical Encoding Benefits"
                    description="Temporal features use sine/cosine encoding to properly represent cyclical patterns (months, days) ensuring the model understands that December is close to January."
                    type="info"
                  />
                </Col>
              </Row>
            )}
          </Card>
        </TabPane>

        <TabPane tab="Lag Features" key="lag">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<ExperimentOutlined />}
                onClick={createLagFeatures}
                loading={loading}
              >
                Create Lag Features
              </Button>
            </div>
            
            {lagFeatures && (
              <Row gutter={[24, 24]}>
                <Col xs={24} lg={12}>
                  <Card title="Lag Configuration" size="small">
                    <p><strong>Lags Created:</strong> {lagFeatures.lags_created?.join(', ')} days</p>
                    <p><strong>Meal Periods:</strong> {lagFeatures.meal_periods?.join(', ')}</p>
                    <p><strong>Total Lag Features:</strong> {lagFeatures.total_lag_features}</p>
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card title="Leakage Prevention" size="small">
                    <Alert
                      message={lagFeatures.leakage_prevention?.method}
                      description={lagFeatures.leakage_prevention?.explanation}
                      type="success"
                      size="small"
                    />
                  </Card>
                </Col>
              </Row>
            )}
          </Card>
        </TabPane>

        <TabPane tab="Rolling Features" key="rolling">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<ExperimentOutlined />}
                onClick={createRollingFeatures}
                loading={loading}
              >
                Create Rolling Features
              </Button>
            </div>
            
            {rollingFeatures && (
              <Row gutter={[24, 24]}>
                <Col xs={24} lg={12}>
                  <Card title="Rolling Windows" size="small">
                    <p><strong>Windows:</strong> {rollingFeatures.windows_created?.join(', ')} days</p>
                    <p><strong>Feature Types:</strong> {rollingFeatures.feature_types?.join(', ')}</p>
                    <p><strong>Total Features:</strong> {rollingFeatures.total_rolling_features}</p>
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card title="Safe Implementation" size="small">
                    <Alert
                      message={rollingFeatures.leakage_prevention?.method}
                      description={rollingFeatures.leakage_prevention?.explanation}
                      type="success"
                      size="small"
                    />
                  </Card>
                </Col>
              </Row>
            )}
          </Card>
        </TabPane>

        <TabPane tab="Leakage Prevention" key="leakage">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<SafetyOutlined />}
                onClick={demonstrateLeakagePrevention}
                loading={loading}
                danger
              >
                Demonstrate Leakage Prevention
              </Button>
            </div>
            
            {leakageDemo && (
              <Row gutter={[24, 24]}>
                <Col xs={24} lg={12}>
                  <Card title="🚨 Leaky Examples (DON'T DO THIS)" size="small">
                    {Object.entries(leakageDemo.leaky_examples || {}).map(([feature, info]) => (
                      <div key={feature} style={{ marginBottom: '16px' }}>
                        <Text strong style={{ color: '#ff4d4f' }}>{feature}</Text>
                        <p>Correlation: {info.correlation_with_target?.toFixed(3)}</p>
                        <p style={{ fontSize: '12px', color: '#666' }}>{info.why_leaky}</p>
                        <Tag color="red">{info.red_flag}</Tag>
                      </div>
                    ))}
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card title="✅ Safe Examples (CORRECT WAY)" size="small">
                    {Object.entries(leakageDemo.safe_examples || {}).map(([feature, info]) => (
                      <div key={feature} style={{ marginBottom: '16px' }}>
                        <Text strong style={{ color: '#52c41a' }}>{feature}</Text>
                        <p>Correlation: {info.correlation_with_target?.toFixed(3)}</p>
                        <p style={{ fontSize: '12px', color: '#666' }}>{info.why_safe}</p>
                        <Tag color="green">{info.validation}</Tag>
                      </div>
                    ))}
                  </Card>
                </Col>
                <Col span={24}>
                  <Card title="Prevention Techniques" size="small">
                    <Row gutter={[16, 16]}>
                      {Object.entries(leakageDemo.prevention_techniques || {}).map(([technique, description]) => (
                        <Col xs={24} sm={12} key={technique}>
                          <div style={{ padding: '8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}>
                            <Text strong>{technique.replace(/_/g, ' ').toUpperCase()}</Text>
                            <p style={{ fontSize: '12px', margin: '4px 0 0 0' }}>{description}</p>
                          </div>
                        </Col>
                      ))}
                    </Row>
                  </Card>
                </Col>
                <Col span={24}>
                  <Alert
                    message="🛡️ Data Leakage Prevention is Critical"
                    description="Proper feature engineering ensures models learn realistic patterns that will generalize to new data, not artifacts from seeing the future."
                    type="warning"
                    showIcon
                  />
                </Col>
              </Row>
            )}
          </Card>
        </TabPane>

        <TabPane tab={<span><BarChartOutlined />Feature Importance</span>} key="importance">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<BarChartOutlined />}
                onClick={analyzeFeatureImportance}
                loading={loading}
              >
                Analyze Feature Importance for Revenue Prediction
              </Button>
            </div>
            
            {featureImportance && (
              <Row gutter={[24, 24]}>
                <Col span={24}>
                  <StyledCard title="Feature Importance Analysis for CheckTotal Prediction">
                    <ImageContainer>
                      <Image
                        src={`data:image/png;base64,${featureImportance.feature_importance_plot}`}
                        alt="Feature Importance Analysis"
                        preview={{
                          mask: 'Click to view full size'
                        }}
                      />
                    </ImageContainer>
                  </StyledCard>
                </Col>

                <Col span={24}>
                  <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                    <Col span={6}>
                      <StyledCard>
                        <Statistic
                          title="Model Performance (R²)"
                          value={featureImportance.model_performance?.r2_score}
                          precision={3}
                          valueStyle={{ color: '#1890ff' }}
                        />
                      </StyledCard>
                    </Col>
                    <Col span={6}>
                      <StyledCard>
                        <Statistic
                          title="Features Analyzed"
                          value={featureImportance.model_performance?.features_count}
                          valueStyle={{ color: '#52c41a' }}
                        />
                      </StyledCard>
                    </Col>
                    <Col span={6}>
                      <StyledCard>
                        <Statistic
                          title="Data Points"
                          value={featureImportance.model_performance?.data_points}
                          valueStyle={{ color: '#722ed1' }}
                        />
                      </StyledCard>
                    </Col>
                    <Col span={6}>
                      <StyledCard>
                        <Statistic
                          title="Model Quality"
                          value={featureImportance.analysis_summary?.model_quality}
                          valueStyle={{ color: '#fa8c16' }}
                        />
                      </StyledCard>
                    </Col>
                  </Row>
                </Col>

                <Col xs={24} lg={12}>
                  <StyledCard title="Top 10 Most Important Features">
                    <Table
                      size="small"
                      dataSource={Object.entries(featureImportance.top_features || {}).map(([feature, importance], index) => ({
                        key: index,
                        rank: index + 1,
                        feature,
                        importance
                      }))}
                      columns={[
                        { title: 'Rank', dataIndex: 'rank', key: 'rank', width: 60 },
                        { title: 'Feature', dataIndex: 'feature', key: 'feature' },
                        { 
                          title: 'Importance', 
                          dataIndex: 'importance', 
                          key: 'importance',
                          render: (value) => (
                            <div>
                              <div style={{ 
                                width: `${value * 300}px`, 
                                height: '12px', 
                                backgroundColor: '#1890ff', 
                                borderRadius: '6px',
                                marginBottom: '4px'
                              }} />
                              <Text style={{ fontSize: '11px' }}>{value.toFixed(4)}</Text>
                            </div>
                          )
                        }
                      ]}
                      pagination={false}
                    />
                  </StyledCard>
                </Col>

                <Col xs={24} lg={12}>
                  <StyledCard title="Feature Categories">
                    {Object.entries(featureImportance.feature_categories || {}).map(([category, features]) => (
                      features.length > 0 && (
                        <div key={category} style={{ marginBottom: '16px' }}>
                          <Text strong style={{ textTransform: 'capitalize' }}>{category} Features:</Text>
                          <div style={{ marginTop: '8px' }}>
                            {features.slice(0, 5).map(feature => (
                              <Tag key={feature} color="blue" style={{ marginBottom: '4px' }}>
                                {feature}
                              </Tag>
                            ))}
                            {features.length > 5 && (
                              <Tag color="default">+{features.length - 5} more</Tag>
                            )}
                          </div>
                        </div>
                      )
                    ))}
                  </StyledCard>
                </Col>

                <Col span={24}>
                  <Alert
                    message="🎯 Feature Importance Insights"
                    description={`The Random Forest model identified "${featureImportance.analysis_summary?.top_feature}" as the most important feature for predicting revenue, with an importance score of ${featureImportance.analysis_summary?.top_importance?.toFixed(4)}. This analysis helps understand which variables have the strongest predictive power for CheckTotal.`}
                    type="info"
                    showIcon
                  />
                </Col>
              </Row>
            )}
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default FeatureEngineering; 