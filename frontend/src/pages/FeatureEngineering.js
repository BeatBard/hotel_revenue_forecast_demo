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
  const [correlationAnalysis, setCorrelationAnalysis] = useState(null);
  const [featureDropResults, setFeatureDropResults] = useState(null);

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

  const analyzeFeatureCorrelations = async () => {
    setLoading(true);
    try {
      const response = await apiService.analyzeFeatureCorrelations();
      setCorrelationAnalysis(response);
    } catch (error) {
      console.error('Failed to analyze feature correlations:', error);
    } finally {
      setLoading(false);
    }
  };

  const dropLowCorrelationFeatures = async () => {
    setLoading(true);
    try {
      const response = await apiService.dropLowCorrelationFeatures({
        threshold: correlationAnalysis?.correlation_threshold || 0.01
      });
      setFeatureDropResults(response);
    } catch (error) {
      console.error('Failed to drop features:', error);
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

        <TabPane tab={<span>📊 Feature Analysis & Importance</span>} key="analysis">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<BarChartOutlined />}
                onClick={analyzeFeatureImportance}
                loading={loading}
                style={{ marginRight: '16px' }}
              >
                Analyze Feature Importance
              </Button>
              <Button 
                type="primary" 
                icon={<BarChartOutlined />}
                onClick={analyzeFeatureCorrelations}
                loading={loading}
              >
                Analyze Feature Correlations
              </Button>
            </div>

            {correlationAnalysis && (
              <div style={{ marginBottom: '24px' }}>
                <StyledCard title="Feature Correlation Analysis">
                  <Row gutter={[24, 24]}>
                    <Col span={24}>
                      <div style={{ marginBottom: '16px' }}>
                        <Button 
                          style={{ backgroundColor: '#fadb14', borderColor: '#fadb14', color: '#000' }}
                          icon={<BarChartOutlined />}
                          onClick={dropLowCorrelationFeatures}
                          loading={loading}
                          disabled={!correlationAnalysis}
                        >
                          Drop Low Correlation Features Now
                        </Button>
                        <span style={{ marginLeft: '16px', color: '#666' }}>
                          This will drop {correlationAnalysis?.summary?.low_correlation_count || 0} features with correlation &lt; {correlationAnalysis?.correlation_threshold || 0.01}
                        </span>
                      </div>
                    </Col>

                    <Col span={24}>
                      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                        <Col span={6}>
                          <StyledCard>
                            <Statistic
                              title="Total Features"
                              value={correlationAnalysis.total_features}
                              valueStyle={{ color: '#1890ff' }}
                            />
                          </StyledCard>
                        </Col>
                        <Col span={6}>
                          <StyledCard>
                            <Statistic
                              title="High Correlation Features"
                              value={correlationAnalysis.summary?.high_correlation_count}
                              valueStyle={{ color: '#52c41a' }}
                            />
                          </StyledCard>
                        </Col>
                        <Col span={6}>
                          <StyledCard>
                            <Statistic
                              title="Low Correlation Features"
                              value={correlationAnalysis.summary?.low_correlation_count}
                              valueStyle={{ color: '#fa541c' }}
                            />
                          </StyledCard>
                        </Col>
                        <Col span={6}>
                          <StyledCard>
                            <Statistic
                              title="Removal Percentage"
                              value={correlationAnalysis.summary?.removal_percentage}
                              suffix="%"
                              valueStyle={{ color: '#722ed1' }}
                            />
                          </StyledCard>
                        </Col>
                      </Row>
                    </Col>

                    <Col xs={24} lg={12}>
                      <StyledCard title="Features to Keep (High Correlation)" size="small">
                        <Table
                          size="small"
                          dataSource={correlationAnalysis.high_correlation_features?.slice(0, 10).map((item, index) => ({
                            key: index,
                            rank: index + 1,
                            feature: item.feature,
                            correlation: item.correlation
                          }))}
                          columns={[
                            { title: 'Rank', dataIndex: 'rank', key: 'rank', width: 60 },
                            { title: 'Feature', dataIndex: 'feature', key: 'feature' },
                            { 
                              title: 'Correlation', 
                              dataIndex: 'correlation', 
                              key: 'correlation',
                              render: (value) => (
                                <div>
                                  <div style={{ 
                                    width: `${value * 300}px`, 
                                    height: '8px', 
                                    backgroundColor: '#52c41a', 
                                    borderRadius: '4px',
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
                      <StyledCard title="Features to Drop (Low Correlation)" size="small">
                        <Table
                          size="small"
                          dataSource={correlationAnalysis.low_correlation_features?.slice(0, 10).map((item, index) => ({
                            key: index,
                            feature: item.feature,
                            correlation: item.correlation
                          }))}
                          columns={[
                            { title: 'Feature', dataIndex: 'feature', key: 'feature' },
                            { 
                              title: 'Correlation', 
                              dataIndex: 'correlation', 
                              key: 'correlation',
                              render: (value) => (
                                <div>
                                  <div style={{ 
                                    width: `${value * 300}px`, 
                                    height: '8px', 
                                    backgroundColor: '#fa541c', 
                                    borderRadius: '4px',
                                    marginBottom: '4px'
                                  }} />
                                  <Text style={{ fontSize: '11px', color: '#fa541c' }}>{value.toFixed(4)}</Text>
                                </div>
                              )
                            }
                          ]}
                          pagination={false}
                        />
                      </StyledCard>
                    </Col>

                    <Col span={24}>
                      <Alert
                        message="🔍 Feature Selection Recommendation"
                        description={`Analysis shows ${correlationAnalysis.summary?.low_correlation_count} features have correlation < ${correlationAnalysis.correlation_threshold} with revenue. Removing these features will reduce model complexity by ${correlationAnalysis.summary?.removal_percentage}% and help prevent overfitting.`}
                        type="warning"
                        showIcon
                        style={{ marginBottom: 16 }}
                      />
                    </Col>

                    {featureDropResults && (
                      <Col span={24}>
                        <StyledCard title="🗑️ Feature Dropping Results" style={{ marginTop: 16 }}>
                          <Row gutter={[16, 16]}>
                            <Col span={6}>
                              <Statistic
                                title="Original Features"
                                value={featureDropResults.original_feature_count}
                                valueStyle={{ color: '#fa541c' }}
                              />
                            </Col>
                            <Col span={6}>
                              <Statistic
                                title="Features Dropped"
                                value={featureDropResults.dropped_feature_count}
                                valueStyle={{ color: '#f5222d' }}
                              />
                            </Col>
                            <Col span={6}>
                              <Statistic
                                title="Features Remaining"
                                value={featureDropResults.remaining_feature_count}
                                valueStyle={{ color: '#52c41a' }}
                              />
                            </Col>
                            <Col span={6}>
                              <Statistic
                                title="Reduction"
                                value={featureDropResults.reduction_percentage}
                                suffix="%"
                                valueStyle={{ color: '#722ed1' }}
                              />
                            </Col>
                          </Row>
                        </StyledCard>
                      </Col>
                    )}
                  </Row>
                </StyledCard>
              </div>
            )}
            
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