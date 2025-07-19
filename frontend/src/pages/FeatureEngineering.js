import React, { useState } from 'react';
import { Card, Row, Col, Button, Typography, Alert, Tabs, Tag, List, Divider } from 'antd';
import { SettingOutlined, SafetyOutlined, ExperimentOutlined } from '@ant-design/icons';
import { apiService } from '../services/apiService';

const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;

const FeatureEngineering = ({ dataLoaded }) => {
  const [loading, setLoading] = useState(false);
  const [temporalFeatures, setTemporalFeatures] = useState(null);
  const [lagFeatures, setLagFeatures] = useState(null);
  const [rollingFeatures, setRollingFeatures] = useState(null);
  const [leakageDemo, setLeakageDemo] = useState(null);

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
      </Tabs>
    </div>
  );
};

export default FeatureEngineering; 