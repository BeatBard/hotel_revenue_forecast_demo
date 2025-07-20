import React, { useState } from 'react';
import { Card, Row, Col, Button, Typography, Alert, Tabs, Progress, List, Tag, Statistic, message } from 'antd';
import { RobotOutlined, ThunderboltOutlined, TrophyOutlined } from '@ant-design/icons';
import { apiService } from '../services/apiService';

const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;

const ModelTraining = ({ dataLoaded }) => {
  const [loading, setLoading] = useState(false);
  const [individualResults, setIndividualResults] = useState(null);
  const [ensembleResults, setEnsembleResults] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);

  const trainIndividualModels = async () => {
    setLoading(true);
    try {
      const response = await apiService.trainIndividualModels({
        models: ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
      });
      setIndividualResults(response);
    } catch (error) {
      console.error('Failed to train individual models:', error);
    } finally {
      setLoading(false);
    }
  };

  const trainEnsembleModels = async () => {
    setLoading(true);
    try {
      const response = await apiService.trainEnsembleModels({
        ensembles: ['simple', 'weighted', 'top3', 'median']
      });
      setEnsembleResults(response);
    } catch (error) {
      console.error('Failed to train ensemble models:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadFeatureImportance = async (model = 'xgboost') => {
    try {
      const response = await apiService.getModelFeatureImportance(model);
      
      // Check if there's an error in the response
      if (response.error) {
        message.error(`Feature Importance Error: ${response.error}`);
        setFeatureImportance(null);
        return;
      }
      
      setFeatureImportance(response);
      message.success(`Feature importance loaded for ${model.toUpperCase()}`);
    } catch (error) {
      console.error('Failed to load feature importance:', error);
      message.error('Failed to load feature importance. Please ensure models are trained first.');
      setFeatureImportance(null);
    }
  };

  const getPerformanceColor = (r2) => {
    if (r2 > 0.4) return '#52c41a'; // Green
    if (r2 > 0.3) return '#faad14'; // Yellow
    return '#ff4d4f'; // Red
  };

  if (!dataLoaded) {
    return (
      <Alert
        message="Data Required"
        description="Please load the data first from the Dashboard to train models."
        type="warning"
        showIcon
      />
    );
  }

  return (
    <div>
      <Title level={2}>🤖 Model Training</Title>
      <Paragraph>
        Train individual base models and combine them using ensemble strategies for superior performance.
      </Paragraph>

      <Tabs defaultActiveKey="individual">
        <TabPane tab="Individual Models" key="individual">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<RobotOutlined />}
                onClick={trainIndividualModels}
                loading={loading}
                size="large"
              >
                Train 5 Base Models
              </Button>
            </div>
            
            {individualResults && (
              <Row gutter={[24, 24]}>
                <Col span={24}>
                  <Alert
                    message="Training Completed Successfully"
                    description={`Trained ${individualResults.models_trained?.length} models using ${individualResults.features_used} features on ${individualResults.training_samples} training samples.`}
                    type="success"
                    style={{ marginBottom: '24px' }}
                  />
                </Col>
                
                {Object.entries(individualResults.performance || {}).map(([modelName, metrics]) => (
                  <Col xs={24} sm={12} lg={8} key={modelName}>
                    <Card 
                      title={modelName.replace(/_/g, ' ').toUpperCase()}
                      size="small"
                    >
                      <Row gutter={[8, 8]}>
                        <Col span={24}>
                          <Statistic
                            title="R² Score"
                            value={metrics.r2}
                            precision={3}
                            valueStyle={{ color: getPerformanceColor(metrics.r2) }}
                            suffix={
                              <Progress
                                percent={(metrics.r2 * 100).toFixed(1)}
                                size="small"
                                status={metrics.r2 > 0.4 ? 'success' : metrics.r2 > 0.3 ? 'normal' : 'exception'}
                                style={{ width: '60px' }}
                              />
                            }
                          />
                        </Col>
                        <Col span={12}>
                          <div>
                            <Text strong>MAE:</Text>
                            <br />
                            <Text>${metrics.mae?.toFixed(0)}</Text>
                          </div>
                        </Col>
                        <Col span={12}>
                          <div>
                            <Text strong>RMSE:</Text>
                            <br />
                            <Text>${metrics.rmse?.toFixed(0)}</Text>
                          </div>
                        </Col>
                      </Row>
                    </Card>
                  </Col>
                ))}
                
                <Col span={24}>
                  <Card title="Model Characteristics" size="small">
                    <Row gutter={[16, 16]}>
                      <Col xs={24} sm={12} lg={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Tag color="blue">Ridge Regression</Tag>
                          <p>Linear model with L2 regularization</p>
                        </div>
                      </Col>
                      <Col xs={24} sm={12} lg={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Tag color="green">Random Forest</Tag>
                          <p>Ensemble of decision trees</p>
                        </div>
                      </Col>
                      <Col xs={24} sm={12} lg={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Tag color="orange">XGBoost</Tag>
                          <p>Gradient boosting framework</p>
                        </div>
                      </Col>
                      <Col xs={24} sm={12} lg={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Tag color="purple">LightGBM</Tag>
                          <p>Fast gradient boosting</p>
                        </div>
                      </Col>
                      <Col xs={24} sm={12} lg={6}>
                        <div style={{ textAlign: 'center' }}>
                          <Tag color="cyan">Gradient Boosting</Tag>
                          <p>Sequential ensemble method</p>
                        </div>
                      </Col>
                    </Row>
                  </Card>
                </Col>
              </Row>
            )}
          </Card>
        </TabPane>

        <TabPane tab="Ensemble Methods" key="ensemble">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<ThunderboltOutlined />}
                onClick={trainEnsembleModels}
                loading={loading}
                size="large"
                disabled={!individualResults}
              >
                Create Ensemble Models
              </Button>
              {!individualResults && (
                <p style={{ marginTop: '8px', color: '#666' }}>
                  Train individual models first
                </p>
              )}
            </div>
            
            {ensembleResults && (
              <Row gutter={[24, 24]}>
                <Col span={24}>
                  <Alert
                    message="Ensemble Training Completed"
                    description={`Created ${ensembleResults.ensemble_strategies?.length} ensemble strategies. Best ensemble: ${ensembleResults.best_ensemble?.[0]} with R² = ${ensembleResults.best_ensemble?.[1]?.r2?.toFixed(3)}`}
                    type="success"
                    style={{ marginBottom: '24px' }}
                  />
                </Col>
                
                {Object.entries(ensembleResults.performance || {}).map(([strategyName, metrics]) => (
                  <Col xs={24} sm={12} lg={6} key={strategyName}>
                    <Card 
                      title={strategyName.replace(/_/g, ' ').toUpperCase()}
                      size="small"
                    >
                      <Statistic
                        title="R² Score"
                        value={metrics.r2}
                        precision={3}
                        valueStyle={{ color: getPerformanceColor(metrics.r2) }}
                      />
                      <div style={{ marginTop: '8px' }}>
                        <Text strong>MAE:</Text> ${metrics.mae?.toFixed(0)}
                        <br />
                        <Text strong>RMSE:</Text> ${metrics.rmse?.toFixed(0)}
                      </div>
                      {metrics.weights && (
                        <div style={{ marginTop: '8px' }}>
                          <Text strong>Weights:</Text>
                          <div>
                            {Object.entries(metrics.weights).map(([model, weight]) => (
                              <Tag key={model} color="blue">
                                {model}: {(weight * 100).toFixed(1)}%
                              </Tag>
                            ))}
                          </div>
                        </div>
                      )}
                      {metrics.models_used && (
                        <div style={{ marginTop: '8px' }}>
                          <Text strong>Models Used:</Text>
                          <div>
                            {metrics.models_used.map(model => (
                              <Tag key={model} color="green">{model}</Tag>
                            ))}
                          </div>
                        </div>
                      )}
                    </Card>
                  </Col>
                ))}
                
                <Col span={24}>
                  <Card title="Ensemble Strategies" size="small">
                    <List
                      itemLayout="horizontal"
                      dataSource={[
                        {
                          title: 'Simple Average',
                          description: 'Equal weight combination of all models'
                        },
                        {
                          title: 'Weighted Average',
                          description: 'Performance-based weights using validation R² scores'
                        },
                        {
                          title: 'Top-3 Average',
                          description: 'Average of the three best performing models'
                        },
                        {
                          title: 'Median Ensemble',
                          description: 'Robust median of all model predictions'
                        }
                      ]}
                      renderItem={item => (
                        <List.Item>
                          <List.Item.Meta
                            title={item.title}
                            description={item.description}
                          />
                        </List.Item>
                      )}
                    />
                  </Card>
                </Col>
              </Row>
            )}
          </Card>
        </TabPane>

        <TabPane tab="Feature Importance" key="importance">
          <Card>
            {!individualResults && (
              <Alert
                message="Models Required"
                description="Please train individual models first to analyze feature importance."
                type="info"
                style={{ marginBottom: '16px' }}
                showIcon
              />
            )}
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<TrophyOutlined />}
                onClick={() => loadFeatureImportance('xgboost')}
                loading={loading}
                disabled={!individualResults}
              >
                Analyze Feature Importance
              </Button>
            </div>
            
            {featureImportance && (
              <Row gutter={[24, 24]}>
                <Col span={24}>
                  <Card title="Top 10 Most Important Features" size="small">
                    <List
                      dataSource={featureImportance.sorted_features?.slice(0, 10)}
                      renderItem={([feature, importance], index) => (
                        <List.Item>
                          <div style={{ width: '100%' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Text strong>{index + 1}. {feature}</Text>
                              <Text>{importance.toFixed(4)}</Text>
                            </div>
                            <Progress 
                              percent={(importance * 100 / featureImportance.sorted_features[0][1]).toFixed(1)} 
                              size="small"
                              showInfo={false}
                            />
                          </div>
                        </List.Item>
                      )}
                    />
                  </Card>
                </Col>
              </Row>
            )}
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default ModelTraining; 