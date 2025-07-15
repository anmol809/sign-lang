import React from 'react';
import { TrendingUp, Clock, Target } from 'lucide-react';
import { useSignLanguage } from '../context/SignLanguageContext';

const PredictionDisplay: React.FC = () => {
  const { currentPrediction, predictionHistory, confidence } = useSignLanguage();

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'text-accent-600 bg-accent-100';
    if (conf >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getConfidenceText = (conf: number) => {
    if (conf >= 0.8) return 'High';
    if (conf >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <div className="prediction-card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Live Predictions</h3>
        <TrendingUp className="w-5 h-5 text-primary-600" />
      </div>
      
      {/* Current Prediction */}
      <div className="mb-6">
        <div className="text-center p-6 bg-gradient-to-br from-primary-50 to-accent-50 rounded-xl border border-primary-100">
          {currentPrediction ? (
            <>
              <div className="text-3xl font-bold text-gray-900 mb-2 capitalize">
                {currentPrediction}
              </div>
              <div className="flex items-center justify-center space-x-2">
                <Target className="w-4 h-4 text-gray-500" />
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(confidence)}`}>
                  {getConfidenceText(confidence)} ({(confidence * 100).toFixed(1)}%)
                </span>
              </div>
            </>
          ) : (
            <div className="text-gray-500">
              <div className="text-xl mb-2">No gesture detected</div>
              <div className="text-sm">Show your hand to start detection</div>
            </div>
          )}
        </div>
      </div>
      
      {/* Confidence Meter */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Confidence</span>
          <span className="text-sm text-gray-500">{(confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-gradient-to-r from-primary-500 to-accent-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${confidence * 100}%` }}
          ></div>
        </div>
      </div>
      
      {/* Recent Predictions */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <Clock className="w-4 h-4 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">Recent Detections</span>
        </div>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {predictionHistory.length > 0 ? (
            predictionHistory.slice(-5).reverse().map((prediction, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="font-medium text-gray-900 capitalize">{prediction.gesture}</span>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-500">
                    {new Date(prediction.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getConfidenceColor(prediction.confidence)}`}>
                    {(prediction.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-4 text-gray-500 text-sm">
              No predictions yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionDisplay;