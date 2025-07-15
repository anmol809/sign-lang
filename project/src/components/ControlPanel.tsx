import React from 'react';
import { Settings, RotateCcw, Download, Info } from 'lucide-react';
import { useSignLanguage } from '../context/SignLanguageContext';

const ControlPanel: React.FC = () => {
  const { 
    isDetecting, 
    predictionHistory, 
    clearHistory, 
    modelLoaded,
    modelLoadingProgress 
  } = useSignLanguage();

  const downloadHistory = () => {
    const data = JSON.stringify(predictionHistory, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sign-language-detections-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="glass-effect rounded-2xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Controls</h3>
        <Settings className="w-5 h-5 text-gray-500" />
      </div>
      
      {/* Model Status */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">AI Model Status</span>
          <div className={`w-3 h-3 rounded-full ${modelLoaded ? 'bg-accent-500' : 'bg-yellow-500'}`}></div>
        </div>
        {!modelLoaded ? (
          <div className="space-y-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-primary-500 to-accent-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${modelLoadingProgress}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-500">Loading model... {modelLoadingProgress.toFixed(0)}%</p>
          </div>
        ) : (
          <p className="text-sm text-accent-600 font-medium">Ready for detection</p>
        )}
      </div>
      
      {/* Statistics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-primary-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-primary-600">{predictionHistory.length}</div>
          <div className="text-xs text-gray-600">Total Detections</div>
        </div>
        <div className="bg-accent-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-accent-600">
            {predictionHistory.length > 0 
              ? (predictionHistory.reduce((acc, p) => acc + p.confidence, 0) / predictionHistory.length * 100).toFixed(0)
              : 0}%
          </div>
          <div className="text-xs text-gray-600">Avg. Confidence</div>
        </div>
      </div>
      
      {/* Action Buttons */}
      <div className="space-y-3">
        <button
          onClick={clearHistory}
          disabled={predictionHistory.length === 0}
          className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 disabled:text-gray-400 text-gray-700 rounded-xl transition-colors font-medium"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Clear History</span>
        </button>
        
        <button
          onClick={downloadHistory}
          disabled={predictionHistory.length === 0}
          className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-primary-100 hover:bg-primary-200 disabled:bg-gray-50 disabled:text-gray-400 text-primary-700 rounded-xl transition-colors font-medium"
        >
          <Download className="w-4 h-4" />
          <span>Export Data</span>
        </button>
      </div>
      
      {/* Info */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <div className="flex items-start space-x-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-blue-800">
            <p className="font-medium mb-1">Detection Tips:</p>
            <ul className="text-xs space-y-1 text-blue-700">
              <li>• Ensure good lighting</li>
              <li>• Keep hand clearly visible</li>
              <li>• Maintain steady position</li>
              <li>• Face the camera directly</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;