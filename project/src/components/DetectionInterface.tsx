import React from 'react';
import WebcamFeed from './WebcamFeed';
import PredictionDisplay from './PredictionDisplay';
import ControlPanel from './ControlPanel';
import { useSignLanguage } from '../context/SignLanguageContext';

const DetectionInterface: React.FC = () => {
  const { isDetecting } = useSignLanguage();

  return (
    <section id="detection" className="py-20 bg-gradient-to-b from-transparent to-gray-50/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Live Detection Interface
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Position your hand in front of the camera and watch as our AI recognizes your sign language gestures in real-time.
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Webcam Feed */}
          <div className="lg:col-span-2">
            <WebcamFeed />
          </div>
          
          {/* Sidebar */}
          <div className="space-y-6">
            <ControlPanel />
            <PredictionDisplay />
          </div>
        </div>
        
        {/* Instructions */}
        <div className="mt-12 glass-effect rounded-2xl p-8">
          <h3 className="text-2xl font-semibold text-gray-900 mb-6 text-center">
            How to Use
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="bg-primary-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-primary-600 font-bold">1</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Enable Camera</h4>
              <p className="text-gray-600 text-sm">Click "Start Detection" to activate your webcam</p>
            </div>
            <div className="text-center">
              <div className="bg-accent-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-accent-600 font-bold">2</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Position Hand</h4>
              <p className="text-gray-600 text-sm">Place your hand clearly in the camera view</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-purple-600 font-bold">3</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">See Results</h4>
              <p className="text-gray-600 text-sm">Watch real-time predictions appear instantly</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DetectionInterface;