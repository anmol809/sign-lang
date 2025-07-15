import React from 'react';
import { Play, Zap, Shield, Globe } from 'lucide-react';

const Hero: React.FC = () => {
  const scrollToDetection = () => {
    document.getElementById('detection')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="relative py-20 overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary-50/50 to-accent-50/30"></div>
      <div className="absolute top-20 left-10 w-72 h-72 bg-primary-200/20 rounded-full blur-3xl"></div>
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-accent-200/20 rounded-full blur-3xl"></div>
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <div className="inline-flex items-center space-x-2 bg-primary-100 text-primary-800 px-4 py-2 rounded-full text-sm font-medium mb-8 animate-fade-in">
            <Zap className="w-4 h-4" />
            <span>Powered by Advanced AI & Computer Vision</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6 animate-slide-up">
            Real-time
            <span className="bg-gradient-to-r from-primary-600 to-accent-600 bg-clip-text text-transparent block">
              Sign Language
            </span>
            Detection
          </h1>
          
          <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed animate-fade-in">
            Experience cutting-edge AI technology that recognizes sign language gestures in real-time. 
            Our advanced neural network model provides instant, accurate detection through your webcam.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
            <button 
              onClick={scrollToDetection}
              className="btn-primary flex items-center space-x-2 text-lg"
            >
              <Play className="w-5 h-5" />
              <span>Start Detection</span>
            </button>
            <button className="btn-secondary flex items-center space-x-2 text-lg">
              <Shield className="w-5 h-5" />
              <span>Learn More</span>
            </button>
          </div>
          
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="glass-effect rounded-2xl p-6 text-center">
              <div className="text-3xl font-bold text-primary-600 mb-2">99.2%</div>
              <div className="text-gray-600">Accuracy Rate</div>
            </div>
            <div className="glass-effect rounded-2xl p-6 text-center">
              <div className="text-3xl font-bold text-accent-600 mb-2">&lt;50ms</div>
              <div className="text-gray-600">Response Time</div>
            </div>
            <div className="glass-effect rounded-2xl p-6 text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">2+</div>
              <div className="text-gray-600">Gestures Supported</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;