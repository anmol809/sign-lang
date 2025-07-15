import React from 'react';
import { Zap, Shield, Globe, Cpu, Eye, Clock } from 'lucide-react';

const Features: React.FC = () => {
  const features = [
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Real-time Processing",
      description: "Instant gesture recognition with sub-50ms latency for seamless interaction.",
      color: "from-yellow-500 to-orange-500"
    },
    {
      icon: <Eye className="w-8 h-8" />,
      title: "Computer Vision",
      description: "Advanced MediaPipe hand tracking with 21-point landmark detection.",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Cpu className="w-8 h-8" />,
      title: "Neural Network",
      description: "LSTM-based deep learning model trained on thousands of gesture samples.",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Privacy First",
      description: "All processing happens locally in your browser. No data leaves your device.",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: <Globe className="w-8 h-8" />,
      title: "Cross-platform",
      description: "Works on any modern browser across desktop, tablet, and mobile devices.",
      color: "from-indigo-500 to-blue-500"
    },
    {
      icon: <Clock className="w-8 h-8" />,
      title: "Continuous Learning",
      description: "Model improves over time with advanced training techniques and data augmentation.",
      color: "from-red-500 to-pink-500"
    }
  ];

  return (
    <section id="features" className="py-20 bg-gradient-to-b from-gray-50/50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Cutting-edge Technology
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Our sign language detection system combines the latest advances in computer vision, 
            machine learning, and web technologies to deliver an unparalleled user experience.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="glass-effect rounded-2xl p-8 hover:shadow-2xl transition-all duration-300 group"
            >
              <div className={`inline-flex p-3 rounded-xl bg-gradient-to-r ${feature.color} text-white mb-6 group-hover:scale-110 transition-transform duration-300`}>
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                {feature.title}
              </h3>
              <p className="text-gray-600 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
        
        {/* Technical Specifications */}
        <div className="mt-20 glass-effect rounded-2xl p-8">
          <h3 className="text-2xl font-semibold text-gray-900 mb-8 text-center">
            Technical Specifications
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-primary-600 mb-2">LSTM</div>
              <div className="text-sm text-gray-600">Neural Network Architecture</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-600 mb-2">21</div>
              <div className="text-sm text-gray-600">Hand Landmarks Tracked</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">30fps</div>
              <div className="text-sm text-gray-600">Processing Rate</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">WebGL</div>
              <div className="text-sm text-gray-600">Hardware Acceleration</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;