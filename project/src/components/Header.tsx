import React from 'react';
import { Hand, Github, Star } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="glass-effect sticky top-0 z-50 border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-br from-primary-500 to-primary-600 p-2 rounded-xl">
              <Hand className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-primary-600 to-primary-800 bg-clip-text text-transparent">
                SignDetect AI
              </h1>
              <p className="text-xs text-gray-500">Real-time Recognition</p>
            </div>
          </div>
          
          <nav className="hidden md:flex items-center space-x-8">
            <a href="#detection" className="text-gray-600 hover:text-primary-600 transition-colors font-medium">
              Detection
            </a>
            <a href="#features" className="text-gray-600 hover:text-primary-600 transition-colors font-medium">
              Features
            </a>
            <a href="#about" className="text-gray-600 hover:text-primary-600 transition-colors font-medium">
              About
            </a>
          </nav>
          
          <div className="flex items-center space-x-4">
            <button className="flex items-center space-x-2 text-gray-600 hover:text-primary-600 transition-colors">
              <Github className="w-5 h-5" />
              <span className="hidden sm:inline text-sm font-medium">GitHub</span>
            </button>
            <button className="flex items-center space-x-1 bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm font-medium hover:bg-yellow-200 transition-colors">
              <Star className="w-4 h-4" />
              <span>Star</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;