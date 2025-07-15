import React from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import DetectionInterface from './components/DetectionInterface';
import Features from './components/Features';
import Footer from './components/Footer';
import { SignLanguageProvider } from './context/SignLanguageContext';

function App() {
  return (
    <SignLanguageProvider>
      <div className="min-h-screen">
        <Header />
        <main>
          <Hero />
          <DetectionInterface />
          <Features />
        </main>
        <Footer />
      </div>
    </SignLanguageProvider>
  );
}

export default App;