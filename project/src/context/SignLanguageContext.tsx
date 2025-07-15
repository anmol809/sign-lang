import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { SignLanguageDetector } from '../utils/SignLanguageDetector';

interface Prediction {
  gesture: string;
  confidence: number;
  timestamp: number;
}

interface SignLanguageContextType {
  isDetecting: boolean;
  currentPrediction: string | null;
  confidence: number;
  predictionHistory: Prediction[];
  modelLoaded: boolean;
  modelLoadingProgress: number;
  startDetection: () => void;
  stopDetection: () => void;
  processFrame: (imageData: ImageData) => void;
  clearHistory: () => void;
}

const SignLanguageContext = createContext<SignLanguageContextType | undefined>(undefined);

export const useSignLanguage = () => {
  const context = useContext(SignLanguageContext);
  if (!context) {
    throw new Error('useSignLanguage must be used within a SignLanguageProvider');
  }
  return context;
};

export const SignLanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [predictionHistory, setPredictionHistory] = useState<Prediction[]>([]);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelLoadingProgress, setModelLoadingProgress] = useState(0);
  
  const detectorRef = useRef<SignLanguageDetector | null>(null);
  const processingRef = useRef<boolean>(false);

  useEffect(() => {
    const initializeDetector = async () => {
      try {
        // Simulate gradual loading
        const progressInterval = setInterval(() => {
          setModelLoadingProgress(prev => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return 90;
            }
            return prev + Math.random() * 15;
          });
        }, 150);

        const newDetector = new SignLanguageDetector();
        await newDetector.initialize();
        
        detectorRef.current = newDetector;
        setModelLoaded(true);
        setModelLoadingProgress(100);
        clearInterval(progressInterval);
        
        console.log('Detector initialized successfully');
      } catch (error) {
        console.error('Failed to initialize detector:', error);
        setModelLoadingProgress(0);
      }
    };

    initializeDetector();

    return () => {
      if (detectorRef.current) {
        detectorRef.current.dispose();
      }
    };
  }, []);

  const startDetection = useCallback(() => {
    if (modelLoaded && detectorRef.current) {
      setIsDetecting(true);
      console.log('Detection started');
    }
  }, [modelLoaded]);

  const stopDetection = useCallback(() => {
    setIsDetecting(false);
    setCurrentPrediction(null);
    setConfidence(0);
    processingRef.current = false;
    console.log('Detection stopped');
  }, []);

  const processFrame = useCallback(async (imageData: ImageData) => {
    if (!detectorRef.current || !isDetecting || processingRef.current) {
      return;
    }

    processingRef.current = true;

    try {
      const result = await detectorRef.current.detectGesture(imageData);
      
      if (result && result.confidence > 0.5) {
        setCurrentPrediction(result.gesture);
        setConfidence(result.confidence);
        
        // Add to history if confidence is high enough and not a duplicate
        if (result.confidence > 0.7) {
          const newPrediction: Prediction = {
            gesture: result.gesture,
            confidence: result.confidence,
            timestamp: Date.now()
          };
          
          setPredictionHistory(prev => {
            // Avoid duplicate consecutive predictions within 2 seconds
            const lastPrediction = prev[prev.length - 1];
            if (lastPrediction && 
                lastPrediction.gesture === newPrediction.gesture && 
                newPrediction.timestamp - lastPrediction.timestamp < 2000) {
              return prev;
            }
            
            // Keep only last 15 predictions to avoid memory issues
            const updated = [...prev, newPrediction];
            return updated.slice(-15);
          });
        }
      } else {
        // Gradually reduce confidence when no gesture is detected
        setConfidence(prev => {
          const newConfidence = Math.max(0, prev - 0.05);
          if (newConfidence < 0.3) {
            setCurrentPrediction(null);
          }
          return newConfidence;
        });
      }
    } catch (error) {
      console.error('Error processing frame:', error);
    } finally {
      processingRef.current = false;
    }
  }, [isDetecting]);

  const clearHistory = useCallback(() => {
    setPredictionHistory([]);
  }, []);

  const value: SignLanguageContextType = {
    isDetecting,
    currentPrediction,
    confidence,
    predictionHistory,
    modelLoaded,
    modelLoadingProgress,
    startDetection,
    stopDetection,
    processFrame,
    clearHistory
  };

  return (
    <SignLanguageContext.Provider value={value}>
      {children}
    </SignLanguageContext.Provider>
  );
};