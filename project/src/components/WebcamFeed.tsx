import React, { useRef, useEffect, useState } from 'react';
import { Camera, CameraOff, AlertCircle } from 'lucide-react';
import { useSignLanguage } from '../context/SignLanguageContext';

const WebcamFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastProcessTimeRef = useRef<number>(0);
  
  const { isDetecting, startDetection, stopDetection, processFrame, currentPrediction, confidence } = useSignLanguage();
  const [error, setError] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  const PROCESS_INTERVAL = 300; // Process every 300ms

  // Initialize camera only once when detection starts
  useEffect(() => {
    const setupCamera = async () => {
      if (!isDetecting || streamRef.current) return;
      
      setIsLoading(true);
      setError('');
      setCameraReady(false);
      
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          }
        });
        
        streamRef.current = mediaStream;
        
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.onloadedmetadata = () => {
            setIsLoading(false);
            setCameraReady(true);
            startProcessing();
          };
        }
      } catch (err) {
        setError('Camera access denied. Please enable camera permissions.');
        setIsLoading(false);
        stopDetection();
      }
    };

    if (isDetecting && !streamRef.current) {
      setupCamera();
    }
  }, [isDetecting, stopDetection]);

  // Start processing frames
  const startProcessing = () => {
    if (animationFrameRef.current) return;

    const processFrames = () => {
      if (!isDetecting || !videoRef.current || !canvasRef.current || !cameraReady) {
        return;
      }
      
      const currentTime = Date.now();
      
      if (currentTime - lastProcessTimeRef.current >= PROCESS_INTERVAL) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        if (ctx && video.readyState === 4) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          processFrame(imageData);
          
          lastProcessTimeRef.current = currentTime;
        }
      }
      
      animationFrameRef.current = requestAnimationFrame(processFrames);
    };

    processFrames();
  };

  // Stop processing and cleanup
  const cleanup = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    setCameraReady(false);
    setError('');
  };

  // Cleanup when detection stops
  useEffect(() => {
    if (!isDetecting) {
      cleanup();
    }
  }, [isDetecting]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  const handleToggleDetection = () => {
    if (isDetecting) {
      stopDetection();
    } else {
      startDetection();
    }
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'bg-green-500';
    if (conf >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="glass-effect rounded-2xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Camera Feed</h3>
        <div className="flex items-center space-x-3">
          <span className={`status-indicator ${isDetecting ? 'status-active' : 'status-inactive'}`}>
            <div className={`w-2 h-2 rounded-full mr-2 ${isDetecting ? 'bg-accent-500 animate-pulse' : 'bg-gray-400'}`}></div>
            {isDetecting ? 'Active' : 'Inactive'}
          </span>
          <button
            onClick={handleToggleDetection}
            className={`p-3 rounded-xl transition-all duration-200 ${
              isDetecting 
                ? 'bg-red-100 text-red-600 hover:bg-red-200' 
                : 'bg-primary-100 text-primary-600 hover:bg-primary-200'
            }`}
            disabled={isLoading}
          >
            {isDetecting ? <CameraOff className="w-5 h-5" /> : <Camera className="w-5 h-5" />}
          </button>
        </div>
      </div>
      
      <div className="relative bg-gray-900 rounded-xl overflow-hidden aspect-video">
        {error ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-white">
              <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-400" />
              <p className="text-lg font-medium mb-2">Camera Error</p>
              <p className="text-sm text-gray-300">{error}</p>
            </div>
          </div>
        ) : isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-white">
              <div className="w-12 h-12 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-lg font-medium">Initializing Camera...</p>
            </div>
          </div>
        ) : !isDetecting ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-white">
              <Camera className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <p className="text-lg font-medium mb-2">Camera Ready</p>
              <p className="text-sm text-gray-300">Click the camera button to start detection</p>
            </div>
          </div>
        ) : null}
        
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />
        
        {/* Prediction Overlay */}
        {isDetecting && cameraReady && (
          <div className="absolute top-4 right-4 z-10">
            <div className="bg-black/70 backdrop-blur-sm rounded-lg p-4 text-white min-w-[200px]">
              <div className="text-center">
                {currentPrediction ? (
                  <>
                    <div className="text-2xl font-bold mb-2 capitalize text-white">
                      {currentPrediction}
                    </div>
                    <div className="flex items-center justify-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${getConfidenceColor(confidence)}`}></div>
                      <span className="text-sm">
                        {(confidence * 100).toFixed(1)}% confidence
                      </span>
                    </div>
                    <div className="w-full bg-gray-600 rounded-full h-1 mt-2">
                      <div 
                        className={`h-1 rounded-full transition-all duration-300 ${getConfidenceColor(confidence)}`}
                        style={{ width: `${confidence * 100}%` }}
                      ></div>
                    </div>
                  </>
                ) : (
                  <div className="text-gray-300">
                    <div className="text-lg mb-1">Detecting...</div>
                    <div className="text-xs">Show your hand</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ display: 'none' }}
        />
      </div>
      
      <div className="mt-4 text-center">
        <p className="text-sm text-gray-600">
          Supported gestures: <span className="font-medium text-primary-600">Hello</span>, <span className="font-medium text-accent-600">Thank You</span>
        </p>
      </div>
    </div>
  );
};

export default WebcamFeed;