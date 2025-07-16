const animationFrameRef = useRef<number | null>(null);
  const lastProcessTimeRef = useRef<number>(0);
  
  const { isDetecting, startDetection, stopDetection, processFrame, currentPrediction, confidence } = useSignLanguage();
  const [error, setError] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  const PROCESS_INTERVAL = 300; // Process every 300ms

  // Initialize camera only once when detection starts
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          }
        });

  const handleToggleDetection = () => {
    if (isDetecting) {
      stopDetection();
    } else {
      startDetection();

        if (ctx && video.readyState === 4) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          processFrame(imageData);

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
        }

      <div className="mt-4 text-center">
        <p className="text-sm text-gray-600">
          Using your trained LSTM model: <span className="font-medium text-primary-600">Hello</span>, <span className="font-medium text-accent-600">Thank You</span>
          <span className="block text-xs text-gray-500 mt-1">Real MediaPipe + TensorFlow.js detection</span>
        </p>
      </div>
    }
    </div>
  }