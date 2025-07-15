@@ .. @@
   const animationFrameRef = useRef<number | null>(null);
   const lastProcessTimeRef = useRef<number>(0);
   
-  const { isDetecting, startDetection, stopDetection, processFrame, currentPrediction, confidence } = useSignLanguage();
+  const { isDetecting, startDetection, stopDetection, processFrame, currentPrediction, confidence, modelLoaded } = useSignLanguage();
   const [error, setError] = useState<string>('');
   const [isLoading, setIsLoading] = useState(false);
   const [cameraReady, setCameraReady] = useState(false);

@@ .. @@
   const handleToggleDetection = () => {
-    if (isDetecting) {
+    if (!modelLoaded) {
+      setError('Model is still loading. Please wait...');
+      return;
+    }
+    
+    if (isDetecting) {
       stopDetection();
     } else {
       startDetection();
@@ .. @@
           <button
             onClick={handleToggleDetection}
-            className={`p-3 rounded-xl transition-all duration-200 ${
-              isDetecting 
+            className={`p-3 rounded-xl transition-all duration-200 ${
+              !modelLoaded
+                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
+                : isDetecting 
                 ? 'bg-red-100 text-red-600 hover:bg-red-200' 
                 : 'bg-primary-100 text-primary-600 hover:bg-primary-200'
             }`}
-            disabled={isLoading}
+            disabled={isLoading || !modelLoaded}
+            title={!modelLoaded ? 'Model is loading...' : ''}
           >
             {isDetecting ? <CameraOff className="w-5 h-5" /> : <Camera className="w-5 h-5" />}
           </button>
@@ .. @@
       <div className="mt-4 text-center">
         <p className="text-sm text-gray-600">
-          Supported gestures: <span className="font-medium text-primary-600">Hello</span>, <span className="font-medium text-accent-600">Thank You</span>
+          Real-time AI detection: <span className="font-medium text-primary-600">Hello</span>, <span className="font-medium text-accent-600">Thank You</span>
+          {!modelLoaded && <span className="block text-xs text-yellow-600 mt-1">Loading neural network model...</span>}
         </p>
       </div>
     </div>