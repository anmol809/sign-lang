@@ .. @@
   useEffect(() => {
     const initializeDetector = async () => {
       try {
        console.log('🚀 Initializing real detector with your trained model...');
        setModelLoadingProgress(20);
-          setModelLoadingProgress(prev => {
-            if (prev >= 90) {
-              clearInterval(progressInterval);
-              return 90;
-            }
-            return prev + Math.random() * 15;
-          });
-        }, 150);
+        setModelLoadingProgress(10);
+        console.log('Starting detector initialization...');
 
         const newDetector = new SignLanguageDetector();
        setModelLoadingProgress(50);
        
         await newDetector.initialize();
        setModelLoadingProgress(90);
         
         detectorRef.current = newDetector;
+        setModelLoadingProgress(100);
         setModelLoaded(true);
-        setModelLoadingProgress(100);
-        clearInterval(progressInterval);
         
-        console.log('Detector initialized successfully');
        console.log('✅ Your trained model is ready for real-time detection!');
       } catch (error) {
        console.error('❌ Failed to initialize detector:', error);
         setModelLoadingProgress(0);
       }
     };

@@ .. @@
   const processFrame = useCallback(async (imageData: ImageData) => {
-    if (!detectorRef.current || !isDetecting || processingRef.current) {
+    if (!detectorRef.current || !isDetecting || processingRef.current || !modelLoaded) {
       return;
     }

@@ .. @@
     try {
       const result = await detectorRef.current.detectGesture(imageData);
       
-      if (result && result.confidence > 0.5) {
+      if (result && result.confidence > 0.6) {
         setCurrentPrediction(result.gesture);
         setConfidence(result.confidence);
         
-        // Add to history if confidence is high enough and not a duplicate
-        if (result.confidence > 0.7) {
+        // Add to history if confidence is high enough
+        if (result.confidence > 0.75) {
           const newPrediction: Prediction = {
             gesture: result.gesture,
             confidence: result.confidence,
@@ .. @@
           setPredictionHistory(prev => {
-            // Avoid duplicate consecutive predictions within 2 seconds
+            // Avoid duplicate consecutive predictions within 1.5 seconds
             const lastPrediction = prev[prev.length - 1];
             if (lastPrediction && 
                 lastPrediction.gesture === newPrediction.gesture && 
-                newPrediction.timestamp - lastPrediction.timestamp < 2000) {
+                newPrediction.timestamp - lastPrediction.timestamp < 1500) {
               return prev;
             }
             
-            // Keep only last 15 predictions to avoid memory issues
+            // Keep only last 20 predictions
             const updated = [...prev, newPrediction];
-            return updated.slice(-15);
+            return updated.slice(-20);
           });
         }
       } else {
-        // Gradually reduce confidence when no gesture is detected
+        // Gradually reduce confidence when no strong gesture is detected
         setConfidence(prev => {
-          const newConfidence = Math.max(0, prev - 0.05);
-          if (newConfidence < 0.3) {
+          const newConfidence = Math.max(0, prev - 0.08);
+          if (newConfidence < 0.4) {
             setCurrentPrediction(null);
           }
           return newConfidence;
@@ .. @@
     } finally {
       processingRef.current = false;
     }
-  }, [isDetecting]);
+  }, [isDetecting, modelLoaded]);