import * as tf from '@tensorflow/tfjs';

export interface DetectionResult {
  gesture: string;
  confidence: number;
}

export class SignLanguageDetector {
  private model: tf.LayersModel | null = null;
  private isProcessing: boolean = false;
  private labelMap: { [key: number]: string } = {
    0: 'hello',
    1: 'thankyou'
  };
  private landmarkSequence: number[][] = [];
  private readonly maxSequenceLength = 30;
  private readonly landmarkCount = 42; // 21 landmarks * 2 (x, y)
  private lastDetectionTime = 0;

  async initialize(): Promise<void> {
    try {
      console.log('🚀 Starting TensorFlow.js initialization...');
      await tf.ready();
      console.log('✅ TensorFlow.js ready');
      
      console.log('📦 Loading trained LSTM model from /sign_lang_tfjs/model.json...');
      this.model = await tf.loadLayersModel('/sign_lang_tfjs/model.json');
      console.log('✅ TensorFlow.js model loaded successfully');
      console.log('📊 Model input shape:', this.model.inputs[0].shape);
      console.log('📊 Model output shape:', this.model.outputs[0].shape);
      
    } catch (error) {
      console.error('❌ Failed to initialize sign language detector:', error);
      throw error;
    }
  }

  async detectGesture(imageData: ImageData): Promise<DetectionResult | null> {
    if (this.isProcessing || !this.model) {
      return null;
    }

    // Throttle processing to every 300ms
    const now = Date.now();
    if (now - this.lastDetectionTime < 300) {
      return null;
    }
    this.lastDetectionTime = now;

    this.isProcessing = true;

    try {
      console.log('🎥 Processing frame for gesture detection...');
      
      // Simulate hand landmark detection from image
      const landmarks = this.extractHandLandmarks(imageData);
      
      if (landmarks) {
        console.log('👋 Hand landmarks extracted:', landmarks.length, 'coordinates');
        
        // Add to sequence
        this.landmarkSequence.push(landmarks);
        
        // Keep only the last maxSequenceLength frames
        if (this.landmarkSequence.length > this.maxSequenceLength) {
          this.landmarkSequence.shift();
        }
        
        console.log('📊 Landmark sequence length:', this.landmarkSequence.length, '/', this.maxSequenceLength);
        
        // Check if we have enough frames for prediction
        if (this.landmarkSequence.length >= 10) {
          console.log('🧠 Making prediction with', this.landmarkSequence.length, 'landmark frames');
          
          // Prepare input tensor
          const inputSequence = this.prepareInputSequence();
          const inputTensor = tf.tensor3d([inputSequence], [1, this.maxSequenceLength, this.landmarkCount]);
          
          console.log('📐 Input tensor shape:', inputTensor.shape);
          
          // Make prediction using the trained model
          const prediction = this.model.predict(inputTensor) as tf.Tensor;
          const predictionData = await prediction.data();
          
          console.log('🎯 Raw prediction probabilities:', Array.from(predictionData));
          
          // Get the predicted class and confidence
          const maxIndex = predictionData.indexOf(Math.max(...Array.from(predictionData)));
          const confidence = predictionData[maxIndex];
          const gesture = this.labelMap[maxIndex];
          
          console.log('🏆 Predicted gesture:', gesture, 'with confidence:', (confidence * 100).toFixed(1) + '%');
          
          // Clean up tensors
          inputTensor.dispose();
          prediction.dispose();
          
          this.isProcessing = false;
          
          // Return result only if confidence is above threshold
          if (confidence > 0.6) {
            console.log('✅ High confidence prediction returned:', gesture);
            return {
              gesture: gesture,
              confidence: confidence
            };
          } else {
            console.log('⚠️ Low confidence, not returning prediction');
            return null;
          }
        } else {
          console.log('⏳ Need more frames for prediction. Current:', this.landmarkSequence.length, 'Required: 10');
        }
      } else {
        console.log('❌ No hand landmarks detected in frame');
      }
      
      this.isProcessing = false;
      return null;
      
    } catch (error) {
      console.error('❌ Error during gesture detection:', error);
      this.isProcessing = false;
      return null;
    }
  }

  private extractHandLandmarks(imageData: ImageData): number[] | null {
    // Simulate hand detection based on image properties
    const pixels = imageData.data;
    let skinColorPixels = 0;
    const sampleSize = Math.min(2000, pixels.length / 4);
    
    // Sample pixels to detect skin color (basic hand detection)
    for (let i = 0; i < sampleSize * 4; i += 16) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      
      // Simple skin color detection
      if (r > 95 && g > 40 && b > 20 && 
          Math.max(r, g, b) - Math.min(r, g, b) > 15 &&
          Math.abs(r - g) > 15 && r > g && r > b) {
        skinColorPixels++;
      }
    }
    
    const skinRatio = skinColorPixels / sampleSize;
    console.log('🔍 Skin detection ratio:', (skinRatio * 100).toFixed(2) + '%');
    
    // If we find enough skin-colored pixels, generate simulated landmarks
    if (skinRatio > 0.03) {
      console.log('✅ Hand detected, generating landmarks');
      
      // Generate 21 hand landmarks (x, y coordinates)
      const landmarks: number[] = [];
      const centerX = 0.5;
      const centerY = 0.5;
      
      // Create realistic hand landmark positions
      const handPoints = [
        // Wrist
        [0.0, 0.0],
        // Thumb
        [-0.1, -0.1], [-0.15, -0.2], [-0.18, -0.25], [-0.2, -0.3],
        // Index finger
        [0.05, -0.15], [0.08, -0.25], [0.1, -0.35], [0.12, -0.4],
        // Middle finger
        [0.15, -0.1], [0.18, -0.2], [0.2, -0.3], [0.22, -0.35],
        // Ring finger
        [0.25, -0.05], [0.28, -0.15], [0.3, -0.25], [0.32, -0.3],
        // Pinky
        [0.35, 0.0], [0.38, -0.1], [0.4, -0.2], [0.42, -0.25]
      ];
      
      // Add some variation based on time for realistic movement
      const timeVariation = Math.sin(Date.now() / 1000) * 0.02;
      
      for (const [dx, dy] of handPoints) {
        landmarks.push(
          Math.max(0, Math.min(1, centerX + dx + timeVariation)),
          Math.max(0, Math.min(1, centerY + dy + timeVariation * 0.5))
        );
      }
      
      return landmarks;
    }
    
    return null;
  }

  private prepareInputSequence(): number[][] {
    const sequence: number[][] = [];
    
    // Use the most recent frames, pad if necessary
    const recentFrames = this.landmarkSequence.slice(-this.maxSequenceLength);
    
    // If we have fewer frames than needed, pad with zeros at the beginning
    const paddingNeeded = this.maxSequenceLength - recentFrames.length;
    
    for (let i = 0; i < paddingNeeded; i++) {
      sequence.push(new Array(this.landmarkCount).fill(0));
    }
    
    // Add the actual landmark data
    sequence.push(...recentFrames);
    
    console.log('📋 Prepared input sequence shape:', sequence.length, 'x', sequence[0]?.length || 0);
    
    return sequence;
  }

  dispose(): void {
    console.log('🧹 Disposing detector resources...');
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.landmarkSequence = [];
  }
}