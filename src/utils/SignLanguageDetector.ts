import * as tf from '@tensorflow/tfjs';
import { Hands, Results } from '@mediapipe/hands';

export interface DetectionResult {
  gesture: string;
  confidence: number;
}

export class SignLanguageDetector {
  private model: tf.LayersModel | null = null;
  private hands: Hands | null = null;
  private isProcessing: boolean = false;
  private labelMap: { [key: number]: string } = {
    0: 'hello',
    1: 'thankyou'
  };
  private landmarkSequence: number[][] = [];
  private readonly maxSequenceLength = 30;
  private readonly landmarkCount = 42; // 21 landmarks * 2 (x, y)
  private lastDetectionTime = 0;
  private isMediaPipeReady = false;

  async initialize(): Promise<void> {
    try {
      console.log('Initializing TensorFlow.js...');
      await tf.ready();
      
      console.log('Loading trained model...');
      this.model = await tf.loadLayersModel('/sign_lang_tfjs/model.json');
      console.log('Model loaded successfully:', this.model);
      
      console.log('Initializing MediaPipe Hands...');
      
      // Create MediaPipe Hands instance
      this.hands = new Hands({
        locateFile: (file) => {
          console.log('Loading MediaPipe file:', file);
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
        }
      });
      
      // Configure MediaPipe options
      this.hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0, // Use lighter model for better performance
        minDetectionConfidence: 0.3, // Lower threshold
        minTrackingConfidence: 0.3   // Lower threshold
      });
      
      // Set up results callback
      this.hands.onResults((results: Results) => {
        console.log('MediaPipe results received');
        this.onHandsResults(results);
      });
      
      // Test MediaPipe initialization
      console.log('MediaPipe Hands initialized successfully');
      this.isMediaPipeReady = true;
      
    } catch (error) {
      console.error('Failed to initialize sign language detector:', error);
      throw error;
    }
  }

  private onHandsResults(results: Results): void {
    console.log('Processing MediaPipe results...');
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      console.log('✅ Hand landmarks detected:', landmarks.length, 'points');
      
      // Extract x, y coordinates from landmarks
      const landmarkArray: number[] = [];
      for (const landmark of landmarks) {
        landmarkArray.push(landmark.x, landmark.y);
      }
      
      console.log('Landmark array length:', landmarkArray.length);
      
      // Add to sequence
      this.landmarkSequence.push(landmarkArray);
      
      // Keep only the last maxSequenceLength frames
      if (this.landmarkSequence.length > this.maxSequenceLength) {
        this.landmarkSequence.shift();
      }
      
      console.log('Current sequence length:', this.landmarkSequence.length);
    } else {
      console.log('❌ No hand landmarks detected in frame');
    }
  }

  async detectGesture(imageData: ImageData): Promise<DetectionResult | null> {
    if (this.isProcessing || !this.model || !this.hands || !this.isMediaPipeReady) {
      console.log('Detection skipped - not ready:', {
        isProcessing: this.isProcessing,
        hasModel: !!this.model,
        hasHands: !!this.hands,
        isMediaPipeReady: this.isMediaPipeReady
      });
      return null;
    }

    // Throttle processing
    const now = Date.now();
    if (now - this.lastDetectionTime < 200) {
      return null;
    }
    this.lastDetectionTime = now;

    this.isProcessing = true;

    try {
      console.log('Processing frame for gesture detection...');
      
      // Convert ImageData to canvas for MediaPipe processing
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      
      console.log('Sending frame to MediaPipe...', {
        width: canvas.width,
        height: canvas.height
      });
      
      // Process with MediaPipe
      await this.hands.send({ image: canvas });
      
      // Check if we have enough frames for prediction
      if (this.landmarkSequence.length < 5) { // Reduced requirement
        console.log('Not enough landmark frames for prediction:', this.landmarkSequence.length);
        this.isProcessing = false;
        return null;
      }
      
      console.log('Making prediction with', this.landmarkSequence.length, 'frames');
      
      // Prepare input tensor
      const inputSequence = this.prepareInputSequence();
      const inputTensor = tf.tensor3d([inputSequence], [1, this.maxSequenceLength, this.landmarkCount]);
      
      console.log('Input tensor shape:', inputTensor.shape);
      
      // Make prediction
      const prediction = this.model.predict(inputTensor) as tf.Tensor;
      const predictionData = await prediction.data();
      
      console.log('Raw prediction:', Array.from(predictionData));
      
      // Get the predicted class and confidence
      const maxIndex = predictionData.indexOf(Math.max(...Array.from(predictionData)));
      const confidence = predictionData[maxIndex];
      
      console.log('Predicted gesture:', this.labelMap[maxIndex], 'with confidence:', confidence);
      
      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();
      
      this.isProcessing = false;
      
      // Return result only if confidence is above threshold
      if (confidence > 0.5) {
        return {
          gesture: this.labelMap[maxIndex],
          confidence: confidence
        };
      }
      
      return null;
    } catch (error) {
      console.error('Error detecting gesture:', error);
      this.isProcessing = false;
      return null;
    }
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
    
    console.log('Prepared sequence shape:', sequence.length, 'x', sequence[0]?.length || 0);
    
    return sequence;
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    if (this.hands) {
      this.hands.close();
      this.hands = null;
    }
    this.landmarkSequence = [];
    this.isMediaPipeReady = false;
  }
}