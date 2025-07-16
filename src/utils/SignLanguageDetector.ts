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
      console.log('🚀 Starting TensorFlow.js initialization...');
      await tf.ready();
      console.log('✅ TensorFlow.js ready');
      
      console.log('📦 Loading trained LSTM model from /sign_lang_tfjs/model.json...');
      this.model = await tf.loadLayersModel('/sign_lang_tfjs/model.json');
      console.log('✅ TensorFlow.js model loaded successfully:', this.model.summary());
      
      console.log('🤖 Initializing MediaPipe Hands...');
      
      // Create MediaPipe Hands instance
      this.hands = new Hands({
        locateFile: (file) => {
          const url = `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
          console.log('📥 Loading MediaPipe file:', file, 'from:', url);
          return url;
        }
      });
      
      // Configure MediaPipe options for better detection
      this.hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      
      console.log('⚙️ MediaPipe options configured');
      
      // Set up results callback
      this.hands.onResults((results: Results) => {
        this.onHandsResults(results);
      });
      
      this.isMediaPipeReady = true;
      console.log('✅ MediaPipe Hands initialized and ready');
      
    } catch (error) {
      console.error('❌ Failed to initialize sign language detector:', error);
      throw error;
    }
  }

  private onHandsResults(results: Results): void {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      console.log('👋 Hand detected! Landmarks:', landmarks.length);
      
      // Extract x, y coordinates from landmarks
      const landmarkArray: number[] = [];
      for (const landmark of landmarks) {
        landmarkArray.push(landmark.x, landmark.y);
      }
      
      // Add to sequence
      this.landmarkSequence.push(landmarkArray);
      
      // Keep only the last maxSequenceLength frames
      if (this.landmarkSequence.length > this.maxSequenceLength) {
        this.landmarkSequence.shift();
      }
      
      console.log('📊 Landmark sequence length:', this.landmarkSequence.length, '/', this.maxSequenceLength);
    } else {
      console.log('🔍 No hand landmarks detected in current frame');
    }
  }

  async detectGesture(imageData: ImageData): Promise<DetectionResult | null> {
    if (this.isProcessing || !this.model || !this.hands || !this.isMediaPipeReady) {
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
      
      // Convert ImageData to canvas for MediaPipe
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      
      // Send to MediaPipe for hand detection
      await this.hands.send({ image: canvas });
      
      // Check if we have enough frames for prediction
      if (this.landmarkSequence.length < 15) {
        console.log('⏳ Need more frames for prediction. Current:', this.landmarkSequence.length, 'Required: 15');
        this.isProcessing = false;
        return null;
      }
      
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
      
    } catch (error) {
      console.error('❌ Error during gesture detection:', error);
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
    
    console.log('📋 Prepared input sequence shape:', sequence.length, 'x', sequence[0]?.length || 0);
    
    return sequence;
  }

  dispose(): void {
    console.log('🧹 Disposing detector resources...');
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