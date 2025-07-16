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

  async initialize(): Promise<void> {
    try {
      console.log('🚀 Initializing TensorFlow.js...');
      await tf.ready();
      console.log('✅ TensorFlow.js ready');
      
      console.log('📦 Loading your trained LSTM model...');
      // Load your actual model files
      this.model = await tf.loadLayersModel('/sign_lang_tfjs/model.json');
      console.log('✅ Model loaded successfully!');
      console.log('📊 Model input shape:', this.model.inputs[0].shape);
      console.log('📊 Model output shape:', this.model.outputs[0].shape);
      
      console.log('🤲 Initializing MediaPipe Hands...');
      this.hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });
      
      this.hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      
      console.log('✅ MediaPipe Hands initialized');
      
    } catch (error) {
      console.error('❌ Failed to initialize detector:', error);
      throw error;
    }
  }

  async detectGesture(imageData: ImageData): Promise<DetectionResult | null> {
    if (this.isProcessing || !this.model || !this.hands) {
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
      console.log('🎥 Processing frame...');
      
      // Convert ImageData to canvas for MediaPipe
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(imageData, 0, 0);
      
      // Process with MediaPipe
      const results = await new Promise<Results>((resolve) => {
        this.hands!.onResults(resolve);
        this.hands!.send({ image: canvas });
      });
      
      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        console.log('👋 Hand detected! Landmarks:', landmarks.length);
        
        // Extract x, y coordinates
        const landmarkArray: number[] = [];
        for (const landmark of landmarks) {
          landmarkArray.push(landmark.x, landmark.y);
        }
        
        console.log('📊 Extracted', landmarkArray.length, 'coordinates');
        
        // Add to sequence
        this.landmarkSequence.push(landmarkArray);
        
        // Keep only recent frames
        if (this.landmarkSequence.length > this.maxSequenceLength) {
          this.landmarkSequence.shift();
        }
        
        console.log('📈 Sequence length:', this.landmarkSequence.length, '/', this.maxSequenceLength);
        
        // Make prediction if we have enough frames
        if (this.landmarkSequence.length >= 15) {
          console.log('🧠 Making prediction...');
          
          const inputSequence = this.prepareInputSequence();
          const inputTensor = tf.tensor3d([inputSequence], [1, this.maxSequenceLength, this.landmarkCount]);
          
          console.log('📐 Input tensor shape:', inputTensor.shape);
          
          // Predict using your trained model
          const prediction = this.model.predict(inputTensor) as tf.Tensor;
          const predictionData = await prediction.data();
          
          console.log('🎯 Raw predictions:', Array.from(predictionData));
          
          // Get best prediction
          const maxIndex = predictionData.indexOf(Math.max(...Array.from(predictionData)));
          const confidence = predictionData[maxIndex];
          const gesture = this.labelMap[maxIndex];
          
          console.log('🏆 Predicted:', gesture, 'confidence:', (confidence * 100).toFixed(1) + '%');
          
          // Cleanup
          inputTensor.dispose();
          prediction.dispose();
          
          this.isProcessing = false;
          
          if (confidence > 0.7) {
            console.log('✅ High confidence prediction returned');
            return { gesture, confidence };
          } else {
            console.log('⚠️ Low confidence, threshold not met');
          }
        } else {
          console.log('⏳ Need more frames. Current:', this.landmarkSequence.length, 'Required: 15');
        }
      } else {
        console.log('❌ No hand detected in frame');
        // Clear sequence if no hand detected for too long
        if (this.landmarkSequence.length > 0) {
          this.landmarkSequence = [];
          console.log('🧹 Cleared landmark sequence');
        }
      }
      
      this.isProcessing = false;
      return null;
      
    } catch (error) {
      console.error('❌ Error during detection:', error);
      this.isProcessing = false;
      return null;
    }
  }

  private prepareInputSequence(): number[][] {
    const sequence: number[][] = [];
    
    // Use recent frames, pad with zeros if needed
    const recentFrames = this.landmarkSequence.slice(-this.maxSequenceLength);
    const paddingNeeded = this.maxSequenceLength - recentFrames.length;
    
    // Pad with zeros at the beginning
    for (let i = 0; i < paddingNeeded; i++) {
      sequence.push(new Array(this.landmarkCount).fill(0));
    }
    
    // Add actual landmark data
    sequence.push(...recentFrames);
    
    console.log('📋 Prepared sequence shape:', sequence.length, 'x', sequence[0]?.length);
    
    return sequence;
  }

  dispose(): void {
    console.log('🧹 Disposing detector...');
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    if (this.hands) {
      this.hands.close();
      this.hands = null;
    }
    this.landmarkSequence = [];
  }
}