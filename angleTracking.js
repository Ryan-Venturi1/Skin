/**
 * Enhanced Angle Tracking Module
 * Provides dynamic guidance similar to Apple's FaceID training
 * for better skin lesion image capture.
 */

// State tracking for the current guidance instruction
let currentGuidanceState = 'center';
let stateStartTime = Date.now();
let lastBrightnessValue = null;
let lastDistanceEstimate = null;
let focusScore = 0;

// Constants for timing
const STATE_DURATION = 3000; // How long to stay in one state (ms)
const STATES = [
  'center', 'closer', 'further', 'left_side', 'right_side', 
  'more_light', 'less_reflection', 'hold_steady'
];

/**
 * Analyze image quality metrics
 * @param {HTMLImageElement|HTMLVideoElement} imageElement - Image or video element to analyze
 * @returns {Object} - Quality metrics
 */
function analyzeImageQuality(imageElement) {
  // Create a canvas to analyze the image
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = imageElement.videoWidth || imageElement.width || 300;
  canvas.height = imageElement.videoHeight || imageElement.height || 300;
  
  // Draw the image onto the canvas
  ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
  
  // Get image data
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  // Calculate brightness
  let brightness = 0;
  let blurriness = 0;
  
  // Sample pixels for brightness (every 10th pixel to save computation)
  for (let i = 0; i < data.length; i += 40) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    // Weighted RGB for perceived brightness
    brightness += (r * 0.299 + g * 0.587 + b * 0.114);
  }
  
  brightness = brightness / (data.length / 40) / 255; // Normalized to 0-1
  
  // Estimate blurriness using a simple edge detection method
  // (In a real application, you would use more sophisticated methods)
  let edgeCount = 0;
  for (let y = 1; y < canvas.height - 1; y += 2) {
    for (let x = 1; x < canvas.width - 1; x += 2) {
      const idx = (y * canvas.width + x) * 4;
      const c1 = data[idx];
      const c2 = data[idx + 4]; // pixel to the right
      const c3 = data[idx + canvas.width * 4]; // pixel below
      
      // If there's a significant difference, it's an edge
      if (Math.abs(c1 - c2) > 20 || Math.abs(c1 - c3) > 20) {
        edgeCount++;
      }
    }
  }
  
  // Normalize edge count to 0-1 (higher is sharper)
  const sharpness = Math.min(1, edgeCount / ((canvas.width * canvas.height) / 10));
  
  // Detect if there's a subject in focus (simulated)
  // In a real app, you'd use object detection to find the skin lesion
  const hasFocusedSubject = sharpness > 0.4;
  
  // Estimate distance based on edge frequency (simplified approach)
  // More edges might mean we're closer to the subject
  const distanceEstimate = 1 - (edgeCount / ((canvas.width * canvas.height) / 5));
  
  // Update focus score based on sharpness and if we have a subject
  focusScore = sharpness * (hasFocusedSubject ? 1.5 : 0.5);
  
  return {
    brightness,
    sharpness,
    hasFocusedSubject,
    distanceEstimate,
    focusScore
  };
}

/**
 * Get the next guidance state based on image analysis
 * @param {Object} metrics - Image quality metrics
 * @returns {string} - Next state
 */
function getNextState(metrics) {
  // If we've been in current state long enough, transition to a new one
  const timeInState = Date.now() - stateStartTime;
  
  if (timeInState > STATE_DURATION && focusScore > 0.7) {
    // If image quality is good, suggest holding steady
    return 'hold_steady';
  }
  
  // Determine next state based on metrics
  if (metrics.brightness < 0.4) {
    return 'more_light';
  }
  if (metrics.brightness > 0.85) {
    return 'less_reflection';
  }
  if (metrics.distanceEstimate < 0.3) {
    return 'further';
  }
  if (metrics.distanceEstimate > 0.7) {
    return 'closer';
  }
  if (!metrics.hasFocusedSubject) {
    // Alternate between suggesting moving left and right
    return Math.random() > 0.5 ? 'left_side' : 'right_side';
  }
  
  return 'center';
}

/**
 * Get angle and position guidance for better image capture
 * @param {HTMLImageElement|HTMLVideoElement} imageElement - Image or video feed to analyze
 * @returns {Object} - Guidance information
 */
function getGuidance(imageElement) {
  // Analyze image quality
  const metrics = analyzeImageQuality(imageElement);
  
  // Check if we should transition to a new state
  const timeInState = Date.now() - stateStartTime;
  if (timeInState > STATE_DURATION) {
    currentGuidanceState = getNextState(metrics);
    stateStartTime = Date.now();
  }
  
  // Update tracking values
  lastBrightnessValue = metrics.brightness;
  lastDistanceEstimate = metrics.distanceEstimate;
  
  // Create guidance message based on current state
  let message = '';
  let progressPercent = Math.min(100, (timeInState / STATE_DURATION) * 100);
  
  switch(currentGuidanceState) {
    case 'center':
      message = 'Center the skin area in the frame';
      break;
    case 'closer':
      message = 'Move camera closer to the skin';
      break;
    case 'further':
      message = 'Move camera farther from the skin';
      break;
    case 'left_side':
      message = 'Move slightly to the left';
      break;
    case 'right_side':
      message = 'Move slightly to the right';
      break;
    case 'more_light':
      message = 'Find better lighting';
      break;
    case 'less_reflection':
      message = 'Reduce glare or reflection';
      break;
    case 'hold_steady':
      message = 'Hold steady for capture';
      progressPercent = Math.min(100, focusScore * 100);
      break;
  }
  
  return {
    message,
    state: currentGuidanceState,
    focusScore: focusScore.toFixed(2),
    progressPercent: Math.round(progressPercent),
    readyForCapture: currentGuidanceState === 'hold_steady' && focusScore > 0.8
  };
}

// Expose the function globally
window.getAngleSuggestion = function(imageElement) {
  const guidance = getGuidance(imageElement);
  return guidance.message;
};

// Expose the full guidance object for more detailed UI
window.getDetailedGuidance = getGuidance;