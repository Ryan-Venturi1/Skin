// Global variable for the TensorFlow.js model
let model;
let guidanceInterval;
let captureReady = false;


// Load the TensorFlow.js model from the "model" folder
async function loadModel() {
  try {
    model = await tf.loadLayersModel('model/model.json');
    logEvent("Model loaded successfully.");
    
    // Update UI to show model is ready
    document.getElementById('model-status').textContent = 'Ready';
    document.getElementById('model-status').classList.add('status-ready');
  } catch (error) {
    logEvent("Error loading model: " + error.message);
    
    // Update UI to show model failed to load
    document.getElementById('model-status').textContent = 'Error';
    document.getElementById('model-status').classList.add('status-error');
  }
}
// Preprocess the image: resize to 224x224, normalize pixel values, and add a batch dimension
function preprocessImage(imageElement) {
  return tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(tf.scalar(255))
    .expandDims();
}

// Analyze the image using the loaded model
async function analyzeImage(imageElement) {
  // Show loading state
  document.getElementById('capture').textContent = 'Analyzing...';
  document.getElementById('capture').disabled = true;

  try {
    const tensor = preprocessImage(imageElement);
    const predictions = model.predict(tensor);
    const data = await predictions.data();
    
    // Clean up tensor to prevent memory leaks
    tensor.dispose();
    predictions.dispose();

    // Assume model outputs probabilities in the following order: [Cancer, Infected, Healthy]
    // Update based on your actual ISIC model classes
    const labels = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", 
                   "Actinic Keratosis", "Benign Keratosis", "Dermatofibroma", 
                   "Vascular Lesion", "Healthy"];
    
    let results = [];
    for (let i = 0; i < data.length; i++) {
      // Only include classes that exist in the model output
      if (i < labels.length) {
        results.push({ label: labels[i], confidence: data[i] });
      }
    }
    
    // Sort results so the highest confidence is first
    results.sort((a, b) => b.confidence - a.confidence);
    logEvent(`Prediction: ${results[0].label} with confidence ${(results[0].confidence * 100).toFixed(2)}%`);
    
    // Reset button state
    document.getElementById('capture').textContent = 'Capture Image';
    document.getElementById('capture').disabled = false;
    
    return results;
  } catch (error) {
    logEvent("Error during analysis: " + error.message);
    
    // Reset button state in case of error
    document.getElementById('capture').textContent = 'Capture Image';
    document.getElementById('capture').disabled = false;
    
    return [{ label: "Error", confidence: 0 }];
  }
}

// Log events to the #logging panel
function logEvent(message) {
  const logDiv = document.getElementById('logging');
  const timestamp = new Date().toLocaleTimeString();
  const logMessage = document.createElement('p');
  logMessage.textContent = `[${timestamp}] ${message}`;
  logDiv.appendChild(logMessage);
  
  // Limit the number of log messages to prevent overwhelming the UI
  if (logDiv.children.length > 20) {
    logDiv.removeChild(logDiv.firstChild);
  }
}

// Start the device camera
async function startCamera() {
  try {
    // Try to get the back camera if available (better for skin images)
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { 
        facingMode: 'environment',
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });
    
    const video = document.getElementById('video');
    video.srcObject = stream;
    
    // Wait for video to be ready
    video.onloadedmetadata = () => {
      logEvent("Camera started. Resolution: " + video.videoWidth + "x" + video.videoHeight);
      
      // Start the guidance system once camera is ready
      startGuidanceSystem();
    };
  } catch (err) {
    alert('Camera error: ' + err.message);
    logEvent("Camera error: " + err.message);
    
    // Update UI to show camera failed
    document.getElementById('camera-status').textContent = 'Error';
    document.getElementById('camera-status').classList.add('status-error');
  }
}

// Update the UI with camera guidance
function updateGuidanceUI() {
  const video = document.getElementById('video');
  const guidanceContainer = document.getElementById('guidance-container');
  const guidanceMessage = document.getElementById('guidance-message');
  const guidanceProgressBar = document.getElementById('guidance-progress-bar');
  const focusIndicator = document.getElementById('focus-indicator');
  
  // Get detailed guidance from our enhanced module
  if (video.readyState === 4) { // HAVE_ENOUGH_DATA
    const guidance = window.getDetailedGuidance(video);
    
    // Update the guidance message
    guidanceMessage.textContent = guidance.message;
    
    // Update the progress bar
    guidanceProgressBar.style.width = `${guidance.progressPercent}%`;
    
    // Update the focus indicator
    if (guidance.focusScore > 0.7) {
      focusIndicator.className = 'focus-indicator good';
      focusIndicator.style.borderWidth = '3px';
    } else if (guidance.focusScore > 0.4) {
      focusIndicator.className = 'focus-indicator';
      focusIndicator.style.borderWidth = '2px';
    } else {
      focusIndicator.className = 'focus-indicator bad';
      focusIndicator.style.borderWidth = '1px';
    }
    
    // Update capture readiness
    captureReady = guidance.readyForCapture;
    
    // Make the capture button pulse when ready
    const captureButton = document.getElementById('capture');
    if (captureReady) {
      captureButton.classList.add('pulse');
      captureButton.textContent = 'Capture Now';
    } else {
      captureButton.classList.remove('pulse');
      captureButton.textContent = 'Capture Image';
    }
  }
}

// Start the guidance system
function startGuidanceSystem() {
  // Set up the guidance update interval (update every 200ms)
  if (guidanceInterval) {
    clearInterval(guidanceInterval);
  }
  
  guidanceInterval = setInterval(updateGuidanceUI, 200);
  logEvent("Guidance system started.");
}

// Capture an image, run analysis, and display results
async function captureImage() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const resultsDiv = document.getElementById('results');
  
  // Change button state
  const captureButton = document.getElementById('capture');
  captureButton.textContent = 'Processing...';
  captureButton.disabled = true;
  captureButton.classList.remove('pulse');
  
  // Draw current video frame on canvas
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  
  // Create an image element from the captured canvas data
  const img = new Image();
  img.src = canvas.toDataURL('image/jpeg');
  
  // Wait for image to load
  img.onload = async () => {
    const timestamp = new Date().toLocaleString();
    logEvent("Image captured at " + timestamp);
    
    // Get final guidance reading for this capture
    const finalGuidance = window.getDetailedGuidance(img);
    logEvent(`Capture quality - Focus score: ${finalGuidance.focusScore}, State: ${finalGuidance.state}`);
    
    // Run inference on the captured image
    const analysisResults = await analyzeImage(img);
    
    // Retrieve confidence scores for summary
    // Make this robust in case some labels aren't found
    const getMalignantScore = () => {
      const melanoma = analysisResults.find(r => r.label === "Melanoma");
      const bcc = analysisResults.find(r => r.label === "Basal Cell Carcinoma");
      const scc = analysisResults.find(r => r.label === "Squamous Cell Carcinoma");
      
      let totalMalignantScore = 0;
      if (melanoma) totalMalignantScore += melanoma.confidence;
      if (bcc) totalMalignantScore += bcc.confidence;
      if (scc) totalMalignantScore += scc.confidence;
      
      return totalMalignantScore;
    };
    
    const getBenignScore = () => {
      const healthy = analysisResults.find(r => r.label === "Healthy");
      const benignKeratosis = analysisResults.find(r => r.label === "Benign Keratosis");
      const dermatofibroma = analysisResults.find(r => r.label === "Dermatofibroma");
      
      let totalBenignScore = 0;
      if (healthy) totalBenignScore += healthy.confidence;
      if (benignKeratosis) totalBenignScore += benignKeratosis.confidence;
      if (dermatofibroma) totalBenignScore += dermatofibroma.confidence;
      
      return totalBenignScore;
    };
    
    const malignantScore = getMalignantScore();
    const benignScore = getBenignScore();
    
    // Add a warning if the analysis suggests concern
    let warningMessage = "";
    if (malignantScore > 0.3) {
      warningMessage = `<p class="warning">Warning: Elevated risk score detected. Please consult a dermatologist promptly.</p>`;
    }
    
    // Create colored risk indicator
    let riskColor = "#4CAF50"; // Green for low risk
    if (malignantScore > 0.7) {
      riskColor = "#F44336"; // Red for high risk
    } else if (malignantScore > 0.3) {
      riskColor = "#FF9800"; // Orange for medium risk
    }
    
    // Create detail rows for all predictions
    const detailRows = analysisResults.slice(0, 5).map(result => {
      return `<div class="result-detail-row">
        <span class="result-label">${result.label}</span>
        <div class="result-bar-container">
          <div class="result-bar" style="width: ${(result.confidence * 100).toFixed(0)}%; 
               background-color: ${result.label === "Healthy" ? "#4CAF50" : 
               (result.label === "Melanoma" ? "#F44336" : "#FF9800")}">
          </div>
        </div>
        <span class="result-percent">${(result.confidence * 100).toFixed(1)}%</span>
      </div>`;
    }).join('');
    
    // Create a result display item
    const historyItem = document.createElement('div');
    historyItem.className = "history-item";
    historyItem.innerHTML = `
      <div class="history-header">
        <img src="${img.src}" class="history-img">
        <div class="history-meta">
          <div class="capture-time">${timestamp}</div>
          <div class="primary-prediction">${analysisResults[0].label}</div>
          <div class="confidence-badge" style="background-color: ${riskColor}">
            ${(analysisResults[0].confidence * 100).toFixed(0)}% confidence
          </div>
        </div>
      </div>
      <div class="history-details">
        <div class="result-details">
          ${detailRows}
        </div>
        <div class="summary-scores">
          <div class="summary-score">
            <span>Benign likelihood:</span>
            <span class="score-value">${(benignScore * 100).toFixed(1)}%</span>
          </div>
          <div class="summary-score">
            <span>Malignant likelihood:</span>
            <span class="score-value">${(malignantScore * 100).toFixed(1)}%</span>
          </div>
        </div>
        ${warningMessage}
      </div>
    `;
    
    // Add the new result at the top
    resultsDiv.insertBefore(historyItem, resultsDiv.firstChild);
    
    // Reset capture button
    captureButton.disabled = false;
    captureButton.textContent = 'Capture Image';
  };
}

// Add camera switching functionality
function switchCamera() {
  const video = document.getElementById('video');
  
  // Stop current stream
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
  }
  
  // Get current facing mode
  const currentFacingMode = video.srcObject?.getVideoTracks()[0]?.getSettings().facingMode;
  const newFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
  
  // Start camera with new facing mode
  navigator.mediaDevices.getUserMedia({
    video: { facingMode: newFacingMode }
  }).then(stream => {
    video.srcObject = stream;
    logEvent(`Switched to ${newFacingMode === 'environment' ? 'back' : 'front'} camera.`);
  }).catch(err => {
    logEvent("Error switching camera: " + err.message);
  });
}

// Set up event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Add event listeners once DOM is fully loaded
  document.getElementById('capture').addEventListener('click', captureImage);
  
  if (document.getElementById('switch-camera')) {
    document.getElementById('switch-camera').addEventListener('click', switchCamera);
  }
  
  // Initialize camera and load the model
  startCamera();
  loadModel();
});

// Check browser compatibility
function checkCompatibility() {
  const issues = [];
  
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    issues.push("Camera access is not supported in this browser.");
  }
  
  if (!window.indexedDB) {
    issues.push("IndexedDB is not supported (needed for model storage).");
  }
  
  if (!window.WebGLRenderingContext) {
    issues.push("WebGL is not supported (needed for TensorFlow.js).");
  }
  
  if (issues.length > 0) {
    const warningDiv = document.createElement('div');
    warningDiv.className = "compatibility-warning";
    warningDiv.innerHTML = `
      <h3>Compatibility Issues</h3>
      <ul>${issues.map(issue => `<li>${issue}</li>`).join('')}</ul>
      <p>The app may not function correctly. Please try a different browser.</p>
    `;
    document.body.insertBefore(warningDiv, document.body.firstChild);
  }
  
  return issues.length === 0;
}

// Check compatibility on load
window.addEventListener('load', checkCompatibility);