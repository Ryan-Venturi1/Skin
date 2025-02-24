// Global variable for the TensorFlow.js model
let model;

// Load the TensorFlow.js model from the "model" folder
async function loadModel() {
  try {
    model = await tf.loadLayersModel('model/model.json');
    logEvent("Model loaded successfully.");
  } catch (error) {
    logEvent("Error loading model: " + error.message);
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
  const tensor = preprocessImage(imageElement);
  const predictions = model.predict(tensor);
  const data = await predictions.data();

  // Assume model outputs probabilities in the following order: [Cancer, Infected, Healthy]
  const labels = ["Cancer", "Infected", "Healthy"];
  let results = [];
  for (let i = 0; i < data.length; i++) {
    results.push({ label: labels[i], confidence: data[i] });
  }
  // Sort results so the highest confidence is first
  results.sort((a, b) => b.confidence - a.confidence);
  logEvent(`Prediction: ${results[0].label} with confidence ${(results[0].confidence * 100).toFixed(2)}%`);
  return results;
}

// Log events to the #logging panel
function logEvent(message) {
  const logDiv = document.getElementById('logging');
  const timestamp = new Date().toLocaleTimeString();
  const logMessage = document.createElement('p');
  logMessage.textContent = `[${timestamp}] ${message}`;
  logDiv.appendChild(logMessage);
}

// Start the device camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }
    });
    const video = document.getElementById('video');
    video.srcObject = stream;
    logEvent("Camera started.");
  } catch (err) {
    alert('Camera error: ' + err.message);
    logEvent("Camera error: " + err.message);
  }
}

// Capture an image, run analysis, and display results
async function captureImage() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const resultsDiv = document.getElementById('results');
  const angleSuggestionSpan = document.getElementById('angle-suggestion');

  // Draw current video frame on canvas
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

  // Create an image element from the captured canvas data
  const img = new Image();
  img.src = canvas.toDataURL('image/jpeg');
  img.onload = async () => {
    const timestamp = new Date().toLocaleString();
    logEvent("Image captured at " + timestamp);

    // Use angle tracking module for a suggestion
    const suggestion = window.getAngleSuggestion(img);
    angleSuggestionSpan.textContent = suggestion;
    logEvent("Angle suggestion: " + suggestion);

    // Run inference on the captured image
    const analysisResults = await analyzeImage(img);

    // Retrieve confidence scores
    const cancerScore = analysisResults.find(r => r.label === "Cancer").confidence;
    const infectedScore = analysisResults.find(r => r.label === "Infected").confidence;
    const healthyScore = analysisResults.find(r => r.label === "Healthy").confidence;

    // Add a warning if the healthy confidence is low (threshold example: 80%)
    let warningMessage = "";
    if (healthyScore < 0.8) {
      warningMessage = `<p class="warning">Warning: Low healthy confidence. Please consult a medical professional.</p>`;
    }

    // Create a result display item
    const historyItem = document.createElement('div');
    historyItem.className = "history-item";
    historyItem.innerHTML = `
      <img src="${img.src}" class="history-img">
      <p>Captured: ${timestamp}</p>
      <p>Primary Prediction: ${analysisResults[0].label} (${(analysisResults[0].confidence * 100).toFixed(2)}%)</p>
      <p>Scores - Cancer: ${(cancerScore * 100).toFixed(2)}%, Infected: ${(infectedScore * 100).toFixed(2)}%, Healthy: ${(healthyScore * 100).toFixed(2)}%</p>
      ${warningMessage}
    `;
    resultsDiv.insertBefore(historyItem, resultsDiv.firstChild);
  };
}

// Set up event listeners
document.getElementById('capture').addEventListener('click', captureImage);

// Initialize camera and load the model on page load
window.addEventListener('load', () => {
  startCamera();
  loadModel();
});