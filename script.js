
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture');
const results = document.getElementById('results');

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        video.srcObject = stream;
    } catch (err) {
        alert('Camera error: ' + err.message);
    }
}

function captureImage() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    const img = document.createElement('img');
    img.src = canvas.toDataURL('image/jpeg');
    
    const timestamp = new Date().toLocaleString();
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.innerHTML = `
        <img src="${img.src}" class="history-img">
        <p>Captured: ${timestamp}</p>
        <p>Status: Analysis complete</p>
    `;
    
    results.insertBefore(historyItem, results.firstChild);
}

captureBtn.addEventListener('click', captureImage);
startCamera();
