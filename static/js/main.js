async function analyzeSentiment() {
    const input = document.getElementById('sentiment-input').value;
    const resultArea = document.getElementById('sentiment-result');
    
    if (!input) return;

    resultArea.style.display = 'block';
    resultArea.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

    try {
        const response = await fetch('/analyze_sentiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input })
        });
        const data = await response.json();
        
        if (data.error) {
            resultArea.innerHTML = `<span style="color: #ff4b2b">Error: ${data.error}</span>`;
        } else {
            resultArea.innerHTML = `Sentiment: <strong>${data.label}</strong> (${data.score})`;
        }
    } catch (e) {
        resultArea.innerHTML = 'Error connecting to server.';
    }
}

async function askAgent(fixedPrompt = null) {
    const inputArea = document.getElementById('agent-input');
    const resultArea = document.getElementById('agent-result');
    const prompt = fixedPrompt || (inputArea ? inputArea.value : null);
    
    if (!prompt) return;

    resultArea.style.display = 'block';
    resultArea.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Agent is thinking...';

    try {
        const response = await fetch('/ask_agent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt })
        });
        const data = await response.json();
        
        if (data.error) {
            resultArea.innerHTML = `<span style="color: #ff4b2b">Error: ${data.error}</span>`;
        } else {
            resultArea.innerHTML = `<div style="text-align: left; font-size: 0.9rem; opacity: 0.9; white-space: pre-wrap;">${data.response}</div>`;
        }
    } catch (e) {
        resultArea.innerHTML = 'Error connecting to server.';
    }
}

document.getElementById('flower-input').onchange = async function(event) {
    const file = event.target.files[0];
    const resultArea = document.getElementById('flower-result');
    
    if (!file) return;

    resultArea.style.display = 'block';
    resultArea.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning Image...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict_flower', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            resultArea.innerHTML = `<span style="color: #ff4b2b">Error: ${data.error}</span>`;
        } else {
            resultArea.innerHTML = `Prediction: <strong>${data.class}</strong> (${data.confidence})`;
        }
    } catch (e) {
        resultArea.innerHTML = 'Error connecting to server.';
    }
};
