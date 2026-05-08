// --- Chatbot Logic ---
let sessionId = 'session_' + Math.random().toString(36).substr(2, 9);

function toggleChat() {
    const widget = document.getElementById('chat-widget');
    widget.classList.toggle('active');
    if (widget.classList.contains('active')) {
        document.getElementById('chat-input').focus();
    }
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const history = document.getElementById('chat-history');
    const prompt = input.value.trim();

    if (!prompt) return;

    // Add user message to UI
    appendMessage('user', prompt);
    input.value = '';

    // Add thinking indicator
    const thinkingId = 'thinking_' + Date.now();
    const thinkingDiv = document.createElement('div');
    thinkingDiv.id = thinkingId;
    thinkingDiv.className = 'message bot-message';
    thinkingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Agent is thinking...';
    history.appendChild(thinkingDiv);
    history.scrollTop = history.scrollHeight;

    try {
        const response = await fetch('/ask_agent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, session_id: sessionId })
        });
        const data = await response.json();
        
        // Remove thinking indicator
        document.getElementById(thinkingId).remove();

        if (data.error) {
            appendMessage('bot', `Error: ${data.error}`, true);
        } else {
            appendMessage('bot', data.response);
        }
    } catch (e) {
        document.getElementById(thinkingId).remove();
        appendMessage('bot', 'Error connecting to server.', true);
    }
}

function appendMessage(sender, text, isError = false) {
    const history = document.getElementById('chat-history');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    if (isError) msgDiv.style.color = '#ff4b2b';
    
    // Use marked for bot messages to render bolding/headings
    if (sender === 'bot') {
        msgDiv.innerHTML = marked.parse(text);
        
        // Add Copy Button for Professional feel
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-msg-btn';
        copyBtn.innerHTML = '<i class="far fa-copy"></i>';
        copyBtn.title = 'Copy Response';
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(text);
            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
            setTimeout(() => copyBtn.innerHTML = '<i class="far fa-copy"></i>', 2000);
        };
        msgDiv.appendChild(copyBtn);
    } else {
        msgDiv.textContent = text;
    }
    
    history.appendChild(msgDiv);
    history.scrollTop = history.scrollHeight;
}

function clearChat() {
    const history = document.getElementById('chat-history');
    history.innerHTML = '<div class="message bot-message">Chat history cleared. How can I assist you professionally today?</div>';
    sessionId = 'session_' + Math.random().toString(36).substr(2, 9); // Reset session
}

// Update existing askAgent to open the chat instead
async function askAgent(fixedPrompt = null) {
    const widget = document.getElementById('chat-widget');
    if (!widget.classList.contains('active')) toggleChat();
    
    if (fixedPrompt) {
        document.getElementById('chat-input').value = fixedPrompt;
        sendMessage();
    }
}

// Re-implementing the original functions to ensure they still work with the UI
async function analyzeSentiment() {
    const inputField = document.getElementById('sentiment-input');
    const input = inputField.value.trim();
    const resultArea = document.getElementById('sentiment-result');
    
    if (!input) {
        inputField.style.borderColor = 'var(--error)';
        setTimeout(() => inputField.style.borderColor = 'var(--glass-border)', 2000);
        return;
    }

    resultArea.style.display = 'block';
    resultArea.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <i class="fas fa-circle-notch fa-spin"></i> Processing linguistics...
        </div>
    `;

    try {
        const response = await fetch('/analyze_sentiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input })
        });
        const data = await response.json();
        
        if (data.error) {
            resultArea.innerHTML = `<span style="color: var(--error)"><i class="fas fa-exclamation-triangle"></i> Error: ${data.error}</span>`;
        } else {
            const label = data.label.toLowerCase();
            let color = 'var(--primary)';
            let width = '50%';
            let icon = 'fa-meh';

            if (label.includes('pos')) { color = 'var(--success)'; width = '90%'; icon = 'fa-smile-beam'; }
            else if (label.includes('neg')) { color = 'var(--error)'; width = '10%'; icon = 'fa-frown-open'; }

            resultArea.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span>Analysis: <strong style="color: ${color}">${data.label}</strong></span>
                    <i class="fas ${icon}" style="color: ${color}"></i>
                </div>
                <div class="sentiment-meter">
                    <div class="sentiment-fill" style="width: ${width}; background: ${color}"></div>
                </div>
            `;
        }
    } catch (e) {
        resultArea.innerHTML = '<span style="color: var(--error)">Network disruption detected.</span>';
    }
}

document.getElementById('flower-input').onchange = async function(event) {
    const file = event.target.files[0];
    const resultArea = document.getElementById('flower-result');
    const previewContainer = document.getElementById('image-preview-container');
    
    if (!file) return;

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewContainer.innerHTML = `<img src="${e.target.result}" class="preview-img" alt="Preview">`;
    };
    reader.readAsDataURL(file);

    resultArea.style.display = 'block';
    resultArea.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas fa-microchip fa-spin"></i> Running inference...
        </div>
    `;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict_flower', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            resultArea.innerHTML = `<span style="color: var(--error)"><i class="fas fa-exclamation-circle"></i> Error: ${data.error}</span>`;
        } else {
            resultArea.innerHTML = `
                <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 4px;">Classification Successful</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: var(--primary); font-size: 1.1rem;">${data.class}</strong>
                    <span style="background: rgba(0, 242, 254, 0.1); padding: 2px 8px; border-radius: 20px; font-size: 0.8rem;">${data.confidence}</span>
                </div>
            `;
        }
    } catch (e) {
        resultArea.innerHTML = '<span style="color: var(--error)">Inference engine unavailable.</span>';
    }
};

// --- Dynamic Data Loading ---

async function loadDynamicContent() {
    try {
        const response = await fetch('/api/content');
        const data = await response.json();
        
        // Update Hero
        if (data.hero_title) document.getElementById('hero-title').innerHTML = data.hero_title;
        if (data.hero_subtitle) document.getElementById('hero-subtitle').textContent = data.hero_subtitle;
        
        // Update Sections
        if (data.vision_title) document.getElementById('vision-title').textContent = data.vision_title;
        if (data.vision_desc) document.getElementById('vision-desc').textContent = data.vision_desc;
        
        if (data.linguistic_title) document.getElementById('linguistic-title').textContent = data.linguistic_title;
        if (data.linguistic_desc) document.getElementById('linguistic-desc').textContent = data.linguistic_desc;
        
        if (data.agentic_title) document.getElementById('agentic-title').textContent = data.agentic_title;
        if (data.agentic_desc) document.getElementById('agentic-desc').textContent = data.agentic_desc;
        
        console.log("✅ Content synchronized");
    } catch (e) {
        console.error("❌ Failed to load dynamic content:", e);
    }
}

async function loadActivityHistory() {
    const feed = document.getElementById('activity-feed');
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.length === 0) {
            feed.innerHTML = '<div class="loading-state">No recent activity found.</div>';
            return;
        }

        feed.innerHTML = '';
        data.forEach(item => {
            const date = new Date(item.timestamp).toLocaleString();
            const typeIcon = {
                'flower_classification': 'fa-microscope',
                'sentiment_analysis': 'fa-fingerprint',
                'agent_chat': 'fa-atom'
            }[item.type] || 'fa-bolt';

            const card = document.createElement('div');
            card.className = 'history-item';
            card.innerHTML = `
                <div class="history-type">
                    <i class="fas ${typeIcon}"></i> ${item.type.replace('_', ' ')}
                </div>
                <div class="history-input">"${item.input}"</div>
                <div class="history-result">${item.result}</div>
                <div class="history-time">${date}</div>
            `;
            feed.appendChild(card);
        });
    } catch (e) {
        feed.innerHTML = '<div class="loading-state" style="color: var(--error)">Failed to connect to activity logs.</div>';
    }
}

// Initial Load
window.onload = () => {
    loadDynamicContent();
    loadActivityHistory();
};

// Wrap existing actions to refresh history
const originalSendMessage = sendMessage;
sendMessage = async () => {
    await originalSendMessage();
    setTimeout(loadActivityHistory, 1000);
};

const originalAnalyzeSentiment = analyzeSentiment;
analyzeSentiment = async () => {
    await originalAnalyzeSentiment();
    setTimeout(loadActivityHistory, 1000);
};

// For flower input, we need to wrap the callback
const flowerInput = document.getElementById('flower-input');
const originalFlowerChange = flowerInput.onchange;
flowerInput.onchange = async (e) => {
    await originalFlowerChange(e);
    setTimeout(loadActivityHistory, 1000);
};
