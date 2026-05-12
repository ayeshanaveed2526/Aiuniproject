// ===== PARTICLE BACKGROUND =====
const canvas = document.getElementById('particle-canvas');
const ctx = canvas.getContext('2d');
let particles = [];
let mouse = { x: null, y: null };
let animId;

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

document.addEventListener('mousemove', e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
});

class Particle {
    constructor() { this.reset(); }
    reset() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 0.5;
        this.speedX = (Math.random() - 0.5) * 0.5;
        this.speedY = (Math.random() - 0.5) * 0.5;
        this.opacity = Math.random() * 0.5 + 0.1;
    }
    update() {
        this.x += this.speedX;
        this.y += this.speedY;
        if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
        if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
        // Mouse interaction
        if (mouse.x && mouse.y) {
            const dx = mouse.x - this.x;
            const dy = mouse.y - this.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 150) {
                const force = (150 - dist) / 150;
                this.x -= dx * force * 0.01;
                this.y -= dy * force * 0.01;
            }
        }
    }
    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 242, 254, ${this.opacity})`;
        ctx.fill();
    }
}

function initParticles(count = 80) {
    particles = [];
    for (let i = 0; i < count; i++) particles.push(new Particle());
}

function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 150) {
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.strokeStyle = `rgba(0, 242, 254, ${0.06 * (1 - dist / 150)})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }
}

function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => { p.update(); p.draw(); });
    drawConnections();
    animId = requestAnimationFrame(animateParticles);
}
initParticles();
animateParticles();

// ===== TOAST NOTIFICATIONS =====
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle' };
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i> ${message}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ===== SCROLL REVEAL =====
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const delay = entry.target.dataset.delay || 0;
                setTimeout(() => entry.target.classList.add('visible'), parseInt(delay));
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

// ===== NAV SHRINK ON SCROLL =====
function initNavScroll() {
    const nav = document.querySelector('nav');
    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 100);
    });
}

// ===== DRAG & DROP ZONE =====
function initDragDrop() {
    const zone = document.getElementById('drop-zone');
    const input = document.getElementById('flower-input');
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        }
    });
}

// ===== CHARACTER COUNTER =====
function initCharCounter() {
    const textarea = document.getElementById('sentiment-input');
    const counter = document.getElementById('char-count');
    if (!textarea || !counter) return;
    textarea.addEventListener('input', () => {
        const len = textarea.value.length;
        counter.textContent = len;
        counter.parentElement.classList.toggle('warning', len > 1500);
        counter.parentElement.classList.toggle('danger', len > 1900);
    });
}

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

    // Add typing indicator
    const thinkingId = 'thinking_' + Date.now();
    const thinkingDiv = document.createElement('div');
    thinkingDiv.id = thinkingId;
    thinkingDiv.className = 'message bot-message typing-indicator';
    thinkingDiv.innerHTML = '<span></span><span></span><span></span>';
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
            showToast('Copied to clipboard', 'success');
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
    showToast('Chat history cleared', 'info');
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
            showToast(`Sentiment analysis: ${data.label}`, 'success');
            
            // Define colors and icons based on sentiment
            let color = 'var(--primary)';
            let icon = 'fa-meh';
            let meterWidth = '50%';

            if (label.includes('pos')) { 
                color = '#00f2fe'; // Positive cyan
                icon = 'fa-smile-beam';
                meterWidth = data.score !== 'N/A' ? data.score : '90%';
            } else if (label.includes('neg')) { 
                color = '#ff4b2b'; // Negative red
                icon = 'fa-frown-open';
                meterWidth = data.score !== 'N/A' ? data.score : '10%';
            }

            resultArea.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.9rem;">Analysis: <strong style="color: ${color}; text-transform: uppercase;">${data.label}</strong></span>
                    <i class="fas ${icon}" style="color: ${color}; font-size: 1.2rem;"></i>
                </div>
                <div class="sentiment-meter" style="height: 6px; background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden; margin-bottom: 8px;">
                    <div class="sentiment-fill" style="width: ${meterWidth}; height: 100%; background: linear-gradient(90deg, ${color}, ${color}dd); transition: width 1s ease-out;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; opacity: 0.6;">
                    <span>Confidence: ${data.score}</span>
                    <span>Context-Aware Engine</span>
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
    document.getElementById('drop-zone').classList.add('has-image');
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
            showToast(`Identified as ${data.class} (${data.confidence})`, 'success');
            
            let badgeColor = 'rgba(0, 242, 254, 0.1)';
            if (data.raw_confidence < 0.6) badgeColor = 'rgba(255, 165, 0, 0.2)';

            resultArea.innerHTML = `
                <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 4px;">Classification Successful</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <strong style="color: var(--primary); font-size: 1.1rem;">${data.class}</strong>
                    <span style="background: ${badgeColor}; padding: 2px 8px; border-radius: 20px; font-size: 0.8rem;">${data.confidence}</span>
                </div>
                <button class="action-btn" style="background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); font-size: 0.8rem; padding: 8px;" onclick="verifyWithAgent()">
                    <i class="fas fa-robot"></i> Verify with Agentic AI
                </button>
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

        // Clear skeleton
        feed.innerHTML = '';
        
        if (data.length === 0) {
            feed.innerHTML = '<div class="loading-state">No recent activity found.</div>';
            return;
        }
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

// --- Agent Verification for Flower ---
async function verifyWithAgent() {
    const input = document.getElementById('flower-input');
    const resultArea = document.getElementById('flower-result');
    const file = input.files[0];
    
    if (!file) return;

    toggleChat();
    appendMessage('bot', `🔍 **Multimodal verification initiated...** Analyzing image context for target classes...`);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/verify_flower', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            appendMessage('bot', `❌ Verification Error: ${data.error}`, true);
        } else {
            appendMessage('bot', `### Verification Results\n\n${data.verification}`);
            showToast('Agentic Verification Complete', 'success');
        }
    } catch (e) {
        appendMessage('bot', '❌ Connection error during verification.', true);
    }
}

// Initial Load
window.onload = () => {
    loadDynamicContent();
    loadActivityHistory();
    initScrollReveal();
    initNavScroll();
    initDragDrop();
    initCharCounter();
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
