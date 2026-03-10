document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewImg = document.getElementById('preview-img');
    const filePreview = document.getElementById('file-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const scannerOverlay = document.getElementById('scanner-overlay');
    const resultsSection = document.getElementById('results');
    const diseaseNameEl = document.getElementById('disease-name');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');
    const suggestionList = document.getElementById('suggestion-list');
    const themeToggleBtn = document.getElementById('theme-toggle');
    const qaInput = document.getElementById('qa-input');
    const qaSubmit = document.getElementById('qa-submit');
    const qaStatus = document.getElementById('qa-status');
    const qaAnswer = document.getElementById('qa-answer');
    const qaLanguage = document.getElementById('qa-language');
    const qaVoiceBtn = document.getElementById('qa-voice');
    const clearImageBtn = document.getElementById('clear-image-btn');

    let lastDetection = null;

    // Simple heuristic biometric profiles per disease.
    // Values are percentages (0–100).
    const biometricProfiles = {
        'Tomato Healthy': { nitrogen: 80, moisture: 65, toxins: 5 },
        'Tomato Early Blight': { nitrogen: 60, moisture: 70, toxins: 45 },
        'Tomato Late Blight': { nitrogen: 55, moisture: 80, toxins: 60 },
        'Tomato Leaf Spot': { nitrogen: 65, moisture: 60, toxins: 40 },
        'Rice Bacterial Blight': { nitrogen: 70, moisture: 75, toxins: 55 },
        'Rice Brown Spot': { nitrogen: 68, moisture: 70, toxins: 50 },
        'Rice Leaf Blast': { nitrogen: 72, moisture: 78, toxins: 65 },
        'Rice Sheath Blight': { nitrogen: 70, moisture: 82, toxins: 58 },
    };

    function getBiometricsForDisease(diseaseName) {
        if (!diseaseName) {
            return { nitrogen: 60, moisture: 60, toxins: 30 };
        }
        const key = diseaseName.trim();
        const profile = biometricProfiles[key];
        if (profile) return profile;
        // Fallback: derive something reasonable from confidence.
        const base = lastDetection?.confidence ?? 60;
        return {
            nitrogen: Math.min(90, base + 5),
            moisture: Math.max(40, base - 10),
            toxins: Math.max(10, 100 - base),
        };
    }

    // Theme toggle (dark / light)
    function applySavedTheme() {
        const saved = window.localStorage.getItem('theme');
        if (saved === 'light') {
            document.body.setAttribute('data-theme', 'light');
            themeToggleBtn.textContent = 'Dark mode';
        } else {
            document.body.removeAttribute('data-theme');
            themeToggleBtn.textContent = 'Light mode';
        }
    }

    applySavedTheme();

    themeToggleBtn.addEventListener('click', () => {
        const isLight = document.body.getAttribute('data-theme') === 'light';
        if (isLight) {
            document.body.removeAttribute('data-theme');
            window.localStorage.setItem('theme', 'dark');
            themeToggleBtn.textContent = 'Light mode';
        } else {
            document.body.setAttribute('data-theme', 'light');
            window.localStorage.setItem('theme', 'light');
            themeToggleBtn.textContent = 'Dark mode';
        }
    });

    // Handle File Browsing
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', handleFileSelect);

    // Handle Drag over
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });

    function handleFileSelect() {
        const file = fileInput.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                filePreview.classList.remove('hidden');
                browseBtn.classList.add('hidden');
                dropZone.querySelector('h3').classList.add('hidden');
                dropZone.querySelector('p').classList.add('hidden');
                dropZone.querySelector('.upload-icon').classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }
    }

    if (clearImageBtn) {
        clearImageBtn.addEventListener('click', () => {
            // Clear file input and preview
            fileInput.value = '';
            previewImg.src = '';
            filePreview.classList.add('hidden');
            browseBtn.classList.remove('hidden');
            dropZone.querySelector('h3').classList.remove('hidden');
            dropZone.querySelector('p').classList.remove('hidden');
            dropZone.querySelector('.upload-icon').classList.remove('hidden');

            // Reset results and nav link
            resultsSection.classList.add('hidden');
            const resultsNav = document.querySelector('.nav-links a[href="#results"]');
            if (resultsNav) {
                resultsNav.classList.add('disabled-link');
            }

            // Reset confidence and biometrics
            confidenceFill.style.width = '0%';
            confidenceValue.textContent = '0%';
            document.querySelectorAll('.bar-fill').forEach(bar => {
                bar.style.height = '0%';
            });

            lastDetection = null;
        });
    }

    // Handle Analysis
    analyzeBtn.addEventListener('click', async () => {
        // Show scanner overlay
        scannerOverlay.classList.remove('hidden');

        try {
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select an image first.');
                scannerOverlay.classList.add('hidden');
                return;
            }

            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch('/api/detect-disease', {
                method: 'POST',
                body: formData
            });

            const raw = await response.text();
            let result = {};
            try {
                result = raw ? JSON.parse(raw) : {};
            } catch (e) {
                console.error('Non‑JSON response from /api/detect-disease:', raw);
                alert('Server returned an invalid response while analyzing the image.');
                scannerOverlay.classList.add('hidden');
                return;
            }

            if (!response.ok || result.error) {
                console.error('Detect disease error:', result.error || response.statusText);
                alert('Error while analyzing image: ' + (result.error || `HTTP ${response.status}`));
                scannerOverlay.classList.add('hidden');
                return;
            }

            scannerOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            document.querySelector('.nav-links a[href="#results"]').classList.remove('disabled-link');

            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });

            // Populate Data from Backend
            displayResults(result);

            // Remember last detection for Q&A context
            lastDetection = result;

        } catch (err) {
            console.error(err);
            alert("An unexpected error occurred during analysis.");
            scannerOverlay.classList.add('hidden');
        }
    });

    function displayResults(data) {
        if (data.low_confidence) {
            diseaseNameEl.textContent = 'Uncertain result – please try another image';
        } else {
            diseaseNameEl.textContent = data.disease;
        }

        // Animate Confidence Bar
        setTimeout(() => {
            confidenceFill.style.width = `${data.confidence}%`;

            // Number counter animation
            let current = 0;
            const target = data.confidence;
            const interval = setInterval(() => {
                if (current >= target) {
                    clearInterval(interval);
                    confidenceValue.textContent = `${target}%`;
                } else {
                    current += 2;
                    confidenceValue.textContent = `${current}%`;
                }
            }, 30);
        }, 500);

        // Populate suggestions (fertilizer + treatment), or low-confidence message
        suggestionList.innerHTML = '';
        if (data.low_confidence) {
            const msg = data.message || 'Confidence is low. Please upload a clearer image or consult a local agriculture officer.';
            const li = document.createElement('li');
            li.innerHTML = `<i class="ph ph-warning-circle glowing-icon" style="font-size: 1.2rem;"></i> <span>${msg}</span>`;
            suggestionList.appendChild(li);
        } else {
            data.suggestions.forEach((sug, index) => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="ph ph-check-circle glowing-icon" style="font-size: 1.2rem;"></i> <span>${sug}</span>`;
                li.style.opacity = '0';
                li.style.transform = 'translateY(10px)';
                li.style.transition = 'all 0.5s ease';
                suggestionList.appendChild(li);

                setTimeout(() => {
                    li.style.opacity = '1';
                    li.style.transform = 'translateY(0)';
                }, 300 * index);
            });
        }

        // Animate Charts with disease‑specific biometric data
        const biometrics = getBiometricsForDisease(data.disease);
        const bars = document.querySelectorAll('.bar-fill');
        bars.forEach(bar => {
            const metric = bar.getAttribute('data-metric');
            const targetValue = biometrics[metric] ?? 0;
            const target = `${targetValue}%`;
            setTimeout(() => {
                bar.style.height = target;
            }, 500);
        });
    }

    // Gemini Q&A
    qaSubmit.addEventListener('click', async () => {
        const question = (qaInput.value || '').trim();
        if (!question) {
            qaStatus.textContent = 'Please type a question first.';
            qaAnswer.classList.add('hidden');
            return;
        }

        if (!lastDetection) {
            qaStatus.textContent = 'Please analyze a plant image first.';
            qaAnswer.classList.add('hidden');
            return;
        }

        if (lastDetection.low_confidence) {
            const language = qaLanguage?.value || 'en';
            qaStatus.textContent = language === 'te'
                ? 'ముందు స్పష్టమైన ఆకుల ఫోటోను మళ్లీ స్కాన్ చేయండి. ఈ ఫోటోపై జెమిని నమ్మకంగా చెప్పలేకపోతుంది.'
                : 'The last scan was low confidence. Please upload a clearer image and scan again before asking Gemini.';
            qaAnswer.classList.add('hidden');
            return;
        }

        const language = qaLanguage?.value || 'en';

        qaStatus.textContent = language === 'te'
            ? 'జెమిని సమాధానం ఇస్తోంది...'
            : 'Asking Gemini...';
        qaAnswer.classList.add('hidden');
        qaSubmit.disabled = true;

        try {
            const res = await fetch('/api/ask-disease', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question,
                    disease: lastDetection.disease,
                    suggestions: lastDetection.suggestions,
                    language
                })
            });

            const raw = await res.text();
            let data = {};
            try {
                data = raw ? JSON.parse(raw) : {};
            } catch (e) {
                console.error('Non‑JSON response from /api/ask-disease:', raw);
                qaStatus.textContent = 'Server returned an invalid response.';
                qaAnswer.classList.add('hidden');
                return;
            }

            if (!res.ok || data.error) {
                qaStatus.textContent = data.error || (language === 'te'
                    ? 'జెమిని నుండి సమాధానం రాలేదు.'
                    : 'Gemini request failed.');
                qaAnswer.classList.add('hidden');
                return;
            }

            qaStatus.textContent = '';
            qaAnswer.textContent = data.answer;
            qaAnswer.classList.remove('hidden');
        } catch (err) {
            console.error(err);
            qaStatus.textContent = language === 'te'
                ? 'జెమినితో మాట్లాడేటప్పుడు సమస్య వచ్చింది.'
                : 'Something went wrong while talking to Gemini.';
            qaAnswer.classList.add('hidden');
        } finally {
            qaSubmit.disabled = false;
        }
    });

    if (qaVoiceBtn) {
        qaVoiceBtn.addEventListener('click', () => {
            const language = qaLanguage?.value || 'en';
            const text = qaAnswer.textContent || '';
            if (!text) {
                alert(language === 'te'
                    ? 'ముందు ప్రశ్న అడిగి సమాధానం తెచ్చుకోండి.'
                    : 'Please ask a question and get an answer first.');
                return;
            }

            if (!('speechSynthesis' in window)) {
                alert(language === 'te'
                    ? 'ఈ బ్రౌజర్‌లో వాయిస్ సపోర్ట్ లేదు.'
                    : 'Your browser does not support speech synthesis.');
                return;
            }

            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            if (language === 'te') {
                utterance.lang = 'te-IN';
            }
            utterance.rate = 1;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
        });
    }
});
