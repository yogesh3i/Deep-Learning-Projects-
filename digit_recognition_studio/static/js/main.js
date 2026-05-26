/* ==========================================================================
   DIGIT RECOGNITION STUDIO — FRONTEND CONTROL LOGIC
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Elements Selection
    const canvas = document.getElementById('paint-canvas');
    const ctx = canvas.getContext('2d');
    const brushSizeInput = document.getElementById('brush-size');
    const brushSizeVal = document.getElementById('brush-size-val');
    const btnClear = document.getElementById('btn-clear');
    const btnPredict = document.getElementById('btn-predict');
    const autoPredictToggle = document.getElementById('auto-predict-toggle');
    
    const winningDigit = document.getElementById('winning-digit');
    const predictionConfidence = document.getElementById('prediction-confidence');
    const predictedDigitBox = document.querySelector('.predicted-digit-box');
    
    const detailsToggle = document.getElementById('details-toggle');
    const detailsContent = document.getElementById('details-content');
    const toggleIcon = document.querySelector('.toggle-icon');

    // 2. State Variables
    let isDrawing = false;
    let autoPredictTimer = null;
    let hasDrawn = false;

    // 3. Canvas Initialization
    // Initialize canvas with a pitch-black background
    function resetCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Stroke settings (white brush, round caps for smooth lines)
        ctx.strokeStyle = '#ffffff';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = brushSizeInput.value;
        
        hasDrawn = false;
    }
    resetCanvas();

    // 4. Drawing Coordinates Helper
    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        
        // Touch events
        if (e.touches && e.touches[0]) {
            return {
                x: e.touches[0].clientX - rect.left,
                y: e.touches[0].clientY - rect.top
            };
        }
        
        // Mouse events
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    // 5. Drawing Event Handlers
    function startDrawing(e) {
        isDrawing = true;
        hasDrawn = true;
        const coords = getCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
        
        // Cancel any pending auto-predictions while active drawing is occurring
        if (autoPredictTimer) {
            clearTimeout(autoPredictTimer);
        }
        e.preventDefault();
    }

    function draw(e) {
        if (!isDrawing) return;
        const coords = getCoordinates(e);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        e.preventDefault();
    }

    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.closePath();
        
        // Trigger auto-prediction if toggled and user drew something
        if (autoPredictToggle.checked && hasDrawn) {
            triggerAutoPredict();
        }
    }

    // Bind Mouse Events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);
    
    // Bind Touch Events (Mobile/Tablet support)
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    window.addEventListener('touchend', stopDrawing);

    // Brush Size Adjustments
    brushSizeInput.addEventListener('input', (e) => {
        const size = e.target.value;
        brushSizeVal.textContent = `${size}px`;
        ctx.lineWidth = size;
    });

    // 6. Predict / Inference Request
    async function classifyDrawing() {
        if (!hasDrawn) return;
        
        // Convert canvas image to Base64
        const dataURL = canvas.toDataURL('image/png');
        
        try {
            // Display visual loading states in prediction boxes
            winningDigit.classList.add('loading-fade');
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            });

            if (!response.ok) {
                throw new Error('Prediction API failed');
            }

            const data = await response.json();
            updatePredictionUI(data);
            
        } catch (error) {
            console.error('[-] Error during classification request:', error);
            winningDigit.textContent = 'Err';
            predictionConfidence.textContent = 'Connection Error';
            predictionConfidence.style.color = 'var(--color-error)';
        } finally {
            winningDigit.classList.remove('loading-fade');
        }
    }

    // Debounced Auto-Predict Trigger
    function triggerAutoPredict() {
        if (autoPredictTimer) {
            clearTimeout(autoPredictTimer);
        }
        autoPredictTimer = setTimeout(() => {
            classifyDrawing();
        }, 350); // 350ms delay after drawing finishes
    }

    // 7. UI Rendering Functions
    function updatePredictionUI(data) {
        // 1. Update winning prediction display
        const digit = data.prediction;
        const confidence = data.confidences[digit];
        
        winningDigit.textContent = digit;
        predictionConfidence.textContent = `${(confidence * 100).toFixed(1)}% Confidence`;
        
        // Play subtle pop animation on a prediction change
        predictedDigitBox.classList.remove('pulse-active');
        void predictedDigitBox.offsetWidth; // Trigger reflow to restart CSS animation
        predictedDigitBox.classList.add('pulse-active');

        // Color coding depending on model certainty
        if (confidence > 0.8) {
            predictionConfidence.style.color = 'var(--color-success)';
        } else if (confidence > 0.5) {
            predictionConfidence.style.color = 'var(--color-warning)';
        } else {
            predictionConfidence.style.color = 'var(--color-error)';
        }

        // 2. Update all probability bars
        for (let i = 0; i < 10; i++) {
            const prob = data.confidences[i];
            const row = document.getElementById(`prob-row-${i}`);
            const fill = document.getElementById(`bar-fill-${i}`);
            const label = document.getElementById(`perc-label-${i}`);
            
            // Set width percentage
            fill.style.width = `${(prob * 100).toFixed(1)}%`;
            label.textContent = `${(prob * 100).toFixed(0)}%`;
            
            // Toggle highlight on winning row
            if (i === digit) {
                row.classList.add('winning-row');
            } else {
                row.classList.remove('winning-row');
            }
        }

        // 3. Render CNN activations (Feature Maps)
        if (data.activation1) {
            const placeholder1 = document.getElementById('act-placeholder-1');
            const img1 = document.getElementById('activation-img-1');
            placeholder1.classList.add('hidden');
            img1.src = data.activation1;
            img1.classList.remove('hidden');
        }
        
        if (data.activation2) {
            const placeholder2 = document.getElementById('act-placeholder-2');
            const img2 = document.getElementById('activation-img-2');
            placeholder2.classList.add('hidden');
            img2.src = data.activation2;
            img2.classList.remove('hidden');
        }
    }

    // 8. Control Panel Handlers
    // Clear canvas button
    btnClear.addEventListener('click', () => {
        resetCanvas();
        
        // Reset UI labels
        winningDigit.textContent = '-';
        predictionConfidence.textContent = '0.0% Confidence';
        predictionConfidence.style.color = 'var(--text-secondary)';
        predictedDigitBox.classList.remove('pulse-active');

        // Clear probability bars
        for (let i = 0; i < 10; i++) {
            document.getElementById(`prob-row-${i}`).classList.remove('winning-row');
            document.getElementById(`bar-fill-${i}`).style.width = '0%';
            document.getElementById(`perc-label-${i}`).textContent = '0%';
        }

        // Re-display activation explorer placeholders
        document.getElementById('act-placeholder-1').classList.remove('hidden');
        document.getElementById('activation-img-1').classList.add('hidden');
        document.getElementById('activation-img-1').src = '';
        
        document.getElementById('act-placeholder-2').classList.remove('hidden');
        document.getElementById('activation-img-2').classList.add('hidden');
        document.getElementById('activation-img-2').src = '';
    });

    // Manual classify button
    btnPredict.addEventListener('click', () => {
        classifyDrawing();
    });

    // 9. Diagnostics Panel Slide Toggle
    detailsToggle.addEventListener('click', () => {
        detailsContent.classList.toggle('hidden');
        toggleIcon.classList.toggle('rotate-180');
    });
});
