/**
 * AI Art Restoration — Client-Side Application Logic
 * Handles file upload, sample loading, validation, API interaction,
 * step-by-step processing animations, interactive slider, and side-by-side mode.
 */
(function () {
    'use strict';

    // --- Constants ---
    const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB
    const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
    const ALLOWED_EXTS  = ['.jpg', '.jpeg', '.png', '.webp'];

    // --- DOM References ---
    const $  = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const dropZone          = $('#drop-zone');
    const fileInput         = $('#file-input');
    const sampleArtBtn      = $('#sample-art-btn');
    
    const uploadSection     = $('#upload-section');
    const previewSection    = $('#preview-section');
    const processingSection = $('#processing-section');
    const resultSection     = $('#result-section');
    const errorSection      = $('#error-section');

    const previewImage      = $('#preview-image');
    const fileName          = $('#file-name');
    const fileSize          = $('#file-size');
    const fileDims          = $('#file-dims');
    const restoreBtn        = $('#restore-btn');
    const cancelBtn         = $('#cancel-btn');

    const processingThumbnail = $('#processing-thumbnail');
    const processingPhaseText = $('#processing-phase-text');
    const pstep1              = $('#pstep-1');
    const pstep2              = $('#pstep-2');
    const pstep3              = $('#pstep-3');

    const modeSliderBtn       = $('#mode-slider-btn');
    const modeSideBtn         = $('#mode-side-btn');
    const sliderViewContainer = $('#slider-view-container');
    const sideViewContainer   = $('#side-view-container');

    const beforeImage         = $('#before-image');
    const afterImage          = $('#after-image');
    const sideBeforeImage     = $('#side-before-image');
    const sideAfterImage      = $('#side-after-image');
    
    const comparisonSlider    = $('#comparison-slider');
    const comparisonWrapper   = $('#comparison-wrapper');
    const resultDetails       = $('#result-details');
    const downloadBtn         = $('#download-btn');
    const newImageBtn         = $('#new-image-btn');

    const errorMessage        = $('#error-message');
    const retryBtn           = $('#retry-btn');

    // --- State ---
    let currentFile       = null;
    let currentOutputId   = null;
    let isProcessing      = false;
    let stepTimer         = null;
    let activeViewMode    = 'slider'; // 'slider' | 'side'

    // ========== UTILITY FUNCTIONS ==========

    function formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function showSection(section) {
        [uploadSection, previewSection, processingSection, resultSection, errorSection]
            .forEach(s => s.classList.add('hidden'));
        section.classList.remove('hidden');
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function validateFile(file) {
        if (!file) return 'No file selected.';

        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTS.includes(ext))
            return `Invalid format "${ext}". Accepted: JPG, PNG, WEBP.`;

        if (!ALLOWED_TYPES.includes(file.type) && file.type !== '')
            return `Invalid MIME type "${file.type}".`;

        if (file.size > MAX_FILE_SIZE)
            return `File too large (${formatBytes(file.size)}). Max: 20 MB.`;

        if (file.size === 0)
            return 'File is empty.';

        return null; // valid
    }

    // ========== FILE HANDLING ==========

    function handleFile(file) {
        const error = validateFile(file);
        if (error) {
            showError(error);
            return;
        }

        currentFile = file;

        // Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.onload = () => {
                fileDims.textContent = `${previewImage.naturalWidth} × ${previewImage.naturalHeight} px`;
            };
            previewImage.src = e.target.result;
            if (processingThumbnail) {
                processingThumbnail.src = e.target.result;
            }
        };
        reader.readAsDataURL(file);

        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);
        fileDims.textContent = 'Analyzing...';

        showSection(previewSection);
    }

    // Load Sample Artwork
    if (sampleArtBtn) {
        sampleArtBtn.addEventListener('click', async () => {
            try {
                const res = await fetch('/static/samples/sample_art.jpg');
                const blob = await res.blob();
                const sampleFile = new File([blob], 'vintage_portrait_degraded.jpg', { type: 'image/jpeg' });
                handleFile(sampleFile);
            } catch (err) {
                showError('Failed to load sample artwork.');
            }
        });
    }

    // ========== DRAG & DROP ==========

    dropZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    // Click-to-browse (prevent double-triggering on mobile touch)
    dropZone.addEventListener('click', (e) => {
        if (e.target !== fileInput) {
            fileInput.click();
        }
    });
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // ========== RESTORE PROCESS ==========

    restoreBtn.addEventListener('click', () => {
        if (!currentFile || isProcessing) return;
        startRestore();
    });

    function startProcessingAnimations() {
        if (pstep1) pstep1.className = 'step-item active';
        if (pstep2) pstep2.className = 'step-item';
        if (pstep3) pstep3.className = 'step-item';

        if (processingPhaseText) processingPhaseText.textContent = 'Step 1/3: Analyzing canvas & scratch artifacts...';

        stepTimer = setTimeout(() => {
            if (pstep2) pstep2.className = 'step-item active';
            if (processingPhaseText) processingPhaseText.textContent = 'Step 2/3: Executing Real-ESRGAN super-resolution model...';
            
            stepTimer = setTimeout(() => {
                if (pstep3) pstep3.className = 'step-item active';
                if (processingPhaseText) processingPhaseText.textContent = 'Step 3/3: Reconstructing facial features with GFPGAN...';
            }, 2500);
        }, 1800);
    }

    function stopProcessingAnimations() {
        if (stepTimer) clearTimeout(stepTimer);
    }

    async function startRestore() {
        isProcessing = true;
        restoreBtn.disabled = true;
        startProcessingAnimations();
        showSection(processingSection);

        const formData = new FormData();
        formData.append('image', currentFile);
        const outscaleSelect = $('#outscale-select');
        if (outscaleSelect) {
            formData.append('outscale', outscaleSelect.value);
            formData.append('enable_faces', outscaleSelect.value === '4' ? 'true' : 'false');
        }

        try {
            const response = await fetch('/api/restore', {
                method: 'POST',
                body: formData,
            });

            let data;
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                data = await response.json();
            } else {
                const text = await response.text();
                if (text.includes('loca.lt') || text.includes('tunnel') || text.includes('Bypass')) {
                    throw new Error('Tunnel verification required. Please visit the tunnel URL in your browser first to click "Click to Continue".');
                }
                throw new Error(`Server returned HTML error (${response.status} ${response.statusText}).`);
            }

            if (!response.ok || !data.success) {
                throw new Error(data.error || `Restoration failed (${response.status})`);
            }

            currentOutputId = data.output_id;
            showResult(data);

        } catch (err) {
            showError(err.message || 'An unexpected error occurred during restoration.');
        } finally {
            isProcessing = false;
            restoreBtn.disabled = false;
            stopProcessingAnimations();
        }
    }

    // ========== RESULTS & VIEW MODES ==========

    function showResult(data) {
        // Set images
        beforeImage.src = previewImage.src;
        if (sideBeforeImage) sideBeforeImage.src = previewImage.src;

        const resultUrl = `/api/result/${data.output_id}?t=${Date.now()}`;
        
        afterImage.onload = null;
        afterImage.onerror = null;

        afterImage.onload = () => {
            setSliderPosition(50);
        };

        afterImage.onerror = () => {
            showError('Failed to load the restored master artwork.');
        };

        afterImage.src = resultUrl;
        if (sideAfterImage) sideAfterImage.src = resultUrl;

        setSliderPosition(50);
        showSection(resultSection);

        // Display technical details badges
        const details = data.details || {};
        const outscaleVal = $('#outscale-select') ? $('#outscale-select').value : '4';
        
        let badges = '';
        badges += `<span class="detail-badge success"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Resolution Enhanced (${outscaleVal}x Scale)</span>`;
        
        if (details.has_faces) {
            badges += '<span class="detail-badge success">Facial Features Detected &amp; Restored</span>';
        } else {
            badges += '<span class="detail-badge">No facial features detected</span>';
        }
        
        if (details.device) {
            badges += `<span class="detail-badge">Hardware: ${details.device.toUpperCase()}</span>`;
        }
        if (details.processing_time) {
            badges += `<span class="detail-badge">Processing Time: ${details.processing_time}s</span>`;
        }
        
        resultDetails.innerHTML = badges;
    }

    // View Mode Switcher
    if (modeSliderBtn && modeSideBtn) {
        modeSliderBtn.addEventListener('click', () => {
            activeViewMode = 'slider';
            modeSliderBtn.classList.add('active');
            modeSliderBtn.setAttribute('aria-pressed', 'true');
            modeSideBtn.classList.remove('active');
            modeSideBtn.setAttribute('aria-pressed', 'false');

            sliderViewContainer.classList.remove('hidden');
            sideViewContainer.classList.add('hidden');
        });

        modeSideBtn.addEventListener('click', () => {
            activeViewMode = 'side';
            modeSideBtn.classList.add('active');
            modeSideBtn.setAttribute('aria-pressed', 'true');
            modeSliderBtn.classList.remove('active');
            modeSliderBtn.setAttribute('aria-pressed', 'false');

            sideViewContainer.classList.remove('hidden');
            sliderViewContainer.classList.add('hidden');
        });
    }

    // ========== COMPARISON SLIDER ==========

    function setSliderPosition(percent) {
        percent = Math.max(0, Math.min(100, percent));
        afterImage.style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
        comparisonSlider.style.left = percent + '%';
        comparisonSlider.setAttribute('aria-valuenow', Math.round(percent));
    }

    let isDragging = false;

    function getSliderPercent(clientX) {
        const rect = comparisonWrapper.getBoundingClientRect();
        return ((clientX - rect.left) / rect.width) * 100;
    }

    comparisonWrapper.addEventListener('mousedown', (e) => {
        isDragging = true;
        setSliderPosition(getSliderPercent(e.clientX));
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        e.preventDefault();
        setSliderPosition(getSliderPercent(e.clientX));
    });

    document.addEventListener('mouseup', () => { isDragging = false; });

    // Touch support
    comparisonWrapper.addEventListener('touchstart', (e) => {
        isDragging = true;
        setSliderPosition(getSliderPercent(e.touches[0].clientX));
    }, { passive: true });

    document.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        setSliderPosition(getSliderPercent(e.touches[0].clientX));
    }, { passive: true });

    document.addEventListener('touchend', () => { isDragging = false; });

    // Keyboard support for slider
    comparisonSlider.addEventListener('keydown', (e) => {
        const current = parseFloat(comparisonSlider.getAttribute('aria-valuenow') || 50);
        const step = 2;
        if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
            e.preventDefault();
            setSliderPosition(current - step);
        } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
            e.preventDefault();
            setSliderPosition(current + step);
        }
    });

    // ========== DOWNLOAD ==========

    downloadBtn.addEventListener('click', () => {
        if (!currentOutputId) return;
        const link = document.createElement('a');
        link.href = `/api/download/${currentOutputId}`;
        link.download = `restored_master_${currentOutputId}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // ========== RESET / CANCEL ==========

    function resetApp() {
        currentFile     = null;
        currentOutputId = null;
        isProcessing    = false;
        stopProcessingAnimations();

        // Clear image handlers
        previewImage.onload     = null;
        previewImage.onerror    = null;
        afterImage.onload       = null;
        afterImage.onerror      = null;
        beforeImage.onload      = null;
        beforeImage.onerror     = null;

        try { fileInput.value = ''; } catch (e) {}

        previewImage.removeAttribute('src');
        beforeImage.removeAttribute('src');
        afterImage.removeAttribute('src');
        if (sideBeforeImage) sideBeforeImage.removeAttribute('src');
        if (sideAfterImage) sideAfterImage.removeAttribute('src');

        resultDetails.innerHTML = '';
        restoreBtn.disabled = false;
        showSection(uploadSection);
    }

    cancelBtn.addEventListener('click', resetApp);
    newImageBtn.addEventListener('click', resetApp);
    retryBtn.addEventListener('click', resetApp);

    // ========== ERROR HANDLING ==========

    function showError(message) {
        stopProcessingAnimations();
        errorMessage.textContent = message;
        showSection(errorSection);
    }

})();
