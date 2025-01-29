document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const predictBtn = document.getElementById('predictBtn');
    const selectedFileName = document.getElementById('selectedFileName');
    const resultsContainer = document.querySelector('.results-container');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const tumorType = document.getElementById('tumorType');
    const confidence = document.getElementById('confidence');
    const explanationImage = document.getElementById('explanationImage');
    const loadingText = document.querySelector('.loading-overlay p');

    let selectedFile = null;

    // File Upload Handling
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            selectedFile = files[0];
            
            // Check file type and size
            if (!selectedFile.type.match('image.*')) {
                showError('Please upload an image file (JPEG, PNG, etc.)');
                resetUpload();
                return;
            }

            if (selectedFile.size > 5 * 1024 * 1024) {
                showError('File size should be less than 5MB');
                resetUpload();
                return;
            }

            // Update UI
            selectedFileName.textContent = selectedFile.name;
            predictBtn.disabled = false;
            dropZone.classList.add('file-selected');
            
            // Show preview with loading indicator
            const reader = new FileReader();
            
            reader.onloadstart = () => {
                dropZone.innerHTML = '<div class="preview-loading">Loading preview...</div>';
            };
            
            reader.onload = function(e) {
                const previewImg = document.createElement('img');
                previewImg.src = e.target.result;
                previewImg.classList.add('preview-image');
                
                previewImg.onload = () => {
                    dropZone.innerHTML = '';
                    dropZone.appendChild(previewImg);
                };
            };
            
            reader.onerror = () => {
                showError('Error loading image preview');
                resetUpload();
            };

            reader.readAsDataURL(selectedFile);
        }
    }

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        try {
            // Update UI for processing state
            loadingOverlay.style.display = 'flex';
            predictBtn.disabled = true;
            predictBtn.innerHTML = `
                <div class="spinner-border spinner-border-sm" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span class="ms-2">Analyzing...</span>
            `;
            resultsContainer.style.display = 'none';

            // Prepare and send data
            const formData = new FormData();
            formData.append('image', selectedFile);

            loadingText.textContent = 'Preprocessing image...';
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            loadingText.textContent = 'Analyzing results...';
            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            // Update results
            tumorType.textContent = result.tumor_type.toUpperCase();
            confidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
            
            // Show enhanced image if available
            if (result.enhanced_image) {
                const enhancedImg = new Image();
                enhancedImg.onload = () => {
                    dropZone.innerHTML = '';
                    enhancedImg.classList.add('preview-image');
                    dropZone.appendChild(enhancedImg);
                };
                enhancedImg.src = result.enhanced_image;
            }
            
            // Show explanation if available
            if (result.explanation) {
                explanationImage.onload = () => {
                    explanationImage.style.display = 'block';
                };
                explanationImage.src = result.explanation;
            }

            // Animate results container
            resultsContainer.style.display = 'block';
            requestAnimationFrame(() => {
                resultsContainer.style.opacity = '1';
            });

        } catch (error) {
            showError(error.message || 'An error occurred during analysis');
        } finally {
            loadingOverlay.style.display = 'none';
            predictBtn.disabled = false;
            predictBtn.innerHTML = 'Predict';
            loadingText.textContent = 'Analyzing Image...';
        }
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        selectedFileName.textContent = '';
        predictBtn.disabled = true;
        dropZone.classList.remove('file-selected');
        dropZone.innerHTML = `
            <img src="/static/images/upload-icon.png" alt="Upload" class="upload-icon">
            <p>Drag & Drop or Click to Upload MRI Scan</p>
        `;
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        const container = document.querySelector('.upload-container');
        const existingError = container.querySelector('.error-message');
        
        if (existingError) {
            existingError.remove();
        }
        
        container.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.classList.add('fade-out');
            setTimeout(() => errorDiv.remove(), 300);
        }, 4700);
    }
});