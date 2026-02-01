// Main JavaScript for Skin Cancer Detection App

let selectedFile = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const analyzeAnotherBtn = document.getElementById('analyzeAnother');

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
removeImageBtn.addEventListener('click', removeImage);
analyzeBtn.addEventListener('click', analyzeImage);
if (analyzeAnotherBtn) {
    analyzeAnotherBtn.addEventListener('click', resetForm);
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = 'var(--primary-color)';
    uploadArea.style.backgroundColor = '#eff6ff';
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = 'var(--border-color)';
    uploadArea.style.backgroundColor = 'var(--light-color)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file select
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file
function handleFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowedTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPG, JPEG, or PNG)');
        return;
    }
    
    // Validate file size (16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size must be less than 16MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Remove image
function removeImage() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
}

// Analyze image
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    // Show loading
    analyzeBtn.disabled = true;
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        // Send request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Show results
            displayResults(data);
        } else {
            // Show error
            showError(data.error || 'An error occurred during analysis');
            analyzeBtn.disabled = false;
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to the server. Please try again.');
        analyzeBtn.disabled = false;
    } finally {
        loading.style.display = 'none';
    }
}

// Display results
function displayResults(data) {
    // Update prediction
    const predictionClass = document.getElementById('predictionClass');
    predictionClass.textContent = data.predicted_class.toUpperCase();
    predictionClass.className = 'label-value ' + data.predicted_class.toLowerCase();
    
    // Update confidence
    const confidence = Math.round(data.confidence * 100);
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    
    confidenceValue.textContent = confidence + '%';
    confidenceFill.style.width = confidence + '%';
    
    // Update probabilities
    const probabilitiesList = document.getElementById('probabilitiesList');
    probabilitiesList.innerHTML = '';
    
    for (const [className, probability] of Object.entries(data.class_probabilities)) {
        const probabilityItem = document.createElement('div');
        probabilityItem.className = 'probability-item';
        
        const prob = Math.round(probability * 100);
        
        probabilityItem.innerHTML = `
            <span class="probability-name">${className}</span>
            <div class="probability-bar-container">
                <div class="probability-bar" style="width: ${prob}%"></div>
            </div>
            <span class="probability-value">${prob}%</span>
        `;
        
        probabilitiesList.appendChild(probabilityItem);
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show error
function showError(message) {
    alert(message);
}

// Reset form
function resetForm() {
    removeImage();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Check server health on load
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            console.warn('Model not loaded. Please train or upload a model.');
            // You could show a warning banner here
        }
    } catch (error) {
        console.error('Failed to check server health:', error);
    }
});
