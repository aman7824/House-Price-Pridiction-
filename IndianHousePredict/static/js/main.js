// Main JavaScript for Indian House Price Prediction

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');
const resultCard = document.getElementById('resultCard');
const infoCard = document.getElementById('infoCard');

// Result elements
const predictedPrice = document.getElementById('predictedPrice');
const lowerBound = document.getElementById('lowerBound');
const upperBound = document.getElementById('upperBound');
const modelName = document.getElementById('modelName');
const rmseValue = document.getElementById('rmseValue');
const maeValue = document.getElementById('maeValue');
const r2Value = document.getElementById('r2Value');

// Form submission handler
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Hide previous results and errors
    hideAllResults();
    
    // Show loading spinner
    loadingSpinner.classList.remove('d-none');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting...';
    
    try {
        // Collect form data
        const formData = {
            bhk: parseInt(document.getElementById('bhk').value),
            area_sqft: parseInt(document.getElementById('area_sqft').value),
            city: document.getElementById('city').value,
            locality: document.getElementById('locality').value,
            age: parseInt(document.getElementById('age').value),
            floor: parseInt(document.getElementById('floor').value),
            bathrooms: parseInt(document.getElementById('bathrooms').value),
            balconies: parseInt(document.getElementById('balconies').value),
            parking: document.getElementById('parking').checked ? 1 : 0,
            lift: document.getElementById('lift').checked ? 1 : 0
        };
        
        // Validate form data
        if (!validateFormData(formData)) {
            throw new Error('Please fill all required fields correctly');
        }
        
        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        // Hide loading spinner and re-enable button
        loadingSpinner.classList.add('d-none');
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-calculator me-2"></i>Predict Price';
    }
});

// Validate form data
function validateFormData(data) {
    // Check all required fields
    const requiredFields = ['bhk', 'area_sqft', 'city', 'locality', 'age', 'floor', 'bathrooms', 'balconies'];
    
    for (const field of requiredFields) {
        if (data[field] === null || data[field] === undefined || data[field] === '') {
            return false;
        }
    }
    
    // Validate numeric ranges
    if (data.bhk < 1 || data.bhk > 10) return false;
    if (data.area_sqft < 100 || data.area_sqft > 10000) return false;
    if (data.age < 0 || data.age > 100) return false;
    if (data.floor < 0 || data.floor > 50) return false;
    if (data.bathrooms < 1 || data.bathrooms > 10) return false;
    if (data.balconies < 0 || data.balconies > 10) return false;
    
    return true;
}

// Display prediction results
function displayResults(data) {
    // Hide info card and error
    infoCard.classList.add('d-none');
    errorAlert.classList.add('d-none');
    
    // Update result values
    predictedPrice.textContent = data.formatted || '₹ 0';
    lowerBound.textContent = data.lower_formatted || '₹ 0';
    upperBound.textContent = data.upper_formatted || '₹ 0';
    modelName.textContent = data.model || 'Unknown';
    rmseValue.textContent = formatNumber(data.rmse);
    maeValue.textContent = formatNumber(data.mae);
    r2Value.textContent = data.r2 ? data.r2.toFixed(4) : '-';
    
    // Show result card with animation
    resultCard.classList.remove('d-none');
    resultCard.classList.add('pulse');
    
    // Remove pulse animation after 2 seconds
    setTimeout(() => {
        resultCard.classList.remove('pulse');
    }, 2000);
    
    // Scroll to results on mobile
    if (window.innerWidth < 768) {
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorAlert.classList.remove('d-none');
    infoCard.classList.add('d-none');
    resultCard.classList.add('d-none');
    
    // Scroll to error on mobile
    if (window.innerWidth < 768) {
        errorAlert.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Hide all result elements
function hideAllResults() {
    infoCard.classList.add('d-none');
    errorAlert.classList.add('d-none');
    resultCard.classList.add('d-none');
}

// Format number with Indian comma notation
function formatNumber(num) {
    if (!num) return '-';
    
    // Convert to number if string
    num = parseFloat(num);
    
    // Indian numbering system
    const numStr = num.toFixed(0);
    const lastThree = numStr.substring(numStr.length - 3);
    const otherNumbers = numStr.substring(0, numStr.length - 3);
    
    if (otherNumbers !== '') {
        return otherNumbers.replace(/\B(?=(\d{2})+(?!\d))/g, ',') + ',' + lastThree;
    } else {
        return lastThree;
    }
}

// City change handler - update localities based on city
document.getElementById('city').addEventListener('change', (e) => {
    const city = e.target.value;
    const localitySelect = document.getElementById('locality');
    
    // Default Bangalore localities
    const localities = {
        'Bangalore': [
            'Whitefield', 'Koramangala', 'Indiranagar', 'HSR Layout', 
            'Electronic City', 'Marathahalli', 'Bellandur'
        ],
        'Mumbai': [
            'Andheri', 'Bandra', 'Powai', 'Thane', 'Malad'
        ],
        'Delhi': [
            'Dwarka', 'Rohini', 'Saket', 'Vasant Kunj', 'Greater Kailash'
        ],
        'Pune': [
            'Hinjewadi', 'Kharadi', 'Baner', 'Wakad', 'Viman Nagar'
        ]
    };
    
    // Clear current options
    localitySelect.innerHTML = '<option value="">Select Locality</option>';
    
    // Add new options
    if (localities[city]) {
        localities[city].forEach(locality => {
            const option = document.createElement('option');
            option.value = locality;
            option.textContent = locality;
            localitySelect.appendChild(option);
        });
    }
});

// Auto-calculate suggested bathrooms based on BHK
document.getElementById('bhk').addEventListener('change', (e) => {
    const bhk = parseInt(e.target.value);
    const bathroomsInput = document.getElementById('bathrooms');
    
    // Only set if bathrooms is empty
    if (!bathroomsInput.value) {
        // Typically, bathrooms = BHK for smaller homes, slightly less for larger
        const suggestedBathrooms = Math.max(1, Math.min(bhk, 4));
        bathroomsInput.value = suggestedBathrooms;
    }
});

// Input validation feedback
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('blur', function() {
        if (this.value && (parseFloat(this.value) < parseFloat(this.min) || 
            parseFloat(this.value) > parseFloat(this.max))) {
            this.classList.add('is-invalid');
        } else {
            this.classList.remove('is-invalid');
        }
    });
    
    input.addEventListener('input', function() {
        this.classList.remove('is-invalid');
    });
});

// Initialize
console.log('Indian House Price Prediction App Loaded');
console.log('Ready to predict house prices!');
