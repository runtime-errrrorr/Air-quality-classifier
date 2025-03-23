document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Show loading spinner
    document.getElementById('loading').classList.remove('d-none');
    document.getElementById('results').classList.add('d-none');
    
    // Collect form data
    const formData = new FormData(this);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Update results
        const predictionLabel = document.getElementById('predictionLabel');
        const predictionMessage = document.getElementById('predictionMessage');
        
        predictionLabel.className = `alert alert-${result.color}`;
        predictionLabel.textContent = result.label;
        predictionMessage.textContent = result.message;
        
        // Show results
        document.getElementById('results').classList.remove('d-none');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        // Hide loading spinner
        document.getElementById('loading').classList.add('d-none');
    }
}); 