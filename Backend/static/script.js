document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission

        // Collect form data
        const formData = new FormData(form);

        // Send the form data to the server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json()) // Parse the JSON response
        .then(data => {
            // Display the prediction or error message
            const predictionDiv = document.getElementById('prediction');
            if (data.error) {
                predictionDiv.innerHTML = `<p>Error: ${data.error}</p>`;
            } else if (data.disease) {
                predictionDiv.innerHTML = `<p>Predicted Disease: ${data.disease}</p>`;
            } else {
                predictionDiv.innerHTML = `<p>No prediction available.</p>`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            const predictionDiv = document.getElementById('prediction');
            predictionDiv.innerHTML = `<p>An error occurred: ${error}</p>`;
        });
    });
});
