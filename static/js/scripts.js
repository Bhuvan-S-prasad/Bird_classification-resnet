const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadButton = document.getElementById("upload-btn");
const resultContainer = document.getElementById("prediction-result");
const uploadedImage = document.getElementById("uploaded-image");

// Handle drag-over event
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("hover");
});

// Handle drag-leave event
dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("hover");
});

// Handle drop event
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("hover");
    const files = e.dataTransfer.files;
    if (files.length) {
        fileInput.files = files; // Assign dropped files to the input
        showPreview(files[0]); // Show preview for the dropped file
    }
});

dropZone.addEventListener("click", () => {
    fileInput.click(); // Open file browser dialog
});


// Function to show image preview
function showPreview(file) {
    const reader = new FileReader();

    reader.onload = function (e) {
        uploadedImage.src = e.target.result;
        uploadedImage.hidden = false;
    };

    reader.readAsDataURL(file);
}

// Handle upload button click
uploadButton.addEventListener("click", () => {
    if (fileInput.files.length === 0) {
        alert("Please select a file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Clear previous results
    resultContainer.innerHTML = "";
    uploadedImage.hidden = true;

    // Add loading text
    resultContainer.innerHTML = "<p>Processing...</p>";

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                resultContainer.innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                const birdInfo = data.details || {
                    name: data.class,
                    description: "Details not available for this bird.",
                    scientificName: "Unknown",
                    habitat: "Unknown",
                    diet: "Unknown",
                    lifespan: "Unknown",
                };

                // Populate the result container with bird data
                resultContainer.innerHTML = `
                    <div class="result-card">
                        <div class="result-image">
                            <img src="${data.file_path}" alt="${birdInfo.name}">
                        </div>
                        <div class="result-details">
                            <h2>${birdInfo.name}</h2>
                            <p><em>${birdInfo.scientificName || "Unknown"}</em></p>
                            <ul class="result-info">
                                <li><span>📍 Habitat:</span> ${birdInfo.habitat || "Unknown"}</li>
                                <li><span>🍴 Diet:</span> ${birdInfo.diet || "Unknown"}</li>
                                <li><span>⏳ Lifespan:</span> ${birdInfo.lifespan || "Unknown"}</li>
                            </ul>
                            <p class="description"><strong>Description:</strong> ${birdInfo.description}</p>
                        </div>
                    </div>
                `;
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            resultContainer.innerHTML = `<p class="error">An error occurred: ${error.message}</p>`;
        });
});
