// Base URL for the FastAPI backend
const API_URL = "http://127.0.0.1:8000";

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const indexBtn = document.getElementById('indexBtn');
const statsLabel = document.getElementById('statsLabel');
const imageGrid = document.getElementById('imageGrid');
const statusMessage = document.getElementById('statusMessage');

// Initial Load: Get database stats
window.addEventListener('DOMContentLoaded', fetchStats);

// Event Listeners
searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});
indexBtn.addEventListener('click', indexNewImages);

/**
 * Fetches the total number of indexed images from the backend.
 */
async function fetchStats() {
    try {
        const response = await fetch(`${API_URL}/stats`);
        if (!response.ok) throw new Error("Backend offline");
        
        const data = await response.json();
        statsLabel.textContent = `${data.total_indexed} images in database`;
        
        if(data.total_indexed === 0) {
            statusMessage.textContent = "Your database is empty. Put some images in the /images folder and click 'Index New Images'.";
        }
    } catch (error) {
        console.error("Error fetching stats:", error);
        statsLabel.textContent = "Backend offline or unreachable";
        statsLabel.style.color = "red";
    }
}

/**
 * Sends a search query to the backend and renders the results.
 */
async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    // Set UI to loading state
    searchBtn.disabled = true;
    searchBtn.textContent = "Searching...";
    imageGrid.innerHTML = "";
    statusMessage.textContent = "Generating query embeddings and searching FAISS index...";

    try {
        const response = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, top_k: 20 })
        });

        if (!response.ok) throw new Error("Search request failed");

        const data = await response.json();
        renderResults(data.results);

    } catch (error) {
        console.error("Search error:", error);
        statusMessage.textContent = "Error executing search. Check backend console.";
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = "Search";
    }
}

/**
 * Converts backend JSON results into HTML Image Cards.
 */
function renderResults(results) {
    statusMessage.textContent = "";

    if (results.length === 0) {
        statusMessage.textContent = "No visually similar images found.";
        return;
    }

    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'image-card';

        const img = document.createElement('img');
        // Tell the browser to load the image through our FastAPI GET endpoint
        img.src = `${API_URL}/image?filename=${encodeURIComponent(result.filename)}`;
        img.loading = "lazy"; // Prevents loading off-screen images immediately

        const overlay = document.createElement('div');
        overlay.className = 'overlay';
        
        // Convert cosine similarity score (typically 0.1 to 0.4 for CLIP text-to-image) to a display metric
        const matchScore = (result.similarity_score * 100).toFixed(1);

        overlay.innerHTML = `
            <div class="overlay-filename">${result.filename}</div>
            <div class="overlay-score">Similarity: ${matchScore}%</div>
        `;

        card.appendChild(img);
        card.appendChild(overlay);
        imageGrid.appendChild(card);
    });
}

/**
 * Triggers the backend pipeline to scan the images/ directory and generate embeddings.
 */
async function indexNewImages() {
    indexBtn.disabled = true;
    indexBtn.textContent = "Indexing... This may take a while.";
    
    try {
        const response = await fetch(`${API_URL}/index`, { method: 'POST' });
        const data = await response.json();
        
        alert(data.message);
        fetchStats(); // Update the count in the UI
        
    } catch (error) {
        console.error("Indexing error:", error);
        alert("Failed to index images. Make sure backend is running.");
    } finally {
        indexBtn.disabled = false;
        indexBtn.textContent = "Index New Images";
    }
}
