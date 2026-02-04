async function checkSong() {
    const lyricsInput = document.getElementById('lyricsInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');
    const resultsList = document.getElementById('resultsList');

    if (!lyricsInput.value.trim()) {
        alert("Please paste some lyrics first!");
        return;
    }


    loader.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lyrics: lyricsInput.value })
        });

        const data = await response.json();

        resultsList.innerHTML = '';
        data.matches.forEach(match => {
            const percentage = (match.confidence * 100).toFixed(1);
            resultsList.innerHTML += `
                <div class="result-item">
                    <p>${match.label}</p>
                    <small>Confidence: ${percentage}%</small>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        });

        resultsSection.classList.remove('hidden');
    } catch (error) {
        console.error("Error connecting to Python API:", error);
        alert("Make sure your Python FastAPI server is running on localhost:8000");
    } finally {
        loader.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}