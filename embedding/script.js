document.addEventListener('DOMContentLoaded', () => {
    const wordInput = document.getElementById('wordInput');
    const getDataBtn = document.getElementById('getDataBtn');
    const visualizationMethodSelect = document.getElementById('visualizationMethodSelect');
    const embeddingPlot = document.getElementById('embeddingPlot');
    const plotLoadingMessage = document.getElementById('plotLoadingMessage');
    const neighborsOutput = document.getElementById('neighborsOutput');
    const dataLoadingMessage = document.getElementById('dataLoadingMessage');
    const resultsContainer = document.querySelector('.results-container');

    getDataBtn.addEventListener('click', () => {
        const word = wordInput.value.trim();
        const visualizationMethod = visualizationMethodSelect.value;

        if (!word) {
            alert('Please enter a word.');
            return;
        }
        dataLoadingMessage.style.display = 'block';
        plotLoadingMessage.style.display = 'block';
        resultsContainer.style.display = 'none';
        embeddingPlot.innerHTML = '';
        neighborsOutput.innerHTML = '';
        
        fetch(`/nearest_neighbors?word=${encodeURIComponent(word)}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                dataLoadingMessage.style.display = 'none';
                resultsContainer.style.display = 'block';

                if (data.message) {
                    neighborsOutput.innerHTML = `<p>${data.message}</p>`;
                    return;
                }

                if (data.error) {
                    neighborsOutput.innerHTML = `<p class="error-message">${data.error}</p>`;
                    return;
                }
                neighborsOutput.innerHTML = `<h3>Nearest Neighbors for "${data.query_word}"</h3><ul>` +  //changed data.word to data.query_word
                    data.neighbors.map(n => `<li><strong>${n.word}</strong>: ${n.similarity.toFixed(3)}</li>`).join('') +
                    '</ul>';

                fetch(`/embeddings_data`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`Failed to fetch embeddings data: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(embeddingsData => {
                        plotLoadingMessage.style.display = 'none';

                        let plotData = [];
                        let title = '';
                        if (visualizationMethod === 'pca' && embeddingsData.pca_embeddings) {
                            plotData = embeddingsData.pca_embeddings;
                            title = 'PCA Word Embeddings';
                        } else if (visualizationMethod === 'tsne' && embeddingsData.tsne_embeddings) {
                            plotData = embeddingsData.tsne_embeddings;
                            title = 't-SNE Word Embeddings';
                        }

                        if (plotData.length > 0) {
                            const trace = {
                                x: plotData.map(d => d.x),
                                y: plotData.map(d => d.y),
                                text: plotData.map(d => d.word),
                                mode: 'markers+text',
                                type: 'scatter',
                                marker: { size: 10 },
                                textposition: 'top center'
                            };

                            const layout = {
                                title: title,
                                xaxis: { title: `${visualizationMethod.toUpperCase()} Component 1` },
                                yaxis: { title: `${visualizationMethod.toUpperCase()} Component 2` },
                            };

                            Plotly.newPlot(embeddingPlot, [trace], layout);
                        } else {
                            embeddingPlot.innerHTML = `<p>No ${visualizationMethod.toUpperCase()} embeddings available.</p>`;
                        }
                    })
                    .catch(err => {
                        plotLoadingMessage.style.display = 'none';
                        embeddingPlot.innerHTML = `<p class="error-message">Error loading embeddings: ${err.message}</p>`;
                        console.error('Embedding fetch error:', err);
                    });
            })
            .catch(error => {
                dataLoadingMessage.style.display = 'none';
                plotLoadingMessage.style.display = 'none';
                neighborsOutput.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
                console.error('Neighbor fetch error:', error);
            });
    });
});
