// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.btn-tab').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to clicked button
    event.target.closest('.btn-tab').classList.add('active');
    
    // Hide results
    document.getElementById('results').style.display = 'none';
    document.getElementById('text-result').style.display = 'none';
}

// Get sentiment color
function getSentimentColor(sentiment) {
    const colors = {
        'Strongly Positive': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'Positive': '#17BF63',
        'Neutral': '#657786',
        'Negative': '#FF6B6B',
        'Strongly Negative': '#E0245E'
    };
    return colors[sentiment] || '#657786';
}

// Get sentiment class
function getSentimentClass(sentiment) {
    return sentiment.toLowerCase().replace(' ', '-');
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// User form submission
document.getElementById('user-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('username').value.trim();
    const count = document.getElementById('user-count').value;
    
    await analyzeTwitter({
        query: username,
        query_type: 'user',
        max_results: parseInt(count)
    });
});

// Hashtag form submission
document.getElementById('hashtag-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const hashtag = document.getElementById('hashtag').value.trim();
    const count = document.getElementById('hashtag-count').value;
    
    await analyzeTwitter({
        query: hashtag,
        query_type: 'hashtag',
        max_results: parseInt(count)
    });
});

// Text form submission
document.getElementById('text-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const text = document.getElementById('custom-text').value.trim();
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('text-result').style.display = 'none';
    
    try {
        const response = await fetch('/api/sentiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        displayTextResult(result);
        
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
});

// Analyze Twitter
async function analyzeTwitter(data) {
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    document.getElementById('text-result').style.display = 'none';
    
    try {
        const response = await fetch('/api/analyze/twitter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Display text result
function displayTextResult(result) {
    const container = document.getElementById('text-result-content');
    
    let html = `
        <div class="tweet-card">
            <div class="tweet-header">
                <h4>Your Text</h4>
                <span class="tweet-sentiment" style="background: ${getSentimentColor(result.sentiment)}">
                    ${result.sentiment}
                </span>
            </div>
            <div class="tweet-text">${result.text}</div>
            <div style="margin-top: 15px;">
                <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%
            </div>
            <div class="probability-bars">
                <h4>Probability Distribution</h4>
    `;
    
    for (const [sentiment, prob] of Object.entries(result.probabilities)) {
        html += `
            <div class="prob-item">
                <div class="prob-label">
                    <span>${sentiment}</span>
                    <span>${(prob * 100).toFixed(1)}%</span>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: ${prob * 100}%; background: ${getSentimentColor(sentiment)}"></div>
                </div>
            </div>
        `;
    }
    
    html += `</div></div>`;
    
    container.innerHTML = html;
    document.getElementById('text-result').style.display = 'block';
}

// Display results
function displayResults(data) {
    // Display summary
    const summaryContainer = document.getElementById('summary-content');
    let summaryHtml = `
        <div class="summary-grid">
    `;
    
    for (const [sentiment, count] of Object.entries(data.summary)) {
        summaryHtml += `
            <div class="summary-item ${getSentimentClass(sentiment)}">
                <h4>${count}</h4>
                <p>${sentiment}</p>
            </div>
        `;
    }
    
    summaryHtml += `</div>`;
    summaryHtml += `
        <div style="text-align: center; padding: 20px; background: var(--bg-light); border-radius: 8px;">
            <h4 style="margin-bottom: 10px;">Overall Analysis</h4>
            <p><strong>Query:</strong> ${data.query} (${data.query_type})</p>
            <p><strong>Total Tweets:</strong> ${data.total_tweets}</p>
            <p><strong>Average Sentiment Score:</strong> ${data.average_sentiment_score.toFixed(2)} / 4.0</p>
        </div>
    `;
    
    summaryContainer.innerHTML = summaryHtml;
    
    // Display tweets
    const tweetsContainer = document.getElementById('tweets-list');
    let tweetsHtml = '';
    
    data.tweets.forEach(tweet => {
        tweetsHtml += `
            <div class="tweet-card">
                <div class="tweet-header">
                    <small style="color: var(--text-light);">${formatDate(tweet.created_at)}</small>
                    <span class="tweet-sentiment" style="background: ${getSentimentColor(tweet.sentiment)}">
                        ${tweet.sentiment}
                    </span>
                </div>
                <div class="tweet-text">${tweet.text}</div>
                <div class="tweet-meta">
                    <span><i class="fas fa-heart"></i> ${tweet.likes}</span>
                    <span><i class="fas fa-retweet"></i> ${tweet.retweets}</span>
                    <span><i class="fas fa-comment"></i> ${tweet.replies}</span>
                    <span><i class="fas fa-chart-bar"></i> ${(tweet.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bars">
                    <h4 style="font-size: 0.9rem; margin-top: 15px; margin-bottom: 10px;">Sentiment Probabilities</h4>
        `;
        
        for (const [sentiment, prob] of Object.entries(tweet.probabilities)) {
            tweetsHtml += `
                <div class="prob-item">
                    <div class="prob-label">
                        <span>${sentiment}</span>
                        <span>${(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: ${prob * 100}%; background: ${getSentimentColor(sentiment)}"></div>
                    </div>
                </div>
            `;
        }
        
        tweetsHtml += `
                </div>
            </div>
        `;
    });
    
    tweetsContainer.innerHTML = tweetsHtml;
    document.getElementById('results').style.display = 'block';
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}