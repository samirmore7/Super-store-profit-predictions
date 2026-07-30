import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the pickle model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'GBML.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model GBML.pkl: {e}")

# HTML Template with UI, CSS Themes, Currency Switching, and Analytics Charts
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GBML Analytics & Prediction Dashboard</title>
    <!-- Chart.js for Analytics -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root[data-theme="dark"] {
            --bg-color: #0b0f19;
            --panel-bg: rgba(22, 31, 49, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --accent-color: #10b981;
            --accent-hover: #059669;
            --text-main: #f3f4f6;
            --text-sub: #9ca3af;
            --input-bg: #1f2937;
            --card-glow: 0 8px 32px 0 rgba(16, 185, 129, 0.15);
        }

        :root[data-theme="cyberpunk"] {
            --bg-color: #0d0221;
            --panel-bg: rgba(26, 14, 56, 0.8);
            --border-color: #ff007f;
            --accent-color: #00f0ff;
            --accent-hover: #ff007f;
            --text-main: #ffffff;
            --text-sub: #d0a3ff;
            --input-bg: #190938;
            --card-glow: 0 0 20px rgba(0, 240, 255, 0.4);
        }

        :root[data-theme="slate"] {
            --bg-color: #0f172a;
            --panel-bg: rgba(30, 41, 59, 0.8);
            --border-color: rgba(255, 255, 255, 0.15);
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: #334155;
            --card-glow: 0 8px 32px 0 rgba(56, 189, 248, 0.15);
        }

        :root[data-theme="light"] {
            --bg-color: #f1f5f9;
            --panel-bg: #ffffff;
            --border-color: #cbd5e1;
            --accent-color: #2563eb;
            --accent-hover: #1d4ed8;
            --text-main: #0f172a;
            --text-sub: #64748b;
            --input-bg: #f8fafc;
            --card-glow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .theme-selector select {
            background: var(--input-bg);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            outline: none;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: var(--card-glow);
        }

        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--accent-color);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .form-group label {
            font-size: 0.8rem;
            color: var(--text-sub);
            font-weight: 600;
            text-transform: uppercase;
        }

        .form-group input {
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.6rem;
            border-radius: 8px;
            outline: none;
        }

        .form-group input:focus {
            border-color: var(--accent-color);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--accent-color);
            color: #000;
            font-weight: 700;
            padding: 0.8rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: filter 0.2s;
        }

        .btn-submit:hover {
            filter: brightness(1.15);
        }

        .currency-switch {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .currency-btn {
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }

        .currency-btn.active {
            background: var(--accent-color);
            color: #000;
        }

        .prediction-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 1rem 0;
            color: var(--text-main);
        }

        .categorical-summary {
            margin-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            padding-top: 1rem;
        }

        .category-tag {
            display: inline-block;
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.85rem;
            margin: 0.2rem;
        }

        .chart-container {
            position: relative;
            height: 250px;
            width: 100%;
            margin-top: 1rem;
        }
    </style>
</head>
<body>

    <div class="header">
        <h1>Gradient Boosting Analytics Studio</h1>
        <div class="theme-selector">
            <label for="theme">Theme: </label>
            <select id="theme" onchange="changeTheme(this.value)">
                <option value="dark">Dark Emerald</option>
                <option value="cyberpunk">Cyberpunk Neon</option>
                <option value="slate">Slate Blue</option>
                <option value="light">Light Elegant</option>
            </select>
        </div>
    </div>

    <div class="dashboard-grid">
        <!-- Input Form Section -->
        <div class="card">
            <h2>Model Input Parameters</h2>
            <form id="predictForm" class="form-grid">
                <div class="form-group">
                    <label>Ship Mode</label>
                    <input type="number" name="Ship Mode" value="1" required>
                </div>
                <div class="form-group">
                    <label>Customer Name ID</label>
                    <input type="number" name="Customer Name" value="12" required>
                </div>
                <div class="form-group">
                    <label>Segment</label>
                    <input type="number" name="Segment" value="0" required>
                </div>
                <div class="form-group">
                    <label>Country</label>
                    <input type="number" name="Country" value="0" required>
                </div>
                <div class="form-group">
                    <label>City</label>
                    <input type="number" name="City" value="10" required>
                </div>
                <div class="form-group">
                    <label>State</label>
                    <input type="number" name="State" value="3" required>
                </div>
                <div class="form-group">
                    <label>Region</label>
                    <input type="number" name="Region" value="2" required>
                </div>
                <div class="form-group">
                    <label>Category</label>
                    <input type="number" name="Category" value="1" required>
                </div>
                <div class="form-group">
                    <label>Sub-Category</label>
                    <input type="number" name="Sub-Category" value="4" required>
                </div>
                <div class="form-group">
                    <label>Product Name ID</label>
                    <input type="number" name="Product Name" value="50" required>
                </div>
                <div class="form-group">
                    <label>Sales</label>
                    <input type="number" step="0.01" name="Sales" value="261.96" required>
                </div>
                <div class="form-group">
                    <label>Quantity</label>
                    <input type="number" name="Quantity" value="2" required>
                </div>
                <div class="form-group">
                    <label>Discount</label>
                    <input type="number" step="0.01" name="Discount" value="0.00" required>
                </div>

                <button type="submit" class="btn-submit">Compute Prediction</button>
            </form>
        </div>

        <!-- Output and Analytics Section -->
        <div>
            <!-- Result Panel -->
            <div class="card" style="margin-bottom: 2rem;">
                <h2>Predicted Output</h2>
                <div class="prediction-value" id="predVal">--</div>
                
                <p style="color: var(--text-sub); font-size: 0.9rem;">Currency Options:</p>
                <div class="currency-switch">
                    <button class="currency-btn active" onclick="setCurrency('USD', 1.0, '$')">USD ($)</button>
                    <button class="currency-btn" onclick="setCurrency('EUR', 0.92, '€')">EUR (€)</button>
                    <button class="currency-btn" onclick="setCurrency('GBP', 0.79, '£')">GBP (£)</button>
                    <button class="currency-btn" onclick="setCurrency('INR', 83.5, '₹')">INR (₹)</button>
                </div>

                <div class="categorical-summary">
                    <p style="color: var(--text-sub); font-size: 0.85rem; margin-bottom: 0.5rem;">Categorical Breakdown Overview</p>
                    <span class="category-tag" id="cat-segment">Segment: --</span>
                    <span class="category-tag" id="cat-region">Region: --</span>
                    <span class="category-tag" id="cat-category">Category: --</span>
                </div>
            </div>

            <!-- Dashboard Analytics -->
            <div class="card">
                <h2>Feature Impact & Distribution</h2>
                <div class="chart-container">
                    <canvas id="analyticsChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        let baseValueUSD = 0;
        let currentRate = 1.0;
        let currentSymbol = '$';
        let chartInstance = null;

        function changeTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
        }

        function setCurrency(code, rate, symbol) {
            currentRate = rate;
            currentSymbol = symbol;
            
            document.querySelectorAll('.currency-btn').forEach(btn => {
                btn.classList.remove('active');
                if(btn.innerText.includes(code)) btn.classList.add('active');
            });
            
            updateDisplay();
        }

        function updateDisplay() {
            if (baseValueUSD !== 0) {
                let converted = (baseValueUSD * currentRate).toFixed(2);
                document.getElementById('predVal').innerText = `${currentSymbol}${converted}`;
            }
        }

        function initChart(sales, quantity, discount) {
            const ctx = document.getElementById('analyticsChart').getContext('2d');
            if(chartInstance) chartInstance.destroy();

            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Sales (Normalized)', 'Quantity x 10', 'Discount %'],
                    datasets: [{
                        label: 'Numerical Input Distribution',
                        data: [sales / 10, quantity * 10, discount * 100],
                        backgroundColor: ['#10b981', '#38bdf8', '#ff007f']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (result.prediction !== undefined) {
                baseValueUSD = result.prediction;
                updateDisplay();

                // Categorical display mapping
                document.getElementById('cat-segment').innerText = `Segment Code: ${data['Segment']}`;
                document.getElementById('cat-region').innerText = `Region Code: ${data['Region']}`;
                document.getElementById('cat-category').innerText = `Category Code: ${data['Category']}`;

                // Update Chart
                initChart(parseFloat(data['Sales']), parseFloat(data['Quantity']), parseFloat(data['Discount']));
            } else {
                alert('Prediction Error: ' + result.error);
            }
        });

        // Initialize empty chart on load
        initChart(261.96, 2, 0);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model pickle file not loaded properly'}), 500

    try:
        data = request.get_json()
        
        # Expected model input features (13 total features extracted from pickle file)
        features_in_order = [
            'Ship Mode', 'Customer Name', 'Segment', 'Country', 'City',
            'State', 'Region', 'Category', 'Sub-Category', 'Product Name',
            'Sales', 'Quantity', 'Discount'
        ]
        
        # Format feature values into array for model prediction
        input_vector = [float(data[feature]) for feature in features_in_order]
        prediction = model.predict([input_vector])[0]

        return jsonify({'prediction': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
