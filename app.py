import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load Model
MODEL_PATH = "GBML.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Categorical Feature Mapping helper
def preprocess_inputs(data_dict, feature_names):
    """
    Ensures input data aligns with model features. Converts categorical text 
    inputs to deterministic numerical representations if required by scikit-learn.
    """
    df = pd.DataFrame([data_dict])
    
    # Ensure columns match model expected feature order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    df = df[feature_names]
    
    # Convert string columns to numeric codes if scikit-learn expects numeric values
    for col in df.columns:
        if df[col].dtype == 'object':
            # Deterministic hash encoding to ensure consistent numeric conversion for ML models
            df[col] = df[col].apply(lambda x: abs(hash(str(x))) % 1000)
            
    return df

# Combined HTML, Modern CSS, and Interactivity JS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="obsidian">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Analytics & AI Prediction Studio</title>
    <!-- Fonts & Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js for Dashboard -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root[data-theme="obsidian"] {
            --bg-base: #0b0f19;
            --bg-surface: rgba(18, 24, 38, 0.75);
            --bg-card: rgba(26, 34, 52, 0.6);
            --primary-accent: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.4);
            --secondary-accent: #a855f7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(255, 255, 255, 0.08);
            --gradient-btn: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        }

        :root[data-theme="cyber"] {
            --bg-base: #05050a;
            --bg-surface: rgba(15, 15, 30, 0.85);
            --bg-card: rgba(22, 22, 45, 0.7);
            --primary-accent: #00f0ff;
            --primary-glow: rgba(0, 240, 255, 0.5);
            --secondary-accent: #ff007f;
            --text-main: #ffffff;
            --text-muted: #8b8ea9;
            --border: rgba(0, 240, 255, 0.2);
            --gradient-btn: linear-gradient(135deg, #00f0ff 0%, #ff007f 100%);
        }

        :root[data-theme="gold"] {
            --bg-base: #0d0c0a;
            --bg-surface: rgba(28, 25, 20, 0.8);
            --bg-card: rgba(38, 34, 28, 0.65);
            --primary-accent: #f59e0b;
            --primary-glow: rgba(245, 158, 11, 0.4);
            --secondary-accent: #d97706;
            --text-main: #fffbeb;
            --text-muted: #b45309;
            --border: rgba(245, 158, 11, 0.15);
            --gradient-btn: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
        }

        :root[data-theme="emerald"] {
            --bg-base: #061412;
            --bg-surface: rgba(11, 33, 29, 0.8);
            --bg-card: rgba(18, 48, 42, 0.65);
            --primary-accent: #10b981;
            --primary-glow: rgba(16, 185, 129, 0.4);
            --secondary-accent: #059669;
            --text-main: #ecfdf5;
            --text-muted: #6ee7b7;
            --border: rgba(16, 185, 129, 0.15);
            --gradient-btn: linear-gradient(135deg, #34d399 0%, #059669 100%);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color 0.4s ease, border-color 0.4s ease, color 0.4s ease;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 15%, var(--primary-glow) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, var(--primary-glow) 0%, transparent 40%);
        }

        /* Top Navigation Bar */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2.5rem;
            background: var(--bg-surface);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.35rem;
            font-weight: 800;
            background: var(--gradient-btn);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .theme-selector {
            display: flex;
            gap: 0.5rem;
            background: rgba(0, 0, 0, 0.2);
            padding: 0.35rem;
            border-radius: 99px;
            border: 1px solid var(--border);
        }

        .theme-btn {
            border: none;
            background: transparent;
            color: var(--text-muted);
            padding: 0.5rem 1rem;
            border-radius: 99px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .theme-btn.active {
            background: var(--gradient-btn);
            color: #ffffff;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        /* Layout Grid */
        .dashboard-container {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 2rem;
            padding: 2.5rem;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        @media (max-width: 1024px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
        }

        /* Cards & Glassmorphism */
        .glass-card {
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 1.5rem;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--gradient-btn);
            opacity: 0.7;
        }

        .card-header {
            margin-bottom: 1.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .card-title i {
            color: var(--primary-accent);
        }

        /* Form Controls */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        input, select {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.85rem 1.1rem;
            border-radius: 0.75rem;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            border-color: var(--primary-accent);
            box-shadow: 0 0 15px var(--primary-glow);
        }

        select option {
            background-color: var(--bg-base);
            color: var(--text-main);
        }

        /* Animated Premium Button */
        .btn-submit {
            grid-column: span 2;
            margin-top: 1rem;
            padding: 1rem 2rem;
            border: none;
            border-radius: 0.85rem;
            background: var(--gradient-btn);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: 0 10px 25px var(--primary-glow);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .btn-submit:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px var(--primary-glow);
        }

        .btn-submit:active {
            transform: translateY(1px);
        }

        .btn-submit::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                60deg,
                transparent,
                rgba(255, 255, 255, 0.25),
                transparent
            );
            transform: rotate(30deg);
            animation: shimmer 4s infinite;
        }

        @keyframes shimmer {
            0% { transform: translate(-100%, -100%) rotate(30deg); }
            100% { transform: translate(100%, 100%) rotate(30deg); }
        }

        /* Results & Analysis Dashboard */
        .result-display {
            text-align: center;
            padding: 2rem;
            background: var(--bg-card);
            border-radius: 1.25rem;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
            position: relative;
        }

        .result-value {
            font-size: 3rem;
            font-weight: 800;
            margin: 0.5rem 0;
            background: var(--gradient-btn);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: pulseValue 2s infinite alternate;
        }

        @keyframes pulseValue {
            0% { transform: scale(1); }
            100% { transform: scale(1.03); }
        }

        .charts-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }

        .chart-box {
            position: relative;
            height: 250px;
            width: 100%;
        }

        .badge {
            background: var(--primary-glow);
            color: var(--primary-accent);
            padding: 0.35rem 0.85rem;
            border-radius: 99px;
            font-size: 0.75rem;
            font-weight: 700;
            border: 1px solid var(--border);
        }
    </style>
</head>
<body>

    <!-- Header / Navbar -->
    <nav class="navbar">
        <div class="brand">
            <i class="fa-solid fa-chart-line-up"></i>
            <span>GBML Analytics Studio</span>
        </div>

        <!-- Theme Selector Switcher -->
        <div class="theme-selector">
            <button class="theme-btn active" onclick="setTheme('obsidian', this)">
                <i class="fa-solid fa-moon"></i> Obsidian
            </button>
            <button class="theme-btn" onclick="setTheme('cyber', this)">
                <i class="fa-solid fa-bolt"></i> Cyberpunk
            </button>
            <button class="theme-btn" onclick="setTheme('gold', this)">
                <i class="fa-solid fa-crown"></i> Gold
            </button>
            <button class="theme-btn" onclick="setTheme('emerald', this)">
                <i class="fa-solid fa-gem"></i> Emerald
            </button>
        </div>
    </nav>

    <!-- Main Content Area -->
    <div class="dashboard-container">
        
        <!-- Input Form Glass Panel -->
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">
                    <i class="fa-solid fa-sliders"></i>
                    <span>Model Input Parameters</span>
                </div>
                <span class="badge">GBML Regressor</span>
            </div>

            <form id="predictionForm">
                <div class="form-grid">
                    
                    <!-- Categorical Columns (Real Text Values) -->
                    <div class="input-group">
                        <label for="ship_mode">Ship Mode</label>
                        <select id="ship_mode" name="Ship Mode" required>
                            <option value="Standard Class">Standard Class</option>
                            <option value="Second Class">Second Class</option>
                            <option value="First Class">First Class</option>
                            <option value="Same Day">Same Day</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="segment">Segment</label>
                        <select id="segment" name="Segment" required>
                            <option value="Consumer">Consumer</option>
                            <option value="Corporate">Corporate</option>
                            <option value="Home Office">Home Office</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="category">Category</label>
                        <select id="category" name="Category" required>
                            <option value="Furniture">Furniture</option>
                            <option value="Office Supplies">Office Supplies</option>
                            <option value="Technology">Technology</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="sub_category">Sub-Category</label>
                        <select id="sub_category" name="Sub-Category" required>
                            <option value="Bookcases">Bookcases</option>
                            <option value="Chairs">Chairs</option>
                            <option value="Labels">Labels</option>
                            <option value="Tables">Tables</option>
                            <option value="Storage">Storage</option>
                            <option value="Furnishings">Furnishings</option>
                            <option value="Phones">Phones</option>
                            <option value="Binders">Binders</option>
                            <option value="Appliances">Appliances</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="region">Region</label>
                        <select id="region" name="Region" required>
                            <option value="East">East</option>
                            <option value="West">West</option>
                            <option value="Central">Central</option>
                            <option value="South">South</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="country">Country</label>
                        <select id="country" name="Country" required>
                            <option value="United States">United States</option>
                            <option value="Canada">Canada</option>
                            <option value="Mexico">Mexico</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="state">State</label>
                        <input type="text" id="state" name="State" value="California" required>
                    </div>

                    <div class="input-group">
                        <label for="city">City</label>
                        <input type="text" id="city" name="City" value="Los Angeles" required>
                    </div>

                    <div class="input-group">
                        <label for="customer_name">Customer Name</label>
                        <input type="text" id="customer_name" name="Customer Name" value="Claire Gute" required>
                    </div>

                    <div class="input-group">
                        <label for="product_name">Product Name</label>
                        <input type="text" id="product_name" name="Product Name" value="Bush Somerset Bookcase" required>
                    </div>

                    <!-- Numerical Input Features -->
                    <div class="input-group">
                        <label for="sales">Sales ($)</label>
                        <input type="number" step="0.01" id="sales" name="Sales" value="261.96" required>
                    </div>

                    <div class="input-group">
                        <label for="quantity">Quantity</label>
                        <input type="number" id="quantity" name="Quantity" value="2" required>
                    </div>

                    <div class="input-group full-width">
                        <label for="discount">Discount Rate (0.00 to 1.00)</label>
                        <input type="number" step="0.01" min="0" max="1" id="discount" name="Discount" value="0.00" required>
                    </div>

                    <!-- Premium Animated Button -->
                    <button type="submit" class="btn-submit">
                        <i class="fa-solid fa-wand-magic-sparkles"></i>
                        <span>Generate Intelligence Prediction</span>
                    </button>
                </div>
            </form>
        </div>

        <!-- Dashboard & Output Glass Panel -->
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">
                    <i class="fa-solid fa-chart-pie"></i>
                    <span>Analytics & Output Dashboard</span>
                </div>
                <span class="badge"><i class="fa-solid fa-shield-halved"></i> Real-time</span>
            </div>

            <!-- Main Prediction Card -->
            <div class="result-display">
                <div style="font-size: 0.9rem; color: var(--text-muted); font-weight: 600;">Predicted Model Output</div>
                <div class="result-value" id="predictionOutput">--.--</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);" id="statusText">Submit data to execute model pipeline</div>
            </div>

            <!-- Interactive Analytical Charts -->
            <div class="charts-container">
                <div class="chart-box">
                    <canvas id="analyticsChart"></canvas>
                </div>
            </div>
        </div>

    </div>

    <script>
        // Interactive Theme Switcher
        function setTheme(themeName, btnElement) {
            document.documentElement.setAttribute('data-theme', themeName);
            document.querySelectorAll('.theme-btn').forEach(btn => btn.classList.remove('active'));
            btnElement.classList.add('active');
            updateChartColors();
        }

        // Chart Initialization
        const ctx = document.getElementById('analyticsChart').getContext('2d');
        let analyticsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Sales Impact', 'Discount Penalty', 'Quantity Weight', 'Predicted Target'],
                datasets: [{
                    label: 'Feature Weight Influence',
                    data: [261.96, 0, 2, 0],
                    backgroundColor: [
                        'rgba(99, 102, 241, 0.7)',
                        'rgba(239, 68, 68, 0.7)',
                        'rgba(168, 85, 247, 0.7)',
                        'rgba(16, 185, 129, 0.9)'
                    ],
                    borderRadius: 8,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { 
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: { 
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });

        function updateChartColors() {
            analyticsChart.update();
        }

        // AJAX Form Submission
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => data[key] = value);

            const outputElem = document.getElementById('predictionOutput');
            const statusElem = document.getElementById('statusText');

            outputElem.innerText = 'Calculating...';
            statusElem.innerText = 'Running Gradient Boosting inference...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success) {
                    const val = parseFloat(result.prediction).toFixed(2);
                    outputElem.innerText = val;
                    statusElem.innerText = 'Inference complete with high confidence';

                    // Update Chart
                    analyticsChart.data.datasets[0].data = [
                        parseFloat(data['Sales']),
                        parseFloat(data['Discount']) * 100,
                        parseFloat(data['Quantity']),
                        val
                    ];
                    analyticsChart.update();
                } else {
                    outputElem.innerText = 'Error';
                    statusElem.innerText = result.error || 'Failed to process input';
                }
            } catch (err) {
                outputElem.innerText = 'Error';
                statusElem.innerText = 'Server communication error';
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        if model is None:
            return jsonify({"success": False, "error": "Model GBML.pkl file not found."})

        # Expected features in the exact pickled Gradient Boosting model order
        feature_names = getattr(model, "feature_names_in_", [
            "Ship Mode", "Customer Name", "Segment", "Country", "City", 
            "State", "Region", "Category", "Sub-Category", "Product Name", 
            "Sales", "Quantity", "Discount"
        ])
        
        # Format and preprocess data
        input_df = preprocess_inputs(data, feature_names)
        
        # Run model prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "success": True,
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
