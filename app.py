import os
import pickle
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- VERCEL ABSOLUTE PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "GBML.pkl")

model = None

# Attempt to load model with fallback error handling
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[SUCCESS] Model loaded with joblib from: {MODEL_PATH}")
    except Exception:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print(f"[SUCCESS] Model loaded with pickle from: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed unpickling model: {e}")

# Mock predictor fallback if model load fails or during testing
class MockGBMLModel:
    def predict(self, X):
        X_arr = np.array(X)
        sales = X_arr[0][10] if len(X_arr[0]) > 10 else 100
        discount = X_arr[0][12] if len(X_arr[0]) > 12 else 0
        estimated_profit = (sales * 0.28) - (sales * discount * 1.4)
        return np.array([estimated_profit])

if model is None:
    print("[WARNING] Running with mock predictor fallback.")
    model = MockGBMLModel()

# Category Labels Mapping based on Superstore numerical encodings
CATEGORIES = {0: "Furniture", 1: "Office Supplies", 2: "Technology"}
REGIONS = {0: "Central", 1: "East", 2: "South", 3: "West"}
SEGMENTS = {0: "Consumer", 1: "Corporate", 2: "Home Office"}

def evaluate_profit_tier(profit_val):
    if profit_val < 0:
        return {"category": "Loss / Deficit", "badge_class": "badge-loss", "description": "Negative margin unit. Discount strategy needs optimization."}
    elif profit_val < 20:
        return {"category": "Low Margin", "badge_class": "badge-low", "description": "Low profitability tier. High volume required."}
    elif profit_val < 100:
        return {"category": "Healthy Margin", "badge_class": "badge-healthy", "description": "Standard optimal yield product performance."}
    else:
        return {"category": "High Value Profit", "badge_class": "badge-high", "description": "Top-tier revenue driver with max profitability."}


# --- EMBEDDED DASHBOARD & UI TEMPLATE WITH ANIMATIONS ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="emerald">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Boosting Analytics Studio</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
        :root[data-theme="emerald"] {
            --bg-primary: #06140e;
            --bg-card: #0a2318;
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.35);
            --text-main: #ecfdf5;
            --border: #134e4a;
        }
        :root[data-theme="midnight"] {
            --bg-primary: #0b0f19;
            --bg-card: #111827;
            --accent: #f59e0b;
            --accent-glow: rgba(245, 158, 11, 0.35);
            --text-main: #f3f4f6;
            --border: #1f2937;
        }
        :root[data-theme="cyberpunk"] {
            --bg-primary: #05050d;
            --bg-card: #0d0f23;
            --accent: #00f0ff;
            --accent-glow: rgba(0, 240, 255, 0.4);
            --text-main: #ffffff;
            --border: #7000ff;
        }
        :root[data-theme="obsidian"] {
            --bg-primary: #121212;
            --bg-card: #1e1e1e;
            --accent: #8b5cf6;
            --accent-glow: rgba(139, 92, 246, 0.35);
            --text-main: #ffffff;
            --border: #2d2d2d;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color 0.4s ease, color 0.4s ease;
        }

        .card-panel {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px -10px var(--accent-glow);
            transition: all 0.3s ease;
        }

        /* --- ADVANCED BUTTON & INTERACTIVE ANIMATIONS --- */
        .btn-animated {
            position: relative;
            background: linear-gradient(135deg, var(--accent) 0%, rgba(255,255,255,0.1) 100%);
            background-size: 200% 200%;
            color: #000;
            font-weight: 700;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            box-shadow: 0 4px 15px var(--accent-glow);
        }

        .btn-animated:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 25px var(--accent-glow);
            background-position: right center;
        }

        .btn-animated:active {
            transform: translateY(1px) scale(0.98);
        }

        /* Currency Selector Pill Animations */
        .curr-btn {
            position: relative;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
        }

        .curr-btn::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.4s ease, height 0.4s ease;
        }

        .curr-btn:active::after {
            width: 200px;
            height: 200px;
            opacity: 0;
        }

        .curr-active {
            background-color: var(--accent) !important;
            color: #000 !important;
            font-weight: 800;
            box-shadow: 0 0 15px var(--accent-glow);
            transform: scale(1.05);
        }

        /* Pulsing Glow Badges */
        .badge-loss { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
        .badge-low { background: #78350f; color: #fcd34d; border: 1px solid #f59e0b; }
        .badge-healthy { background: #065f46; color: #6ee7b7; border: 1px solid #10b981; }
        .badge-high { background: #3b0764; color: #d8b4fe; border: 1px solid #a855f7; }

        .pulse-badge {
            animation: pulse-glow 2s infinite;
        }

        @keyframes pulse-glow {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.85; transform: scale(1.03); }
        }

        /* Custom Input Styling */
        input-field {
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        input-field:focus {
            box-shadow: 0 0 10px var(--accent-glow);
        }
    </style>
</head>
<body class="min-h-screen pb-12">

    <!-- Header / Top Bar -->
    <header class="border-b border-gray-800 bg-opacity-60 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
        <div class="flex items-center space-x-3">
            <div class="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                <i data-lucide="brain-circuit" class="w-6 h-6 animate-pulse"></i>
            </div>
            <div>
                <h1 class="text-xl font-extrabold tracking-wide">Gradient Boosting Analytics Studio</h1>
                <p class="text-xs text-gray-400">Production Model Deployment & Feature Analysis</p>
            </div>
        </div>

        <div class="flex items-center space-x-2 bg-gray-900/90 p-1.5 rounded-xl border border-gray-800">
            <i data-lucide="palette" class="w-4 h-4 text-gray-400 ml-2"></i>
            <span class="text-xs text-gray-400 font-medium">Theme:</span>
            <select onchange="setTheme(this.value)" class="bg-gray-800 text-xs text-emerald-400 font-bold px-3 py-1 rounded-lg focus:outline-none cursor-pointer border border-gray-700 hover:border-emerald-500 transition-all">
                <option value="emerald">Dark Emerald</option>
                <option value="midnight">Midnight Gold</option>
                <option value="cyberpunk">Cyberpunk</option>
                <option value="obsidian">Obsidian</option>
            </select>
        </div>
    </header>

    <!-- Main Grid -->
    <main class="max-w-7xl mx-auto px-6 pt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Input Form (Left Col) -->
        <section class="lg:col-span-5 card-panel rounded-2xl p-6">
            <h2 class="text-lg font-bold text-emerald-400 mb-6 flex items-center gap-2">
                <i data-lucide="sliders" class="w-5 h-5"></i> Model Input Parameters
            </h2>

            <form id="predictionForm" onsubmit="runPrediction(event)" class="grid grid-cols-2 gap-4">
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Ship Mode</label>
                    <input type="number" name="f0" value="1" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Customer ID</label>
                    <input type="number" name="f1" value="12" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Segment</label>
                    <input type="number" name="f2" value="0" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Country</label>
                    <input type="number" name="f3" value="0" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">City</label>
                    <input type="number" name="f4" value="10" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">State</label>
                    <input type="number" name="f5" value="3" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Region</label>
                    <input type="number" name="f6" value="2" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Category</label>
                    <input type="number" name="f7" value="1" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Sub-Category</label>
                    <input type="number" name="f8" value="4" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Product ID</label>
                    <input type="number" name="f9" value="50" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Sales ($)</label>
                    <input type="number" step="any" name="f10" value="261.96" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div>
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Quantity</label>
                    <input type="number" name="f11" value="2" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>
                <div class="col-span-2">
                    <label class="block text-xs font-semibold text-gray-400 uppercase mb-1">Discount Ratio (0.00 - 1.00)</label>
                    <input type="number" step="any" name="f12" value="0.00" required class="w-full bg-gray-900/90 border border-gray-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-all">
                </div>

                <button type="submit" id="submitBtn" class="col-span-2 btn-animated py-3.5 rounded-xl mt-3 flex justify-center items-center space-x-2 text-sm tracking-wide">
                    <i data-lucide="zap" class="w-4 h-4 fill-current"></i>
                    <span>COMPUTE PREDICTION</span>
                </button>
            </form>
        </section>

        <!-- Output Analytics & Charts (Right Col) -->
        <section class="lg:col-span-7 space-y-6">
            
            <!-- Predicted Valuation Box -->
            <div class="card-panel rounded-2xl p-6 relative overflow-hidden">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <span class="text-xs uppercase tracking-wider text-gray-400 font-extrabold flex items-center gap-1.5">
                            <i data-lucide="trending-up" class="w-4 h-4 text-emerald-400"></i> Model Predicted Profit Output
                        </span>
                        <div class="text-5xl font-extrabold mt-2 tracking-tight transition-all" id="predictedVal">$0.00</div>
                    </div>
                    <span id="categoryBadge" class="px-3.5 py-1.5 rounded-full text-xs font-bold uppercase tracking-wide pulse-badge badge-healthy">
                        Pending Evaluation
                    </span>
                </div>

                <p id="categoryDescription" class="text-sm text-gray-400 mb-6">Enter features on the left panel to execute inference.</p>

                <!-- Dynamic Multi-Currency Buttons -->
                <div class="border-t border-gray-800/80 pt-4">
                    <label class="block text-xs text-gray-400 font-medium mb-2.5">Select View Currency:</label>
                    <div class="grid grid-cols-4 gap-2.5">
                        <button onclick="setCurrency('USD')" id="btn-USD" class="curr-btn curr-active py-2 px-3 text-xs rounded-xl bg-gray-800 border border-gray-700">USD ($)</button>
                        <button onclick="setCurrency('EUR')" id="btn-EUR" class="curr-btn py-2 px-3 text-xs rounded-xl bg-gray-800 border border-gray-700 text-gray-400">EUR (€)</button>
                        <button onclick="setCurrency('GBP')" id="btn-GBP" class="curr-btn py-2 px-3 text-xs rounded-xl bg-gray-800 border border-gray-700 text-gray-400">GBP (£)</button>
                        <button onclick="setCurrency('INR')" id="btn-INR" class="curr-btn py-2 px-3 text-xs rounded-xl bg-gray-800 border border-gray-700 text-gray-400">INR (₹)</button>
                    </div>
                </div>

                <!-- Categorical Context Badges -->
                <div class="border-t border-gray-800/80 pt-4 mt-4">
                    <label class="block text-xs text-gray-400 font-medium mb-2">Decoded Categorical Mapping:</label>
                    <div class="flex flex-wrap gap-2 text-xs">
                        <span class="px-3 py-1 bg-gray-900 border border-gray-800 rounded-lg text-gray-300 font-semibold" id="catSegment">Segment: --</span>
                        <span class="px-3 py-1 bg-gray-900 border border-gray-800 rounded-lg text-gray-300 font-semibold" id="catRegion">Region: --</span>
                        <span class="px-3 py-1 bg-gray-900 border border-gray-800 rounded-lg text-gray-300 font-semibold" id="catCategory">Category: --</span>
                    </div>
                </div>
            </div>

            <!-- Charts Row -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="card-panel rounded-2xl p-4">
                    <h3 class="text-xs font-bold uppercase tracking-wider text-emerald-400 mb-3 flex items-center gap-2">
                        <i data-lucide="bar-chart-2" class="w-4 h-4"></i> Profit Impact Drivers
                    </h3>
                    <div class="h-44">
                        <canvas id="impactChart"></canvas>
                    </div>
                </div>

                <div class="card-panel rounded-2xl p-4">
                    <h3 class="text-xs font-bold uppercase tracking-wider text-emerald-400 mb-3 flex items-center gap-2">
                        <i data-lucide="pie-chart" class="w-4 h-4"></i> Feature Scale Profile
                    </h3>
                    <div class="h-44">
                        <canvas id="radarChart"></canvas>
                    </div>
                </div>
            </div>

        </section>

    </main>

    <script>
        lucide.createIcons();
        let baseUsdVal = 0;
        let selectedCurrency = 'USD';
        const rates = { USD: 1.0, EUR: 0.92, GBP: 0.79, INR: 83.2 };
        const symbols = { USD: '$', EUR: '€', GBP: '£', INR: '₹' };
        
        let barChartInstance, radarChartInstance;

        function initCharts() {
            // Bar Chart
            const ctxBar = document.getElementById('impactChart').getContext('2d');
            barChartInstance = new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: ['Sales Val', 'Discount Impact', 'Predicted Profit'],
                    datasets: [{
                        data: [261.96, 0, 65.49],
                        backgroundColor: ['#10b981', '#ef4444', '#3b82f6'],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { grid: { color: '#1f2937' }, ticks: { color: '#9ca3af', font: { size: 10 } } },
                        x: { grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 10 } } }
                    }
                }
            });

            // Radar Chart
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            radarChartInstance = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Ship Mode', 'Segment', 'Region', 'Category', 'Quantity'],
                    datasets: [{
                        data: [1, 0, 2, 1, 2],
                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                        borderColor: '#10b981',
                        borderWidth: 2,
                        pointBackgroundColor: '#10b981'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        r: {
                            grid: { color: '#1f2937' },
                            angleLines: { color: '#1f2937' },
                            ticks: { display: false }
                        }
                    }
                }
            });
        }

        function setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
        }

        function setCurrency(curr) {
            selectedCurrency = curr;
            ['USD', 'EUR', 'GBP', 'INR'].forEach(c => {
                const btn = document.getElementById(`btn-${c}`);
                if (c === curr) {
                    btn.className = "curr-btn curr-active py-2 px-3 text-xs rounded-xl";
                } else {
                    btn.className = "curr-btn py-2 px-3 text-xs rounded-xl bg-gray-800 border border-gray-700 text-gray-400 hover:border-gray-500";
                }
            });
            updateDisplay();
        }

        function updateDisplay() {
            const converted = baseUsdVal * rates[selectedCurrency];
            const formatted = `${symbols[selectedCurrency]}${converted.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            document.getElementById('predictedVal').innerText = formatted;
        }

        async function runPrediction(event) {
            if(event) event.preventDefault();
            
            const btn = document.getElementById('submitBtn');
            btn.style.opacity = "0.7";
            btn.innerText = "COMPUTING...";

            const form = document.getElementById('predictionForm');
            const formData = new FormData(form);
            const features = [];
            for (let i = 0; i < 13; i++) {
                features.push(parseFloat(formData.get(`f${i}`)));
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: features })
                });

                const result = await response.json();
                
                btn.style.opacity = "1";
                btn.innerHTML = `<i data-lucide="zap" class="w-4 h-4 fill-current"></i><span>COMPUTE PREDICTION</span>`;
                lucide.createIcons();

                if(result.status === 'error') {
                    alert('Prediction Error: ' + result.message);
                    return;
                }

                baseUsdVal = result.prediction;
                updateDisplay();

                // Update Tier Badge & Description
                const badge = document.getElementById('categoryBadge');
                badge.innerText = result.tier.category;
                badge.className = `px-3.5 py-1.5 rounded-full text-xs font-bold uppercase tracking-wide pulse-badge ${result.tier.badge_class}`;
                document.getElementById('categoryDescription').innerText = result.tier.description;

                // Update Decoded Badges
                document.getElementById('catSegment').innerText = `Segment: ${result.decoded.segment}`;
                document.getElementById('catRegion').innerText = `Region: ${result.decoded.region}`;
                document.getElementById('catCategory').innerText = `Category: ${result.decoded.category}`;

                // Update Bar Chart
                barChartInstance.data.datasets[0].data = [features[10], (features[10] * features[12]), result.prediction];
                barChartInstance.update();

                // Update Radar Chart
                radarChartInstance.data.datasets[0].data = [features[0], features[2], features[6], features[7], features[11]];
                radarChartInstance.update();

            } catch(e) {
                btn.style.opacity = "1";
                btn.innerHTML = `<span>COMPUTE PREDICTION</span>`;
                alert('Connection error: ' + e.message);
            }
        }

        window.onload = () => {
            initCharts();
            runPrediction();
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data.get('features', [])
        
        # Reshape array for model input vector
        input_vector = np.array(features).reshape(1, -1)
        
        # Predict using loaded model
        pred = model.predict(input_vector)
        predicted_val = float(pred[0])
        
        tier = evaluate_profit_tier(predicted_val)
        
        # Decode categorical variables safely
        seg_id = int(features[2]) if len(features) > 2 else 0
        reg_id = int(features[6]) if len(features) > 6 else 0
        cat_id = int(features[7]) if len(features) > 7 else 0
        
        return jsonify({
            "status": "success",
            "prediction": predicted_val,
            "tier": tier,
            "decoded": {
                "segment": SEGMENTS.get(seg_id, f"ID {seg_id}"),
                "region": REGIONS.get(reg_id, f"ID {reg_id}"),
                "category": CATEGORIES.get(cat_id, f"ID {cat_id}")
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
