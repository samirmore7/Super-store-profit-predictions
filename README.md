# Super-store-profit-predictions
https://super-store-profit-predictions.onrender.com

# 🚀 Superstore Profit Prediction - GBML Analytics Studio

A high-performance machine learning web application built with **Flask**, **Scikit-Learn**, **Tailwind CSS**, and **Chart.js**, designed to deploy and serve a Gradient Boosting Model (`GBML.pkl`). 

This application predicts product profit margins based on retail parameter vectors (e.g., Sales, Segment, Region, Discount) and provides an interactive, real-time executive dashboard.

---

## ✨ Features

- **Gradient Boosting Machine Learning Model**: Uses `GBML.pkl` (with built-in fallback simulation for testing environments).
- **Vercel Serverless Ready**: Uses dynamic absolute path resolution to prevent file-loading issues in serverless contexts.
- **Dynamic Multi-Currency View**: Instant currency conversion across **USD ($)**, **EUR (€)**, **GBP (£)**, and **INR (₹)**.
- **Categorical Mapping & Tiering**: Classifies predicted profit into market tiers (*Loss / Deficit*, *Low Margin*, *Healthy Margin*, *High Value Profit*).
- **Interactive Visual Analytics**:
  - **Profit Impact Bar Chart**: Displays Sales vs. Discount Impact vs. Net Profit.
  - **Feature Profile Radar Chart**: Visualizes input feature distribution.
- **Multiple Visual Themes**: Toggle live between *Dark Emerald*, *Midnight Gold*, *Cyberpunk*, and *Obsidian*.
- **Animated Micro-Interactions**: Smooth button elevation, pulsing badge indicators, and interactive UI feedback.

---

## 📂 Project Structure

```text
.
├── app.py              # Main Flask application containing embedded dashboard UI & ML backend
├── GBML.pkl            # Trained Gradient Boosting Model file
├── requirements.txt    # Python dependencies
├── vercel.json         # Vercel serverless build routing configuration
└── README.md           # Project documentation
🛠️ Local Installation & Setup
Prerequisites
Python 3.9+ installed on your machine.

1. Clone the Repository
Bash
git clone [https://github.com/your-username/superstore-profit-predictions.git](https://github.com/your-username/superstore-profit-predictions.git)
cd superstore-profit-predictions
2. Create a Virtual Environment
Bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Application
Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000.

☁️ Deploying to Vercel
Install Vercel CLI (Optional):

Bash
npm i -g vercel
Push Code to GitHub:
Ensure app.py, GBML.pkl, requirements.txt, and vercel.json are committed to your GitHub repository.

Deploy via Vercel Dashboard:

Import your GitHub repository into Vercel.

Leave the build settings as default (Vercel automatically detects vercel.json).

Click Deploy.

📊 API Endpoint Specification
POST /predict
Executes inference on the Gradient Boosting model.

Request Payload
JSON
{
  "features": [1, 12, 0, 0, 10, 3, 2, 1, 4, 50, 261.96, 2, 0.00]
}
Response Format
JSON
{
  "status": "success",
  "prediction": 65.49,
  "tier": {
    "category": "Healthy Margin",
    "badge_class": "badge-healthy",
    "description": "Standard optimal yield product performance."
  },
  "decoded": {
    "segment": "Consumer",
    "region": "South",
    "category": "Office Supplies"
  }
}
