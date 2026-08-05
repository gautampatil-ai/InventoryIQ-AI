from flask import Flask, render_template_string, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load Pre-trained Pipeline / Model
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    model = None
    MODEL_LOADED = False
    print(f"[WARNING] Model load failed: {e}")

# Features extracted from pipeline metadata
CAT_FEATURES = ['family']
NUM_FEATURES = ['id', 'store_nbr', 'onpromotion', 'year', 'month', 'day', 'week', 'dayofweek']
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

FAMILY_OPTIONS = [
    'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 
    'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 
    'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES', 
    'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 
    'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'POULTRY', 'PREPARED FOODS', 
    'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'
]

# ---------------------------------------------------------------------------
# Professional Dashboard UI Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Sales Forecasting Engine</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #06b6d4;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .navbar {
            background-color: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        .form-label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .form-control, .form-select {
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: 8px;
            padding: 0.65rem 1rem;
        }

        .form-control:focus, .form-select:focus {
            background-color: #0f172a;
            border-color: var(--primary);
            color: var(--text-main);
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.25);
        }

        .btn-predict {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border: none;
            color: white;
            font-weight: 600;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            transition: all 0.2s ease;
        }

        .btn-predict:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .metric-badge {
            background: rgba(79, 70, 229, 0.1);
            border: 1px solid rgba(79, 70, 229, 0.3);
            color: var(--accent);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
        }

        .prediction-result {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px dashed var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }

        .prediction-value {
            font-size: 3rem;
            font-weight: 700;
            color: var(--accent);
            text-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
        }

        footer {
            margin-top: auto;
            border-top: 1px solid var(--border-color);
            padding: 1.5rem 0;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark py-3">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center gap-2" href="#">
                <i class="fa-solid fa-chart-line text-primary"></i>
                <span class="fw-bold fs-5">Sales Intelligence Engine</span>
            </a>
            <span class="metric-badge"><i class="fa-solid fa-microchip me-1"></i> XGBoost Pipeline</span>
        </div>
    </nav>

    <div class="container py-5">
        <div class="row g-4">
            <!-- Form Input Column -->
            <div class="col-lg-7">
                <div class="card p-4">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h5 class="card-title m-0 fw-bold"><i class="fa-solid fa-sliders me-2 text-primary"></i>Feature Configurations</h5>
                        <span class="text-muted small">Enter transactional metrics</span>
                    </div>

                    <form id="predictionForm" method="POST" action="/predict">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Transaction ID</label>
                                <input type="number" name="id" class="form-control" value="{{ request.form.get('id', 1001) }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Store Number</label>
                                <input type="number" name="store_nbr" class="form-control" value="{{ request.form.get('store_nbr', 1) }}" required>
                            </div>
                            <div class="col-md-12">
                                <label class="form-label">Product Family</label>
                                <select name="family" class="form-select" required>
                                    {% for fam in family_options %}
                                        <option value="{{ fam }}" {% if request.form.get('family') == fam %}selected{% endif %}>{{ fam }}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">On Promotion Count</label>
                                <input type="number" name="onpromotion" class="form-control" value="{{ request.form.get('onpromotion', 0) }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Year</label>
                                <input type="number" name="year" class="form-control" value="{{ request.form.get('year', 2026) }}" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Month</label>
                                <input type="number" min="1" max="12" name="month" class="form-control" value="{{ request.form.get('month', 8) }}" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Day</label>
                                <input type="number" min="1" max="31" name="day" class="form-control" value="{{ request.form.get('day', 15) }}" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Week of Year</label>
                                <input type="number" min="1" max="53" name="week" class="form-control" value="{{ request.form.get('week', 33) }}" required>
                            </div>
                            <div class="col-md-12">
                                <label class="form-label">Day of Week (0=Mon, 6=Sun)</label>
                                <input type="number" min="0" max="6" name="dayofweek" class="form-control" value="{{ request.form.get('dayofweek', 2) }}" required>
                            </div>
                        </div>

                        <div class="mt-4">
                            <button type="submit" class="btn btn-predict w-100">
                                <i class="fa-solid fa-wand-magic-sparkles me-2"></i>Generate Forecast
                            </button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Prediction Display Column -->
            <div class="col-lg-5">
                <div class="card p-4 h-100 d-flex flex-column justify-content-between">
                    <div>
                        <h5 class="card-title mb-4 fw-bold"><i class="fa-solid fa-chart-pie me-2 text-primary"></i>Forecast Output</h5>
                        
                        {% if not model_loaded %}
                            <div class="alert alert-warning text-dark small" role="alert">
                                <i class="fa-solid fa-triangle-exclamation me-1"></i>
                                Model file (<code>model.pkl</code>) was not detected. Save your deserialized pickled pipeline locally to activate predictions.
                            </div>
                        {% endif %}

                        <div class="prediction-result my-4">
                            <div class="text-uppercase small fw-bold text-muted mb-2">Estimated Unit Sales</div>
                            {% if prediction is not none %}
                                <div class="prediction-value">{{ "%.2f"|format(prediction) }}</div>
                                <div class="text-success small mt-2"><i class="fa-solid fa-circle-check me-1"></i>Inference Completed</div>
                            {% else %}
                                <div class="prediction-value text-muted">--</div>
                                <div class="text-muted small mt-2">Submit parameters to trigger inference</div>
                            {% endif %}
                        </div>
                    </div>

                    <div class="border-top border-secondary pt-3">
                        <div class="d-flex justify-content-between small text-muted">
                            <span>Pipeline Architecture:</span>
                            <span class="text-light">ColumnTransformer + XGBRegressor</span>
                        </div>
                        <div class="d-flex justify-content-between small text-muted mt-1"><span>Categorical Handling:</span>
                            <span class="text-light">OneHotEncoder (Ignore Unknowns)</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <div class="container text-center">
            <p class="m-0">Designed & Engineered by Data Science Team &bull; Powered by Flask & XGBoost</p>
        </div>
    </footer>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template_string(
        HTML_TEMPLATE, 
        family_options=FAMILY_OPTIONS, 
        prediction=None,
        model_loaded=MODEL_LOADED
    )

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return render_template_string(
            HTML_TEMPLATE, 
            family_options=FAMILY_OPTIONS, 
            prediction=None,
            model_loaded=False
        )

    try:
        # Extract fields from HTML Form
        input_data = {
            'id': [int(request.form.get('id'))],
            'store_nbr': [int(request.form.get('store_nbr'))],
            'family': [request.form.get('family')],
            'onpromotion': [int(request.form.get('onpromotion'))],
            'year': [int(request.form.get('year'))],
            'month': [int(request.form.get('month'))],
            'day': [int(request.form.get('day'))],
            'week': [int(request.form.get('week'))],
            'dayofweek': [int(request.form.get('dayofweek'))]
        }

        # Build DataFrame maintaining correct column sequence required by Pipeline
        df = pd.DataFrame(input_data)[ALL_FEATURES]
        
        # Execute prediction pipeline
        prediction = model.predict(df)[0]
        
        return render_template_string(
            HTML_TEMPLATE, 
            family_options=FAMILY_OPTIONS, 
            prediction=prediction,
            model_loaded=True
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
