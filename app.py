from flask import Flask, request, render_template
import joblib
import pandas as pd
import shap
import numpy as np

app = Flask(__name__)

def identity_func(x):
    return x

def inverse_clip(y):
    return np.clip(y, 0.0, 1.0)

model = joblib.load('client_risk_score_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'INV_PROFIT_MARGIN': float(request.form['INV_PROFIT_MARGIN']),
        'INV_AVG_OUTSTANDING_RATIO': float(request.form['INV_AVG_OUTSTANDING_RATIO']),
        'INV_HIGH_VALUE_RATIO': float(request.form['INV_HIGH_VALUE_RATIO']),
        'INV_FREQ': float(request.form['INV_FREQ']),
        'PAY_WEIGHTED_AVG_DELAY': float(request.form['PAY_WEIGHTED_AVG_DELAY']),
        'PAY_COLLECTION_EFFICIENCY': float(request.form['PAY_COLLECTION_EFFICIENCY']),
        'CLIENT_LIFESPAN': float(request.form['CLIENT_LIFESPAN']),
        'FOREIGN': int(request.form['FOREIGN']),
        'TOTAL_UNPAID_CHQ_EFF': float(request.form['TOTAL_UNPAID_CHQ_EFF']),
        'NUM_UNPAID_CHQ_EFF': int(request.form['NUM_UNPAID_CHQ_EFF']),
        'CREDIT_MEMO_TO_SALES_RATIO': float(request.form['CREDIT_MEMO_TO_SALES_RATIO']),
        'REFUND_TO_SALES_RATIO': float(request.form['REFUND_TO_SALES_RATIO'])
    }

    data_df = pd.DataFrame([data])

    preprocessor = model.named_steps['preprocessor']

    data_preprocessed = preprocessor.transform(data_df)

    prediction = model.named_steps['regressor'].predict(data_preprocessed)[0]

    core_model = model.named_steps['regressor'].regressor_
    explainer = shap.TreeExplainer(core_model)
    shap_values = explainer.shap_values(data_preprocessed)
    feature_importances = shap_values[0]

    explanation = []
    for i, feature in enumerate(data_df.columns):
        if feature_importances[i] > 0:
            explanation.append(f"{feature} is high (+{feature_importances[i]:.4f})")
        else:
            explanation.append(f"{feature} is low ({feature_importances[i]:.4f})")

    explanation_str = "\n".join(explanation)

    return render_template('result.html', prediction=prediction, explanation=explanation_str)

if __name__ == '__main__':
    app.run(debug=True)