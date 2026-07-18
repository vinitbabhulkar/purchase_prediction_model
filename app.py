import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Naive Bayes model safely
MODEL_PATH = "naive_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Attractive HTML layout with embedded Tailwind CSS and a professional color scheme
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Target Marketing Predictor</title>
    <!-- Tailwind CSS CDN for elegant modern styling -->
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-50 font-sans min-h-screen flex items-center justify-center p-4">

    <div class="max-w-md w-full bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-100">
        <!-- Header Card -->
        <div class="bg-gradient-to-r from-blue-900 to-teal-700 p-6 text-center text-white">
            <h1 class="text-2xl font-bold tracking-tight">🎯 Customer Purchase Predictor</h1>
            <p class="text-slate-100 text-sm mt-1">Predict the likelihood of customer purchase engagement</p>
        </div>

        <!-- Form Block -->
        <form method="POST" class="p-6 space-y-5">
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Gender</label>
                <select name="gender" class="w-full rounded-lg border border-slate-200 p-2.5 text-slate-800 focus:ring-2 focus:ring-teal-500 focus:outline-none bg-white">
                    <option value="Male" {% if inputs and inputs.gender == 'Male' %}selected{% endif %}>Male</option>
                    <option value="Female" {% if inputs and inputs.gender == 'Female' %}selected{% endif %}>Female</option>
                </select>
            </div>

            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Age (Years)</label>
                <input type="number" name="age" min="18" max="100" value="{{ inputs.age if inputs else 35 }}" required
                       class="w-full rounded-lg border border-slate-200 p-2.5 text-slate-800 focus:ring-2 focus:ring-teal-500 focus:outline-none">
            </div>

            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Estimated Annual Salary ($)</label>
                <input type="number" name="salary" min="0" max="500000" step="1000" value="{{ inputs.salary if inputs else 50000 }}" required
                       class="w-full rounded-lg border border-slate-200 p-2.5 text-slate-800 focus:ring-2 focus:ring-teal-500 focus:outline-none">
            </div>

            <button type="submit" class="w-full bg-gradient-to-r from-blue-800 to-teal-600 hover:from-blue-900 hover:to-teal-700 text-white font-medium py-3 px-4 rounded-lg transition duration-200 transform active:scale-[0.98] shadow-md">
                Analyze Customer Profile
            </button>
        </form>

        <!-- Dynamic Results Display -->
        {% if result is not none %}
        <div class="px-6 pb-6">
            <div class="border-t border-slate-100 pt-4">
                {% if result == 1 %}
                <div class="bg-emerald-50 border border-emerald-200 rounded-xl p-4 text-center">
                    <span class="text-emerald-800 font-bold text-lg block">🎉 High Likelihood!</span>
                    <span class="text-emerald-600 text-sm">This customer is likely to purchase (Confidence: {{ "%.1f"|format(prob * 100) }}%)</span>
                </div>
                {% else %}
                <div class="bg-rose-50 border border-rose-200 rounded-xl p-4 text-center">
                    <span class="text-rose-800 font-bold text-lg block">❌ Low Likelihood.</span>
                    <span class="text-rose-600 text-sm">This customer is unlikely to purchase (Confidence: {{ "%.1f"|format(prob * 100) }}%)</span>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        {% if error %}
        <div class="px-6 pb-6">
            <div class="bg-amber-50 border border-amber-200 rounded-xl p-4 text-center text-amber-800 text-sm">
                {{ error }}
            </div>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template_string(HTML_TEMPLATE, result=None, inputs=None)
    
    if not model:
        return render_template_string(HTML_TEMPLATE, result=None, inputs=None, error="Model file 'naive_model.pkl' not found on server.")

    # Get data from form fields
    gender = request.form.get("gender")
    age = int(request.form.get("age", 35))
    salary = float(request.form.get("salary", 50000))
    
    # Bundle inputs to preserve choices on page refresh
    inputs = {"gender": gender, "age": age, "salary": salary}
    
    # Match data preprocessing (Male=1, Female=0)
    gender_encoded = 1 if gender == "Male" else 0
    
    # Create DataFrame to feed the Naive Bayes model
    input_data = pd.DataFrame([{
        'Gender': gender_encoded,
        'Age': age,
        'EstimatedSalary': salary
    }])
    
    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        prob = probabilities[1] if prediction == 1 else probabilities[0]
        
        return render_template_string(HTML_TEMPLATE, result=prediction, prob=prob, inputs=inputs)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=None, inputs=inputs, error=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
