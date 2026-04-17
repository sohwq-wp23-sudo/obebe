import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Load ML artifacts
# -------------------
with open('model_rf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# -------------------
# Page Settings
# -------------------
st.set_page_config(page_title="Obesity Level Predictor", layout="wide")
st.title("🏋️‍♀️ Obesity Level Prediction and Model Comparison")
st.markdown("Use data-driven insights and prediction models to assess obesity levels.")

# ------------------------------------------
# SIDEBAR – user input for new prediction
# ------------------------------------------
st.sidebar.header("🔢 Enter Your Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    age = st.sidebar.slider("Age", 14, 60, 25)
    height = st.sidebar.slider("Height (m)", 1.4, 2.0, 1.70)
    weight = st.sidebar.slider("Weight (kg)", 40.0, 170.0, 70.0)
    family_overweight = st.sidebar.selectbox("Family history of overweight?", ['yes','no'])
    favc = st.sidebar.selectbox("Frequent high-caloric food consumption?", ['yes','no'])
    fcvc = st.sidebar.slider("Vegetable consumption frequency (1–3)", 1.0, 3.0, 2.0)
    ncp = st.sidebar.slider("Main meals daily (1–4)", 1.0, 4.0, 3.0)
    caec = st.sidebar.selectbox("Food between meals?", ['no','Sometimes','Frequently','Always'])
    smoke = st.sidebar.selectbox("Smokes?", ['yes','no'])
    ch2o = st.sidebar.slider("Water intake (1–3)", 1.0, 3.0, 2.0)
    scc = st.sidebar.selectbox("Monitors calories?", ['yes','no'])
    faf = st.sidebar.slider("Physical activity level (0–3)", 0.0, 3.0, 1.0)
    tue = st.sidebar.slider("Technology use (0–2)", 0.0, 2.0, 1.0)
    calc = st.sidebar.selectbox("Alcohol consumption", ['no','Sometimes','Frequently','Always'])
    mtrans = st.sidebar.selectbox("Transportation", ['Automobile','Bike','Motorbike','Public_Transportation','Walking'])

    bmi = weight / (height ** 2)

    data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_overweight,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans,
        'BMI': bmi
    }
    return pd.DataFrame([data])

user_data = user_input_features()

# -------------------
# Prediction
# -------------------
st.subheader("⚙️ Prediction Result")

# Match model input columns (use same one-hot encoding as training)
categorical_cols = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
X_input = pd.get_dummies(user_data, columns=categorical_cols, drop_first=True)

# Add missing columns if any
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in X_input.columns:
        X_input[col] = 0

X_input = X_input[model_columns]
X_scaled = scaler.transform(X_input)

# Predict
prediction = model.predict(X_scaled)
pred_label = le.inverse_transform(prediction)[0]

st.success(f"### 🧭 Predicted Obesity Level: `{pred_label}`")
st.info(f"**Calculated BMI:** {user_data['BMI'][0]:.2f}")

# -------------------
# Model Comparison Plot
# -------------------
st.header("📊 Model Performance Comparison")

comparison_data = {
    'Model': ['KNN','Logistic Regression','Decision Tree','Random Forest'],
    'Accuracy': [87.56, 96.41, 97.37, 98.09],
    'Precision': [86.86, 96.36, 97.45, 98.12],
    'Recall': [87.09, 96.26, 97.24, 97.96],
    'F1-Score': [86.89, 96.29, 97.31, 98.0]
}
df_metrics = pd.DataFrame(comparison_data)

metric = st.selectbox("Select metric to view comparison", ['Accuracy','Precision','Recall','F1-Score'])
colors = ['red','green','blue','orange']

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x='Model', y=metric, data=df_metrics, palette=colors, ax=ax)
ax.set_ylim(80, 100)
ax.set_ylabel(f"{metric} (%)")
ax.set_title(f"{metric} Comparison Across Models")
st.pyplot(fig)

# Optional: summary display
st.write("### Summary Metrics Table")
st.dataframe(df_metrics.style.highlight_max(axis=0, color='lightgreen'))
