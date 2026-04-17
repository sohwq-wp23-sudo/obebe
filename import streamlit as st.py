import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# -----------------------------
# Page Title
# -----------------------------
st.title("Obesity Level Prediction App")

# -----------------------------
# Sidebar Menu
# -----------------------------
menu = st.sidebar.selectbox(
    "Choose Section",
    ["Home", "Predict Obesity Level", "Model Comparison"]
)

# -----------------------------
# Load Model Assets
# -----------------------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("random_forest_classifier (2).pkl", "rb"))
    scaler = pickle.load(open("scaler (2).pkl", "rb"))
    le = pickle.load(open("label_encoder (2).pkl", "rb"))
    return model, scaler, le


# -----------------------------
# Home Page
# -----------------------------
if menu == "Home":
    st.subheader("Welcome")
    st.write(
        "This application predicts obesity levels based on user lifestyle habits "
        "and physical characteristics."
    )
    st.write(
        "Use the sidebar to navigate to the prediction section or view the model comparison."
    )


# -----------------------------
# Prediction Page
# -----------------------------
elif menu == "Predict Obesity Level":
    st.subheader("Prediction Section")

    try:
        model, scaler, le = load_assets()
    except FileNotFoundError:
        st.error(
            "Required .pkl files not found. Please make sure these files are in the same folder:\n"
            "- random_forest_classifier (2).pkl\n"
            "- scaler (2).pkl\n"
            "- label_encoder (2).pkl"
        )
        st.stop()

    # Exact column order used during training
    train_cols = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI',
        'Gender_Male', 'family_history_with_overweight_yes', 'FAVC_yes',
        'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'SMOKE_yes', 'SCC_yes',
        'CALC_Frequently', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike',
        'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking'
    ]

    st.write("Enter your health and lifestyle details below:")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=1.0, max_value=100.0, value=25.0)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0)
        family_history = st.selectbox("Family history with overweight?", ["yes", "no"])
        favc = st.selectbox("Frequent high caloric food consumption?", ["yes", "no"])
        fcvc = st.slider("Vegetable consumption frequency (FCVC)", 1.0, 3.0, 2.0)
        ncp = st.slider("Number of main meals (NCP)", 1.0, 4.0, 3.0)

    with col2:
        caec = st.selectbox(
            "Food consumption between meals (CAEC)",
            ["Sometimes", "Frequently", "Always", "no"]
        )
        smoke = st.selectbox("Do you smoke?", ["yes", "no"])
        ch2o = st.slider("Daily water intake (CH2O)", 1.0, 3.0, 2.0)
        scc = st.selectbox("Do you monitor calories? (SCC)", ["yes", "no"])
        faf = st.slider("Physical activity frequency (FAF)", 0.0, 3.0, 1.0)
        tue = st.slider("Technology usage time (TUE)", 0.0, 2.0, 1.0)
        calc = st.selectbox(
            "Alcohol consumption (CALC)",
            ["Sometimes", "Frequently", "Always", "no"]
        )
        mtrans = st.selectbox(
            "Main transportation method (MTRANS)",
            ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"]
        )

    if st.button("Predict Obesity Level"):
        bmi_val = weight / (height ** 2)

        input_dict = {
            'Age': age,
            'Height': height,
            'Weight': weight,
            'FCVC': fcvc,
            'NCP': ncp,
            'CH2O': ch2o,
            'FAF': faf,
            'TUE': tue,
            'BMI': bmi_val,

            'Gender_Male': 1 if gender == 'Male' else 0,
            'family_history_with_overweight_yes': 1 if family_history == 'yes' else 0,
            'FAVC_yes': 1 if favc == 'yes' else 0,

            'CAEC_Frequently': 1 if caec == 'Frequently' else 0,
            'CAEC_Sometimes': 1 if caec == 'Sometimes' else 0,
            'CAEC_no': 1 if caec == 'no' else 0,

            'SMOKE_yes': 1 if smoke == 'yes' else 0,
            'SCC_yes': 1 if scc == 'yes' else 0,

            'CALC_Frequently': 1 if calc == 'Frequently' else 0,
            'CALC_Sometimes': 1 if calc == 'Sometimes' else 0,
            'CALC_no': 1 if calc == 'no' else 0,

            'MTRANS_Bike': 1 if mtrans == 'Bike' else 0,
            'MTRANS_Motorbike': 1 if mtrans == 'Motorbike' else 0,
            'MTRANS_Public_Transportation': 1 if mtrans == 'Public_Transportation' else 0,
            'MTRANS_Walking': 1 if mtrans == 'Walking' else 0
        }

        # Convert to DataFrame and arrange in training column order
        input_df = pd.DataFrame([input_dict])[train_cols]

        try:
            scaled_data = scaler.transform(input_df)
            prediction = model.predict(scaled_data)
            final_label = le.inverse_transform(prediction)

            st.markdown("---")
            st.success(f"Predicted Obesity Level: **{final_label[0]}**")
            st.info(f"Calculated BMI: **{bmi_val:.2f}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -----------------------------
# Model Comparison Page
# -----------------------------
elif menu == "Model Comparison":
    st.subheader("Model Comparison")

    data = {
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "Accuracy": [96.41, 87.56, 97.37, 98.09],
        "Precision": [96.36, 86.86, 97.45, 98.12],
        "Recall": [96.26, 87.09, 97.24, 97.96],
        "F1-score": [96.29, 86.89, 97.31, 98.00]
    }

    df = pd.DataFrame(data)

    st.write("### Performance Comparison Table")
    st.dataframe(df)

    st.write("### Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["Model"], df["Accuracy"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison of Models")
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.write("### Conclusion")
    st.write(
        "Random Forest is selected as the final model because it achieved the best "
        "overall performance compared to Logistic Regression, KNN, and Decision Tree."
    )
