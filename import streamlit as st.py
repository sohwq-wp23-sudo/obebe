import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Obesity Level Prediction App")

menu = st.sidebar.selectbox("Choose Section", ["Home", "Predict Obesity Level", "Model Comparison"])

if menu == "Home":
    st.subheader("Welcome")
    st.write("This application predicts obesity levels based on user lifestyle and physical characteristics.")

elif menu == "Predict Obesity Level":
    st.subheader("Prediction Section")
    st.write("Put your prediction form here.")

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
    fig, ax = plt.subplots()
    ax.bar(df["Model"], df["Accuracy"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison of Models")
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.write("""
    **Conclusion:** Random Forest is selected as the final model because it achieved the best overall performance
    compared to Logistic Regression, KNN, and Decision Tree.
    """)
