import streamlit as st
from Day_25_inference import predict_survival
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np 
from keras.models import load_model

# page configuration to make it looks good on mobile
st.set_page_config(page_title="Titanic predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("##### Enter passenger details to see then choose the model add see if the passenger survives the Titanic.")

# Ui input

# use columns for the layout
col1, col2 = st.columns(2)
with col1:
    p_class = st.selectbox("**Passenger's Class**", [1,2,3], )
    age  = st.slider("**Age**", 0, 80, 25)
    sex = st.radio("**Sex**", ["male", "female"])
    model = st.selectbox("**Choose Model**", ["Random Forest", "Neural Network"], index=1)

with col2:
    sibsp = st.number_input("**Sibilings /Spouses Aboard (Sibsp)**",value=0)
    parch = st.number_input("**Parent / Childern Aboaed (Parch)**", value=0, min_value=0)
    fare = st.number_input("**Fare Paid ($)**", value=32, min_value=0)
    embarked = st.selectbox("**Embarked From**", ["C", "Q", "S"], index=2)


st.divider()
# use the input to create the dataset
if st.button("**Calculate survival probalility**", type="primary"):
    data = {
        "Pclass": p_class,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare" : fare,
        "Sex": sex,
        "Embarked":embarked
    }

    if model == "Random Forest":
        model_path  = "models/best_rf_estimator.joblib"
    else:
        model_path = "models/best_titanic_removed_nn_model.keras"

    with st.spinner("**Analyzing passenger manifest...**"):
    # call make a prediction with the data
        prediction = predict_survival(data, model_path)

        # render the result with some flavors
        if prediction is not None:
            if prediction > 0.5:
                st.success(f"### ✅ Survival Likely: {prediction:.2%}")
                st.balloons()
            else:
                st.error(f"### ❌ Survival Unlikely: {prediction:.2%}")
        else:
            st.error("### ❌ Error: Could not make prediction")  

        st.subheader("💡Model Insights")
        st.write("Which factor influenced this specific prediction the most?")
        rf_explainer_path  = "models/best_rf_estimator.joblib"
        nn_explainer_path  = "models/best_titanic_removed_nn_model.keras"
    
        if model == "Random Forest" and os.path.exists(rf_explainer_path):
            explainer = joblib.load(rf_explainer_path)
            feature_importances = explainer.feature_importances_
            features = ["Pclass", "Age", "Fare", "Sex_male", "Embarked_Q", "Embarked_S", "FamilySize", "IsAlone"]
    
            # plot feature importances
            fig, ax = plt.subplots(figsize=(10, 6))
            sort_index = feature_importances.argsort()
            plt.barh([features[i] for i in sort_index], feature_importances[sort_index], color='#0078D4')
            plt.xlabel("Feature Importance")
            plt.title("Feature Importance for Random Forest Model")
            st.pyplot(fig)
        elif model == "Neural Network" and os.path.exists(nn_explainer_path):
            nn_explainer = load_model(nn_explainer_path)
            weights , baises  = nn_explainer.layers[0].get_weights()
            feature_importances = np.mean(np.abs(weights), axis=1)
            # plot the feature importances
            fig, ax = plt.subplots(figsize=(10, 6))
            sort_index = feature_importances.argsort()
            features = ["Pclass", "Age", "Fare", "Sex_male", "Embarked_Q", "Embarked_S", "FamilySize", "IsAlone"]
            plt.barh([features[i] for i in sort_index], feature_importances[sort_index], color='#FF4B4B')
            ax.set_xlabel("Average Absolute Weight (Importance Proxy)")
            ax.set_title("Feature Importance for Neural Network")
            st.pyplot(fig)
    
    st.write("---")
    st.write("##### Developed by AI Engineering Bootcamp Student Bate kamorou")












