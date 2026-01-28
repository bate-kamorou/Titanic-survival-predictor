import streamlit as st
from Day_25_inference import predict_survival

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
        model_path  = "model/best_rf_estimator.joblib"
    else:
        model_path = "model/best_titanic_removed_nn_model.keras"

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





