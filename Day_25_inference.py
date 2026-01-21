import pandas as pd
import os
import keras
from Day_6_data_cleaner import DataCleaner
from keras.models import load_model



# create a fake person data to make prediction on
poor_guy  = {
    "Pclass":3,
    "Age": 23,
    "SibSp":0,
    "Parch":0,
    "Fare":20,
    "Sex": "male",
    "Embarked":"Q",

}
rich_lady  = {
    "Pclass":1,
    "Age": 28,
    "SibSp":0,
    "Parch":0,
    "Fare":500,
    "Sex": "female",
    "Embarked":"S",

}


def predict_survival(passenger_dict):
    """
    Takes a dictionary of passsenger's details and return a survial probalility
    """
    # instantiate  cleaner class
    cleaner = DataCleaner()

    # check if the scaler exist before loading it
    scaler_path = "data/processor/scaler.joblib"
    # if os.path.exists(scaler_path):
    #     # load the scaler
    #     cleaner.load_scaler(scaler_path)
    #     print(f"scaler successfully loaded from {scaler_path}")
    # else:
    #     print(f"Scaler not found at {scaler_path}")
    #     print("Available scaler are :" ,[f for f in os.listdir("data/processor") if f.endswith(".joblib")])

    cleaner.load_scaler(scaler_path)
    
    # put passenger into  Dataclass dataframe
    cleaner.df = pd.DataFrame([passenger_dict])  

    # apply the same cleaning logic as to the training data with is_training set to false
    df_input = cleaner.clean_all(
        missing_col="Age",
        strategy="median",
        scale_col="Fare",
        encode_cols=[],
        remove_cols=[],
        is_training=False
        )
    # the one hot encoding is not going to work as it is only seeing a single exemple at a time so in will hand code it
    if "Sex" in df_input.columns:
        df_input["Sex_male"] = 1 if df_input["Sex"].values[0] == "male" else 0
        df_input = df_input.drop(columns="Sex")
    if "Embarked" in df_input.columns:
        df_input["Embarked_Q"] = 1 if df_input["Embarked"].values[0] == "Q" else 0
        df_input["Embarked_S"] = 1 if df_input["Embarked"].values[0] == "S" else 0
        df_input = df_input.drop(columns="Embarked")


    # load the model
    # check if the model exist before loading
    model_path = "models/best_titanic_nn_model.keras"  # Use an existing model
    if os.path.exists(model_path):
        loaded_model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        print("Available models:", [f for f in os.listdir("models") if f.endswith('.keras')])
        return None
    
    print(df_input.head())
    # define the features to match that of the training and test sets
    features = ["Pclass","Age","SibSp","Parch","Fare","Sex_male","Embarked_Q","Embarked_S"]
    X_new = df_input.reindex(columns=features)
    print(X_new)

    # 5 predicte the passenger survival probability
    prediction = loaded_model.predict(X_new)    
    probability = prediction[0][0]
     
    return probability


print(f"rich lady survival probability: {predict_survival(rich_lady):.2%}")
print(f"poor guy survival probability: {predict_survival(poor_guy):.2%}")
