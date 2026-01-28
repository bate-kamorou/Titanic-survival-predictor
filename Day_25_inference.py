import pandas as pd
import os
from Day_6_data_cleaner import DataCleaner
from keras.models import load_model
import joblib



# create a fake person data to make prediction on
# poor_guy  = {
#     "Pclass":3,
#     "Age": 23,
#     "SibSp":0,
#     "Parch":0,
#     "Fare":20,
#     "Sex": "male",
#     "Embarked":"Q",

# }
# rich_lady  = {
#     "Pclass":1,
#     "Age": 28,
#     "SibSp":0,
#     "Parch":0,
#     "Fare":500,
#     "Sex": "female",
#     "Embarked":"S"
# }


def predict_survival(passenger_dict, model_path):
    """
    Takes a dictionary of passsenger's details and return a survial probalility
    """
    # instantiate  cleaner class
    cleaner = DataCleaner()

    # check if the scaler exist before loading it
    scaler_path = "scaler.joblib"
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
        remove_cols=["SibSp","Parch"],
        is_training=False
        )


    # take a look at the processed input data
    print("Processed input data for prediction:\n", df_input)

    # load the model
    # check if the model exist before loading

    # model_path = "models/best_titanic_nn_model.keras"  # Use an existing model
    model_path = model_path  # Use an existing model
    # load either keras model or joblib model based on the file extension to make a prediction
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available models:", [f for f in os.listdir("model") if f.endswith('.keras') or f.endswith('.joblib')])
        return None
    
    if model_path.endswith(".keras"):
        loaded_model = load_model(model_path)
        if loaded_model is None:
            print(f"Failed to load model from {model_path}")
            return None
        print(f"Model loaded successfully from {model_path}")
        # make prediction with the loaded model
        prediction = loaded_model.predict(df_input)    
        probability = prediction[0][0]      
        return probability
    elif model_path.endswith(".joblib"):
        loaded_model = joblib.load(model_path)
        if not loaded_model:
            print(f"Failed to load model from {model_path}")
            return None
        print(f"Model loaded successfully from {model_path}")
        prediction = loaded_model.predict_proba(df_input)
        probability = prediction[0][1]
        return probability
    else:
        print("Unrecognized model format. Please use a .keras or .joblib model.")
        return None 
    

# # make a prediction
# try:    
#     rich_lady_survival = predict_survival(rich_lady)
#     print(f"rich lady survival probability: {rich_lady_survival:.2%}")  
# except Exception as e:
#     print(f"Error during prediction: {e}")

# print(f"poor guy survival probability: {predict_survival(poor_guy):.2%}")

