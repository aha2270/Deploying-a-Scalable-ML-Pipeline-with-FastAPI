import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

    # This is needed to allow FastAPI to convert the Pydantic model to a dictionary
    # with the correct field names (e.g., "education-num" instead of "education_num").
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            int: lambda v: int(v)
        }

# Determine the project root dynamically
# Assuming main.py is in the project root
project_root = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_root, "model")

# TODO: enter the path for the saved encoder
encoder_path = os.path.join(model_dir, "encoder.pkl")
encoder = load_model(encoder_path)
print(f"Loaded encoder from: {encoder_path}")

# TODO: enter the path for the saved model
model_path = os.path.join(model_dir, "model.pkl")
model = load_model(model_path)
print(f"Loaded model from: {model_path}")

# Load the LabelBinarizer as well, it's needed for `apply_label`
lb_path = os.path.join(model_dir, "lb.pkl")
lb = load_model(lb_path)
print(f"Loaded LabelBinarizer from: {lb_path}")


# TODO: create a RESTful API using FastAPI
app = FastAPI(
    title="Census Income Prediction API",
    description="A simple API to predict income level based on census data.",
    version="1.0.0",
)

# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"message": "Welcome to the Census Income Prediction API! Visit /docs for API documentation."}


# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict(by_alias=True) # Use by_alias=True to get original field names like "education-num"
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    # The provided data.dict() already handles aliases, so direct conversion is fine.
    # We need to ensure all features expected by process_data are present, even if they are continuous.
    # Let's rebuild the DataFrame to ensure correct column order and types.

    # Reconstruct DataFrame to match expected input for process_data
    # This ensures all columns are present and in the expected order,
    # and handles the single-row input correctly.
    input_df = pd.DataFrame([data_dict])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the input data using the pre-trained encoder
    data_processed, _, _, _ = process_data(
        input_df, # use data as data input (the input_df created above)
        categorical_features=cat_features,
        label=None, # No label for inference
        training=False, # use training = False
        encoder=encoder, # Pass the loaded encoder
        lb=lb # Pass the loaded label binarizer (even if not directly used by process_data when label=None, it's part of the pipeline context)
    )

    # Perform inference
    _inference = inference(model, data_processed) # your code here to predict the result using data_processed

    # Apply the label transformation (0 or 1 to ">50K" or "<=50K")
    result_label = apply_label(_inference)

    return {"prediction": result_label}