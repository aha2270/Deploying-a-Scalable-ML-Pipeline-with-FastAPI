import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np # Added this import as it's often useful and implied by numpy arrays

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Define project root and data path
# Using os.path.dirname(os.path.abspath(__file__)) gets the directory of the current script.
# Assuming train_model.py is at the root of your project:
project_root = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_root, "data", "census.csv")

print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Define the label column
LABEL = "salary" # Define the label column name

# Ensure the model directory exists
model_dir = os.path.join(project_root, "model")
os.makedirs(model_dir, exist_ok=True) # exist_ok=True prevents error if directory already exists

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
# Using a common split ratio (80/20) and random_state for reproducibility
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data[LABEL])

# DO NOT MODIFY
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

# TODO: use the process_data function provided to process the data.
# For training data, we fit the encoder and lb
X_train, y_train, encoder, lb = process_data(
    train, # use the train dataset
    categorical_features=cat_features,
    label=LABEL, # use the defined LABEL
    training=True, # use training=True
    # do not need to pass encoder and lb as input
)

# For test data, we transform using the fitted encoder and lb from training
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label=LABEL,
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(model_dir, "model.pkl") # Use model_dir here
encoder_path = os.path.join(model_dir, "encoder.pkl") # Use model_dir here
lb_path = os.path.join(model_dir, "lb.pkl") # Also save the LabelBinarizer

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path) # Save the LabelBinarizer as well, it's needed for inference if you want to inverse transform.

print(f"Model saved to {model_path}")
print(f"Encoder saved to {encoder_path}")
print(f"LabelBinarizer saved to {lb_path}")

# load the model (demonstrating loading, though not strictly necessary right after saving in this script)
# model = load_model(model_path) # Already have the model in memory, this is more for demonstrating load functionality
# encoder = load_model(encoder_path)
# lb = load_model(lb_path)

print("\nLoading model for inference...")
loaded_model = load_model(model_path) # Use a different variable name to avoid confusion
# You would load encoder and lb here if this were a separate inference script

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(loaded_model, X_test) # Use the loaded_model for inference

# Calculate and print the overall metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# Clear the slice_output.txt file before appending
with open("slice_output.txt", "w") as f:
    pass

print("\nComputing performance on data slices and saving to slice_output.txt...")
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]

        # TODO: your code here for performance_on_categorical_slice
        p, r, fb = performance_on_categorical_slice(
            data=test, # Use the original 'test' DataFrame for slicing
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label=LABEL,
            encoder=encoder, # Pass the fitted encoder
            lb=lb,           # Pass the fitted label binarizer
            model=model      # Pass the trained model
        )
        with open("slice_output.txt", "a") as f:
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f) # Adjusted order to match example output

print("Slice performance output saved to slice_output.txt")