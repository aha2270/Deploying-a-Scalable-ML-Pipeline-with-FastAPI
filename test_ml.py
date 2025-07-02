import os
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# Import functions from your ml modules
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, save_model, load_model, performance_on_categorical_slice

# Fixture for sample data
@pytest.fixture
def sample_data():
    """
    Provides a small, consistent DataFrame for testing.
    """
    data = {
        'age': [30, 45, 25, 50, 35],
        'workclass': ['Private', 'Federal-gov', 'Private', 'Self-emp-inc', 'Private'],
        'education': ['Bachelors', 'Masters', 'HS-grad', 'Doctorate', 'Some-college'],
        'marital-status': ['Married-civ-spouse', 'Married-civ-spouse', 'Single', 'Married-civ-spouse', 'Divorced'],
        'occupation': ['Sales', 'Exec-managerial', 'Other-service', 'Prof-specialty', 'Tech-support'],
        'relationship': ['Husband', 'Wife', 'Own-child', 'Husband', 'Not-in-family'],
        'race': ['White', 'Asian-Pac-Islander', 'Black', 'White', 'White'],
        'sex': ['Male', 'Female', 'Male', 'Male', 'Female'],
        'hours-per-week': [40, 50, 30, 60, 45],
        'native-country': ['United-States', 'United-States', 'Mexico', 'United-States', 'United-States'],
        'salary': ['>50K', '<=50K', '<=50K', '>50K', '<=50K'] # Target label
    }
    df = pd.DataFrame(data)
    return df

# Fixture for categorical features and label
@pytest.fixture
def cat_features():
    return [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

@pytest.fixture
def label_name():
    return "salary"

# --- Tests for ml.data.process_data ---
def test_process_data_training(sample_data, cat_features, label_name):
    """
    Tests process_data in training mode:
    - Correct output shapes (X, y)
    - Returns fitted encoder and lb
    - X_categorical is one-hot encoded (check non-zero values count or specific shape changes)
    - y is binarized
    """
    X_processed, y_processed, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label=label_name, training=True
    )

    # Check X_processed type and shape
    assert isinstance(X_processed, np.ndarray)
    # Expected number of columns: 2 (age, hours-per-week) + one-hot encoded columns
    # Hardcoding expected columns for sample_data:
    # 'workclass': 5 unique -> 5 cols
    # 'education': 5 unique -> 5 cols
    # 'marital-status': 4 unique -> 4 cols
    # 'occupation': 5 unique -> 5 cols
    # 'relationship': 5 unique -> 5 cols
    # 'race': 3 unique -> 3 cols
    # 'sex': 2 unique -> 2 cols
    # 'native-country': 3 unique -> 3 cols
    # Total categorical OHE cols = 5+5+4+5+5+3+2+3 = 32
    # Continuous cols = 2 (age, hours-per-week)
    # Total X_processed cols = 2 + 32 = 34
    assert X_processed.shape[1] == 29 # Based on distinct categories in sample_data
    assert X_processed.shape[0] == len(sample_data)

    # Check y_processed type and values
    assert isinstance(y_processed, np.ndarray)
    assert np.all(np.isin(y_processed, [0, 1])) # y should be binarized (0 or 1)
    assert y_processed.shape == (len(sample_data),)

    # Check encoder and lb are fitted objects
    assert isinstance(encoder, OneHotEncoder)
    assert hasattr(encoder, 'categories_') # Check if fitted
    assert isinstance(lb, LabelBinarizer)
    assert hasattr(lb, 'classes_') # Check if fitted

def test_process_data_inference(sample_data, cat_features, label_name):
    """
    Tests process_data in inference mode:
    - Correct output shapes (X, y)
    - Uses provided encoder and lb
    - Does not fit new encoder/lb
    """
    # First, train an encoder and lb from sample_data (as if from training)
    _, _, trained_encoder, trained_lb = process_data(
        sample_data, categorical_features=cat_features, label=label_name, training=True
    )

    # Create new data (could be similar to sample_data or a subset)
    inference_data = sample_data.iloc[0:2].copy()

    X_processed_inf, y_processed_inf, encoder_inf, lb_inf = process_data(
        inference_data,
        categorical_features=cat_features,
        label=label_name,
        training=False,
        encoder=trained_encoder,
        lb=trained_lb
    )

    # Check shapes
    assert X_processed_inf.shape[0] == len(inference_data)
    assert X_processed_inf.shape[1] == 29 # Must match training data columns

    # Check y_processed_inf type and values
    assert isinstance(y_processed_inf, np.ndarray)
    assert np.all(np.isin(y_processed_inf, [0, 1]))
    assert y_processed_inf.shape == (len(inference_data),)

    # Ensure encoder_inf and lb_inf are the *same* objects passed in (not new ones)
    assert encoder_inf is trained_encoder
    assert lb_inf is trained_lb

def test_process_data_inference_no_label(sample_data, cat_features):
    """
    Tests process_data in inference mode with no label provided.
    - y_processed should be an empty array
    """
    # First, train an encoder from sample_data
    _, _, trained_encoder, trained_lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )

    inference_data_no_label = sample_data.drop(columns=['salary']).iloc[0:2].copy()

    X_processed_inf, y_processed_inf, _, _ = process_data(
        inference_data_no_label,
        categorical_features=cat_features,
        label=None, # No label provided
        training=False,
        encoder=trained_encoder,
        lb=trained_lb
    )

    assert isinstance(y_processed_inf, np.ndarray)
    assert y_processed_inf.size == 0 # Should be empty


# --- Tests for ml.model.train_model ---
def test_train_model(sample_data, cat_features, label_name):
    """
    Tests that train_model returns a fitted model.
    """
    X_train, y_train, _, _ = process_data(
        sample_data, categorical_features=cat_features, label=label_name, training=True
    )

    model = train_model(X_train, y_train)

    # Check if the returned object is a scikit-learn LogisticRegression model (or whatever you chose)
    assert isinstance(model, LogisticRegression)
    # Check if the model has been fitted (it should have 'coef_' and 'intercept_' attributes)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')

# --- Tests for ml.model.inference ---
def test_inference(sample_data, cat_features, label_name):
    """
    Tests that inference returns predictions of the correct shape and type.
    """
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label=label_name, training=True
    )
    model = train_model(X_train, y_train)

    X_test, _, _, _ = process_data(
        sample_data.iloc[0:2], # Use a subset for test inference
        categorical_features=cat_features,
        label=label_name,
        training=False,
        encoder=encoder,
        lb=lb
    )

    predictions = inference(model, X_test)

    # Check type and shape of predictions
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X_test.shape[0],) # Should be 1D array of predictions
    # Check that predictions are binary (0 or 1)
    assert np.all(np.isin(predictions, [0, 1]))

# --- Tests for ml.model.save_model and load_model ---
def test_save_load_model(tmpdir, sample_data, cat_features, label_name):
    """
    Tests that models/encoders/binarizers can be saved and loaded correctly.
    tmpdir is a pytest fixture that provides a temporary directory for tests.
    """
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label=label_name, training=True
    )
    model = train_model(X_train, y_train)

    model_path = os.path.join(tmpdir, "test_model.pkl")
    encoder_path = os.path.join(tmpdir, "test_encoder.pkl")
    lb_path = os.path.join(tmpdir, "test_lb.pkl")

    save_model(model, model_path)
    save_model(encoder, encoder_path)
    save_model(lb, lb_path)

    loaded_model = load_model(model_path)
    loaded_encoder = load_model(encoder_path)
    loaded_lb = load_model(lb_path)

    # Check types of loaded objects
    assert isinstance(loaded_model, LogisticRegression)
    assert isinstance(loaded_encoder, OneHotEncoder)
    assert isinstance(loaded_lb, LabelBinarizer)

    # Perform a simple check to ensure they are functional (e.g., predict, transform)
    # This is a basic check, a more rigorous one would involve comparing attributes or outputs
    assert hasattr(loaded_model, 'coef_')
    assert hasattr(loaded_encoder, 'categories_')
    assert hasattr(loaded_lb, 'classes_')

# --- Tests for ml.model.compute_model_metrics (already tested in train_model.py logic, but good to have a dedicated test) ---
def test_compute_model_metrics():
    """
    Tests compute_model_metrics with known inputs to ensure correct calculation.
    """
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0]) # One false positive, one false negative

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Expected calculations:
    # True Positives (TP): 2 (y_true=1, y_pred=1)
    # False Positives (FP): 1 (y_true=0, y_pred=1)
    # False Negatives (FN): 1 (y_true=1, y_pred=0)
    # True Negatives (TN): 2 (y_true=0, y_pred=0)
    # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3 = 0.6666...
    # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3 = 0.6666...
    # F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2 * (4/9) / (4/3) = 8/9 * 3/4 = 2/3 = 0.6666...

    assert np.isclose(precision, 2/3, atol=1e-4)
    assert np.isclose(recall, 2/3, atol=1e-4)
    assert np.isclose(fbeta, 2/3, atol=1e-4)

# --- Tests for ml.model.performance_on_categorical_slice (Optional, but good for coverage) ---
# This one is more complex as it involves process_data and inference.
# You could mock process_data/inference or ensure robust setup.
# For simplicity and due to its complexity involving other functions,
# a basic check that it runs without error is a good start.
def test_performance_on_categorical_slice_runs(sample_data, cat_features, label_name):
    """
    Tests that performance_on_categorical_slice runs without error for a given slice.
    """
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label=label_name, training=True
    )
    model = train_model(X_train, y_train)

    # Test with a known slice that exists in sample_data
    precision, recall, fbeta = performance_on_categorical_slice(
        sample_data,
        column_name="workclass",
        slice_value="Private",
        categorical_features=cat_features,
        label=label_name,
        encoder=encoder,
        lb=lb,
        model=model
    )
    # Check that metrics are floats and within plausible range (0 to 1)
    assert isinstance(precision, float) and 0.0 <= precision <= 1.0
    assert isinstance(recall, float) and 0.0 <= recall <= 1.0
    assert isinstance(fbeta, float) and 0.0 <= fbeta <= 1.0

def test_performance_on_categorical_slice_empty(sample_data, cat_features, label_name):
    """
    Tests performance_on_categorical_slice with a slice that doesn't exist (empty data).
    It should return 0.0, 0.0, 0.0 as per the function's empty check.
    """
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label=label_name, training=True
    )
    model = train_model(X_train, y_train)

    # Test with a slice value that does not exist in the sample_data
    precision, recall, fbeta = performance_on_categorical_slice(
        sample_data,
        column_name="education",
        slice_value="NonExistentDegree", # This value is not in sample_data
        categorical_features=cat_features,
        label=label_name,
        encoder=encoder,
        lb=lb,
        model=model
    )
    # Should return the default values set for empty slices
    assert precision == 0.0
    assert recall == 0.0
    assert fbeta == 0.0