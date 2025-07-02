import json
import requests

# Base URL of your running FastAPI application
BASE_URL = "http://127.0.0.1:8000"

# --- Test GET endpoint ---
print("--- Testing GET / ---")
# TODO: send a GET using the URL http://127.0.0.1:8000
r = requests.get(BASE_URL) # Your code here

# TODO: print the status code
print(f"Status Code for GET /: {r.status_code}")
# TODO: print the welcome message
print(f"Welcome Message (GET /): {r.json().get('message', 'No message found')}\n")


# --- Test POST endpoint for inference ---
print("--- Testing POST /data/ ---")
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
# requests.post's json parameter automatically handles converting the dict to JSON
# and setting the Content-Type header.
r = requests.post(f"{BASE_URL}/data/", json=data) # Your code here

# TODO: print the status code
print(f"Status Code for POST /data/: {r.status_code}")
# TODO: print the result
# Check if the request was successful before trying to parse JSON
if r.status_code == 200:
    print(f"Inference Result (POST /data/): {r.json().get('prediction', 'No prediction found')}")
else:
    print(f"Error Response (POST /data/): {r.text}")