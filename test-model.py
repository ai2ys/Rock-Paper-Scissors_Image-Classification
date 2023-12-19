import requests
import json
import numpy as np
from PIL import Image

# Load and preprocess the image

image_path = '/app/tensorflow_datasets/test_samples/paper_01.png'
image = Image.open(image_path)
image = image.resize((160, 160))  # Replace with the size expected by your model
image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
image = image.tolist()  # Convert to list

class_mapping = {"0": "rock", "1": "paper", "2": "scissors"}

# The data to send in the request body
data = {
    "signature_name": "serving_default",
    "instances": [image]
}

# Convert the data to JSON
data = json.dumps(data)

# Send a POST request to the predict API
response = requests.post("http://tf-serving:8501/v1/models/rock_paper_scissors:predict", data=data)

# Print the response
print(image_path)
print(response.json())
prediction_values = response.json()['predictions'][0]
prediciton_max_index = np.argmax(response.json()['predictions'][0])
predicted_class = class_mapping[str(prediciton_max_index)]
predicted_probabiltiy = prediction_values[prediciton_max_index]
print(f"Predicted label: {predicted_class}, {predicted_probabiltiy}")
