from flask import Flask
from flask import request
from flask import jsonify

import requests
import json
import numpy as np
from PIL import Image
from io import BytesIO

import logging

class_mapping = {"0": "rock", "1": "paper", "2": "scissors"}
target_size = (160, 160)

app = Flask('rock-paper-scissors-model')
app.logger.setLevel(logging.DEBUG)


def load_and_preprocess_image(image_url:str):
    # Fetch the image from the URL
    response = requests.get(image_url)
    image = None
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image = image.resize(target_size)  # Replace with the size expected by your model
        image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image = image.tolist()  # Convert to list
    else:
        print("Error fetching image from URL")
    return image


@app.route('/predict', methods=['POST'])
def predict():
    url = request.get_json()
    app.logger.debug(f"url: {url['url']}")
    app.logger.info(f"url: {url['url']}")

    image = load_and_preprocess_image(url['url'])
    # The data to send in the request body
    data = {
        "signature_name": "serving_default",
        "instances": [image]
    }
    #app.logger.debug(f"data: {data}")

    # Convert the data to JSON
    data = json.dumps(data)

    # Send a POST request to the predict API
    response = requests.post("http://tf-serving:8501/v1/models/rock_paper_scissors:predict", data=data)
    app.logger.debug(f"response: {response}")
    app.logger.debug(f"{response.json()}")
    # print(response.json())

    prediction_values = response.json()['predictions'][0]
    prediciton_max_index = np.argmax(response.json()['predictions'][0])
    
    predicted_class = class_mapping[str(prediciton_max_index)]
    predicted_probabiltiy = prediction_values[prediciton_max_index]
    
    app.logger.debug(f"Predicted label: {predicted_class}, {predicted_probabiltiy}")

    result = {
        'predicted_class': predicted_class,
        'predicted_prob': f"{predicted_probabiltiy:.03f}",
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)