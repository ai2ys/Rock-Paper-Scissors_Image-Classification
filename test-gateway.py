import requests
import os

# set default values
ip_gateway = os.getenv('IP_GATEWAY', 'localhost')

url = "https://github.com/ai2ys/mlzoomcamp-capstone-1/raw/main/tensorflow_datasets/test_samples/paper_00.png"

req = {
    "url": url
}

url = f'http://{ip_gateway}:9696/predict'

response = requests.post(url, json=req)
print(response)
print(response.json())

