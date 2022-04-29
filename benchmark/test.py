import requests
import json
import random
from locust import HttpUser, task

TEXT = [
    'apple'
]

res = requests.post(
    "http://localhost:5000/predict",
    headers={"content-type": "application/json"},
    data=json.dumps({'text': random.sample(TEXT, k=1)[0]})
)

print(res.text)