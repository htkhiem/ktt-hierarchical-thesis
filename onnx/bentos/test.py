import requests
print(requests.post(
    "http://127.0.0.1:5000/predict",
    headers={"content-type": "application/json"},
    data="Apple iPhone 6S 64GB").text)
