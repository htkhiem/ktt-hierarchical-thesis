import requests
# while True:
#     print(requests.post(
#         "http://127.0.0.1:5000/predict",
#         headers={"content-type": "application/json"},
#         data="Tomato ketchup 10 oz").text)

print(requests.post(
    "http://127.0.0.1:5000/predict",
    headers={"content-type": "application/json"},
    data="Tomato ketchup 10 oz").text)
