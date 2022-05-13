"""Locustfile.

Use this file with Locus by starting a new Locust instance in the same
directory as this file.
"""
import json
import random
from locust import HttpUser, task


# Read dataset text
try:
    with open('text-strings.json', 'r') as text_json:
        TEXT = json.load(text_json)['strings']
except FileNotFoundError:
    # No dataset template found - use default strings.
    TEXT = [
        'apple', 'orange', 'sedan', 'lorry', 'table salt', 'suit', 't-shirt'
    ]

class KttUser(HttpUser):
    """Simulated user querying classifications from KTT."""

    @task
    def send_sampled_query(self):
        """Send a query with text sampled from a dataset."""
        self.client.post(
            "/predict",
            headers={"content-type": "application/json"},
            data=json.dumps({'text': random.sample(TEXT, k=1)[0]})
        )
