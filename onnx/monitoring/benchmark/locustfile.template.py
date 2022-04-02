from locust import HttpUser, task


TERMS =


class KttUser(HttpUser):
    """Simulated user querying classifications from KTT."""

    @task
    def send_fixed_query(self):
        """Send a query with a fixed text sample repeatedly."""
        self.client.post(
            "/predict",
            headers={"content-type": "application/json"},
            data='Tomato ketchup 10 oz'
        )

    def send_trained_query():
