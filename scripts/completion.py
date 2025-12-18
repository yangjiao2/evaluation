import requests


class ChatCompletionClient:
    def __init__(self, base_url: str, headers: dict = None, timeout_secs: int = 30):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.REQUESTS_TIMEOUT_SECS = timeout_secs
        self.session = requests.Session()

    def post_data(self, options: dict):
        api_endpoint = f"{self.base_url}"

        try:
            response: requests.Response = self.session.post(
                api_endpoint,
                headers=self.headers,
                json=options,
                timeout=self.REQUESTS_TIMEOUT_SECS,
                stream=False,
            )

            # Raise for HTTP errors
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error during POST request: {e}")
            return None


# Example usage
if __name__ == "__main__":
    client = ChatCompletionClient(
        base_url="https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/6bebc771-bd0a-4ecc-8599-4987f2b19829",
        headers={"Authorization": "Bearer <nvdev_it_ent_ai>"}
    )

    payload = {
        "model": "mistralai/mixtral-8x22b-instruct-v0.1",
        "messages": [
        {
          "content":"What should I do for a 4 day vacation at Cape Hatteras National Seashore?",
          "role": "user"
        }],
        "top_p": 1,
        "n": 1,
        "max_tokens": 1024,
        "stream": False,
        "frequency_penalty": 0.0,
        "stop": ["STOP"]
      }
    result = client.post_data(payload)
    print(result)
