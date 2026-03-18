import requests
import json

ENDEE_URL = "http://localhost:8080/api/v1"
headers = {"Content-Type": "application/json", "Authorization": "endee_token"}

def diag():
    url = f"{ENDEE_URL}/index/create"
    payload = {
        "index_name": "food_items",
        "dim": 384,
        "space_type": "cosine"
    }
    print(f"Testing POST to {url}...")
    try:
        r = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {r.status_code}")
        print(f"Response Body: {r.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    diag()
