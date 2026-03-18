import requests
import json
import msgpack
import numpy as np

BASE_URL = "http://localhost:8080/api/v1"

def check():
    print("--- Checking Index List ---")
    res = requests.get(f"{BASE_URL}/index/list")
    data = res.json()
    print(f"List type: {type(data)}")
    print(f"List content: {data}")
    
    COLLECTION_NAME = "food_items"
    print(f"\n--- Checking Search in {COLLECTION_NAME} ---")
    payload = {
        "vector": [0.1] * 384,
        "k": 1
    }
    res = requests.post(f"{BASE_URL}/index/{COLLECTION_NAME}/search", json=payload)
    print(f"Search status: {res.status_code}")
    if res.status_code == 200:
        results = msgpack.unpackb(res.content, raw=False)
        print(f"Search results type: {type(results)}")
        print(f"Search results sample: {results[:1] if results else 'Empty'}")
    else:
        print(f"Search failed: {res.text}")

if __name__ == "__main__":
    check()
