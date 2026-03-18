from endee_client import EndeeClient
from sentence_transformers import SentenceTransformer
import json
import os
import time
import requests

ENDEE_URL = "http://localhost:8080"
ENDEE_TOKEN = "endee_token"
COLLECTION_NAME = "food_items"

client = EndeeClient(base_url=ENDEE_URL, auth_token=ENDEE_TOKEN)
model = SentenceTransformer('all-MiniLM-L6-v2')

def reseed():
    print("--- Starting Robust Re-seeding Process ---")
    
    # 1. Force Delete existing collection
    print(f"Attempting to clear existing collection '{COLLECTION_NAME}'...")
    try:
        # Use direct requests for more control over error handling
        delete_url = f"{ENDEE_URL}/api/v1/index/{COLLECTION_NAME}/delete"
        r = requests.delete(delete_url, headers={"Authorization": ENDEE_TOKEN})
        if r.status_code == 200:
            print("Collection deleted successfully.")
        elif r.status_code == 404:
            print("Collection did not exist. Skipping delete.")
        else:
            print(f"Delete response: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Warning during delete: {e}")

    time.sleep(1)

    # 2. Create new collection
    print(f"Creating new collection '{COLLECTION_NAME}'...")
    try:
        client.create_collection(COLLECTION_NAME, dimension=384)
        print("Collection created.")
    except Exception as e:
        if "409" in str(e):
            print("Collection already exists despite delete attempt. Proceeding to insert.")
        else:
            print(f"Error creating collection: {e}")
            return

    time.sleep(1)

    # 3. Load expanded data
    data_path = os.path.join(os.path.dirname(__file__), '../data/sample_data.json')
    with open(data_path, 'r') as f:
        food_items = json.load(f)

    # 4. Generate embeddings
    print(f"Generating embeddings for {len(food_items)} items...")
    ids = [item['id'] for item in food_items]
    # Enriched text for better semantic search
    texts = [f"{item['name']}: {item['description']} ({', '.join(item.get('tags', []))}) cuisine: {item['cuisine']} rating: {item['rating']}" for item in food_items]
    embeddings = model.encode(texts)

    # 5. Store in Endee
    print("Storing 30 vectors in Endee...")
    client.add_vectors(COLLECTION_NAME, ids, embeddings.tolist(), metadata=food_items)
    print("--- Success: 30 items successfully stored in Endee ---")
    print("Data successfully stored in Endee")

if __name__ == "__main__":
    reseed()
