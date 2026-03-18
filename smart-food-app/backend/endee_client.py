import requests
import json
import numpy as np

class EndeeClient:
    """
    A simple HTTP client for interacting with the Endee Vector Database.
    Handles creating indexes, inserting vectors, and performing similarity searches.
    """
    def __init__(self, base_url="http://localhost:8080", auth_token=None):
        self.base_url = f"{base_url}/api/v1"
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = auth_token

    def create_collection(self, name, dimension, metric="cosine"):
        """Creates a new vector index in Endee."""
        url = f"{self.base_url}/index/create"
        payload = {
            "index_name": name,
            "dim": dimension,
            "space_type": metric
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        try:
            return response.json()
        except:
            return {"message": response.text}

    def add_vectors(self, collection_name, ids, vectors, metadata=None):
        """Inserts vectors and their metadata into a collection."""
        url = f"{self.base_url}/index/{collection_name}/vector/insert"
        data = []
        for i, vid in enumerate(ids):
            item = {
                "id": str(vid),
                "vector": vectors[i].tolist() if isinstance(vectors[i], np.ndarray) else vectors[i]
            }
            if metadata and i < len(metadata):
                item["meta"] = json.dumps(metadata[i])
            data.append(item)
        
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return {"status": "success"}

    def search(self, collection_name, vector, limit=5, filter_query=None):
        """Performs a nearest-neighbor similarity search against a collection."""
        url = f"{self.base_url}/index/{collection_name}/search"
        payload = {
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector,
            "k": limit
        }
        if filter_query:
            formatted_filter = []
            for k, v in filter_query.items():
                formatted_filter.append({k: {"$eq": v}})
            payload["filter"] = json.dumps(formatted_filter)
            
        response = requests.post(url, json=payload, headers=self.headers)
        
        if response.status_code == 200:
            import msgpack
            raw_results = msgpack.unpackb(response.content, raw=False)
            
            normalized_results = []
            for res in raw_results:
                if isinstance(res, list) and len(res) >= 3:
                    item = {
                        "score": res[0],
                        "id": res[1],
                        "meta": res[2].decode('utf-8') if isinstance(res[2], bytes) else res[2]
                    }
                    normalized_results.append(item)
                else:
                    normalized_results.append(res)
                    
            return {"results": normalized_results}
        
        response.raise_for_status()
        return response.json()

    def list_collections(self):
        """Lists all active indexes in Endee."""
        url = f"{self.base_url}/index/list"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def check_health(self):
        """Returns True if the Endee server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
