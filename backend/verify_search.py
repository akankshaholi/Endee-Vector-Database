import requests
import json

BASE_URL = "http://localhost:5000"

test_queries = [
    "spicy food",
    "pizza",
    "biryani",
    "healthy breakfast"
]

def run_tests():
    print(f"--- Running Image & Metadata Verification ---")
    all_passed = True
    for query in test_queries:
        print(f"\nTesting Query: '{query}'")
        try:
            r = requests.get(f"{BASE_URL}/search", params={"q": query})
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            count = len(results)
            print(f"Results Count: {count}")
            
            if count > 0:
                print("Status: PASSED (Found results)")
                for i, res in enumerate(results[:2]):
                    name = res.get('name', 'Unknown')
                    img = res.get('image_url', 'MISSING')
                    score = res.get('enhanced_score', 0)
                    print(f"  {i+1}. {name} (Score: {score:.4f})")
                    print(f"     Image: {img}")
                    if img == 'MISSING' or not img.startswith('http'):
                        print("     WARNING: Image URL is invalid or missing!")
                        all_passed = False
            else:
                print(f"Status: FAILED (No results for '{query}')")
                all_passed = False
        except Exception as e:
            print(f"Error testing query '{query}': {e}")
            all_passed = False
    
    if all_passed:
        print("\n--- ALL IMAGE & METADATA TESTS PASSED ---")
    else:
        print("\n--- SOME TESTS FAILED ---")

if __name__ == "__main__":
    run_tests()
