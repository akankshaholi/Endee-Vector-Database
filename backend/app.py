from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import json
import os
import sys
import numpy as np
from endee_client import EndeeClient

app = Flask(__name__)
CORS(app)

# --- Endee Integration Configuration ---
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
COLLECTION_NAME = "food_items"

client = EndeeClient(base_url=ENDEE_URL)

# Verification: Ensure Endee is running before starting the app
if not client.check_health():
    print(f"CRITICAL ERROR: Could not connect to Endee at {ENDEE_URL}")
    print("Please start Endee using Docker: docker run -p 8080:8080 endeeio/endee-server:latest")

# --- AI Model Initialization ---
print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory session history for RAG personalization
search_history = []

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "Smart Food App Backend is running",
        "endpoints": ["/search", "/recommend", "/suggestions", "/add-data", "/history"]
    })

def init_db():
    """ Initial seed of the Endee vector database. """
    print("Initializing Endee collection and seeding food data...")
    try:
        # 1. Create collection if not exists
        try:
            client.create_collection(COLLECTION_NAME, dimension=384)
            print("Created new Endee collection.")
        except:
            print("Collection already exists, checking data...")

        # 2. Check if collection is empty
        res = client.list_collections()
        collections = res if isinstance(res, list) else res.get('indexes', [])
        target_index = next((idx for idx in collections if idx.get('name') == COLLECTION_NAME), None)
        
        if target_index and target_index.get('total_elements', 0) == 0:
            print("Collection is empty. Seeding data...")
            data_path = os.path.join(os.path.dirname(__file__), '../data/sample_data.json')
            with open(data_path, 'r') as f:
                food_items = json.load(f)
            
            ids = [item['id'] for item in food_items]
            # Include tags and description in text embedding for keyword synergy
            texts = [f"{item['name']}: {item['description']} ({', '.join(item.get('tags', []))}) cuisine: {item['cuisine']} rating: {item['rating']}" for item in food_items]
            embeddings = model.encode(texts)
            
            client.add_vectors(COLLECTION_NAME, ids, embeddings.tolist(), metadata=food_items)
            print("Success: Endee collection seeded with expanded 30-item dataset.")
            print("Data successfully stored in Endee")
        else:
            print(f"Collection already contains {target_index.get('total_elements', 0) if target_index else 0} elements. Skipping seed.")
    
        # Verification search
        dummy_vector = np.random.rand(384).tolist()
        test_res = client.search(COLLECTION_NAME, dummy_vector, limit=1)
        if test_res.get('results'):
            first_item = test_res['results'][0]
            meta = json.loads(first_item.get('meta', '{}'))
            print(f"VERIFIED: Endee integration is active. Sample result: {meta.get('name', 'Unknown')}")
        else:
            print("WARNING: Endee collection appears empty or not responding to search.")
    except Exception as e:
        print(f"VERIFICATION FAILED: Could not query Endee: {e}")

@app.route('/test-search', methods=['GET'])
def test_search():
    """ Internal test route to verify Endee similarity search. """
    test_query = "something spicy and authentic"
    vector = model.encode([test_query])[0]
    res = client.search(COLLECTION_NAME, vector, limit=3)
    results = res.get('results', [])
    for item in results:
        try:
            item['payload'] = json.loads(item.get('meta', '{}'))
        except:
            item['payload'] = {}
            
    return jsonify({
        "test_query": test_query,
        "results": results,
        "status": "success" if results else "empty"
    })

@app.route('/add-data', methods=['POST'])
def add_data():
    """ Endpoint to add new food items. """
    item = request.json
    text = f"{item['name']}: {item['description']} ({item['cuisine']}) Tags: {', '.join(item.get('tags', []))}"
    embedding = model.encode([text])[0]
    
    res = client.add_vectors(COLLECTION_NAME, [item['id']], [embedding], metadata=[item])
    return jsonify(res)

@app.route('/search', methods=['GET'])
def search():
    """ Enhanced Semantic Search with Tag Boosting and Personalization. """
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    food_type = request.args.get('type')
    
    if not query:
        return jsonify({"results": []})
    
    query_lower = query.lower()
    search_history.append(query)
    if len(search_history) > 5: search_history.pop(0)
    
    # 1. Detect Vague Queries
    vague_keywords = ["food", "dish", "something", "tasty", "delicious", "eat", "good", "hungry", "anything", "meal"]
    is_vague = any(word == query_lower for word in vague_keywords) or query_lower in ["something tasty", "something delicious", "anything good"]
    
    if is_vague:
        search_vector = model.encode(["popular trending delicious top-rated food"])[0]
    else:
        search_vector = model.encode([query])[0]
    

    # 2. Filter logic
    filter_query = {}
    if food_type and food_type != "":
        filter_query["type"] = food_type
        
    # 3. Endee Search
    res = client.search(COLLECTION_NAME, search_vector, limit=15, filter_query=filter_query if filter_query else None)
    results = res.get('results', [])
    
    # 4. Fallback if no results found
    if not results:
        fallback_vector = model.encode(["delicious popular food"])[0]
        res = client.search(COLLECTION_NAME, fallback_vector, limit=10)
        results = res.get('results', [])

    # 5. Hybrid Scoring and Re-ranking

    available_tags = ["spicy", "sweet", "healthy", "veg", "non-veg", "fast-food"]
    query_tags = [t for t in available_tags if t in query_lower]
    
    # Analyze cuisine preferences from history
    cuisine_counts = {}
    for h in search_history:
        for c in ["Indian", "Chinese", "Italian", "Continental", "Middle Eastern", "American"]:
            if c.lower() in h.lower():
                cuisine_counts[c] = cuisine_counts.get(c, 0) + 1
    pref_cuisine = max(cuisine_counts, key=cuisine_counts.get) if cuisine_counts else None

    processed_results = []
    for item in results:
        try:
            payload = json.loads(item.get('meta', '{}'))
        except:
            payload = {}
        
        item['payload'] = payload
        base_score = item.get('score', 0)
        
        # Calculate Boosts
        item_tags = payload.get('tags', [])
        tag_boost = 0
        for t in query_tags:
            if t in item_tags:
                tag_boost += 0.2
        
        cuisine_boost = 0.15 if pref_cuisine and payload.get('cuisine') == pref_cuisine else 0
        
        # Rating boost: higher importance for vague queries
        rating_val = payload.get('rating', 0)
        rating_boost = (rating_val / 10.0)
        if is_vague:
            rating_boost *= 1.5 # Extra weight on rating for broad queries
        
        final_score = base_score + tag_boost + cuisine_boost + rating_boost
        item['enhanced_score'] = final_score
        
        # Build Explanation
        if is_vague:
            reasoning = f"Since you're looking for '{query}', check out this trending favorite! "
            if rating_val > 4.5:
                reasoning += f"It's exceptionally rated at {rating_val}/5."
            else:
                reasoning += "It's a crowd favorite and highly recommended."
        else:
            reasoning = f"Since you're looking for '{query}', "
            matched = [t for t in query_tags if t in item_tags]
            if matched:
                reasoning += f"this perfectly matches your interest in {', '.join(matched)} flavors. "
            if cuisine_boost > 0:
                reasoning += f"It matches your preference for {pref_cuisine} food. "
            
            if rating_val > 4.5:
                reasoning += f"Plus, it's a top-rated choice at {rating_val}/5!"
            else:
                reasoning += "It's a strong semantic match for your current craving."
        
        item['explanation'] = reasoning
        processed_results.append(item)

    # Prepare final flat results as requested
    final_results = []
    for item in processed_results[:limit]:
        p = item['payload']
        # Flatten and include score/explanation
        flat_item = {
            **p,
            "match_score": item['score'],
            "enhanced_score": item['enhanced_score'],
            "explanation": item['explanation']
        }
        final_results.append(flat_item)
    
    return jsonify({"results": final_results})

@app.route('/recommend', methods=['GET'])
def recommend():
    """ Enhanced Personalized Recommendations. """
    city = request.args.get('city', 'Mumbai')
    
    context_text = " ".join(search_history) if search_history else "best popular dishes"
    context_vector = model.encode([context_text])[0]
    
    res = client.search(COLLECTION_NAME, context_vector, limit=6, filter_query={"location": city})
    items = res.get('results', [])
    processed_items = []

    for item in items:
        try:
            p = json.loads(item.get('meta', '{}'))
            item['payload'] = p
            # Boost by rating for recommendations
            item['enhanced_score'] = item.get('score', 0) + (p.get('rating', 0) / 10.0)
            processed_items.append(item)
        except:
            processed_items.append(item)
    
    final_items = []
    for r in processed_items:
        p = r.get('payload', {})
        flat_item = {
            **p,
            "match_score": r.get('score', 0),
            "enhanced_score": r.get('enhanced_score', 0)
        }
        final_items.append(flat_item)

    return jsonify({
        "message": msg,
        "items": final_items,
        "popular_near_you": recommendations[0] if recommendations else "Biryani"
    })

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": search_history})

@app.route('/suggestions', methods=['GET'])
def suggestions():
    """ Autocomplete Suggestions. """
    q = request.args.get('q', '').lower()
    if not q:
        return jsonify({"suggestions": []})
    
    data_path = os.path.join(os.path.dirname(__file__), '../data/sample_data.json')
    try:
        with open(data_path, 'r') as f:
            food_items = json.load(f)
        name_matches = [item['name'] for item in food_items if q in item['name'].lower()]
    except:
        name_matches = []
    
    history_matches = [h for h in search_history if q in h.lower()]
    all_suggestions = list(set(name_matches + history_matches))
    all_suggestions.sort(key=lambda x: x.lower().find(q))
    
    return jsonify({"suggestions": all_suggestions[:8]})

if __name__ == '__main__':
    init_db()
    app.run(port=5000, debug=True)
