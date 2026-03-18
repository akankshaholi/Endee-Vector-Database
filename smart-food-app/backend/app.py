from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from flasgger import Swagger
import json
import os
import sys
import numpy as np
from endee_client import EndeeClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Swagger UI Configuration ---
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs"
}
swagger_template = {
    "info": {
        "title": "🍽️ Smart Food App API",
        "description": "Semantic food search and recommendations powered by Endee vector DB.",
        "version": "1.0"
    }
}
swagger = Swagger(app, config=swagger_config, template=swagger_template)

# --- Endee Integration Configuration ---
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "endee_token")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "food_items")

client = EndeeClient(base_url=ENDEE_URL, auth_token=ENDEE_TOKEN)

# Verification: Ensure Endee is running before starting the app
if not client.check_health():
    print(f"CRITICAL ERROR: Could not connect to Endee at {ENDEE_URL}")
    print("Please start Endee using Docker: docker run -p 8080:8080 endeeio/endee-server:latest")

# --- AI Model Initialization ---
print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory session history for RAG personalization
search_history = []

# --- Diet Type Detection ---
VEG_KEYWORDS = ["veg", "vegetarian", "veggie", "paneer", "tofu", "aloo", "palak", "dal", "sabzi", "chole", "rajma"]
NON_VEG_KEYWORDS = ["non veg", "nonveg", "non-veg", "chicken", "mutton", "lamb", "fish", "egg", "meat", "prawn", "shrimp", "beef", "pork", "keema", "tandoori chicken", "butter chicken"]

def detect_diet_type(query, explicit_type=None):
    """Detect veg/non-veg intent from query text or explicit param."""
    if explicit_type and explicit_type in ["veg", "non-veg"]:
        return explicit_type
    q = query.lower()
    # Check non-veg first ("non veg" contains "veg", so order matters)
    for kw in NON_VEG_KEYWORDS:
        if kw in q:
            return "non-veg"
    for kw in VEG_KEYWORDS:
        if kw in q:
            return "veg"
    return None

@app.route('/')
def home():
    """
    Health check
    ---
    tags:
      - General
    responses:
      200:
        description: Backend status
    """
    return jsonify({
        "status": "online",
        "message": "Smart Food App Backend is running",
        "endpoints": ["/search", "/recommend", "/suggestions", "/add-data", "/history", "/apidocs"]
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
    """
    Add a new food item to the vector database
    ---
    tags:
      - Data
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - id
            - name
            - description
            - cuisine
          properties:
            id:
              type: string
              example: "99"
            name:
              type: string
              example: "Test Burger"
            description:
              type: string
              example: "Juicy beef burger with lettuce and tomato"
            cuisine:
              type: string
              example: "American"
            tags:
              type: array
              items:
                type: string
              example: ["fast-food", "non-veg"]
            rating:
              type: number
              example: 4.2
            type:
              type: string
              example: "non-veg"
            price:
              type: string
              example: "medium"
    responses:
      200:
        description: Item added successfully
    """
    item = request.json
    text = f"{item['name']}: {item['description']} ({item['cuisine']}) Tags: {', '.join(item.get('tags', []))}"
    embedding = model.encode([text])[0]
    
    res = client.add_vectors(COLLECTION_NAME, [item['id']], [embedding], metadata=[item])
    return jsonify(res)

@app.route('/search', methods=['GET'])
def search():
    """
    Semantic food search
    ---
    tags:
      - Search
    parameters:
      - name: q
        in: query
        type: string
        required: true
        description: Search query (e.g. pizza, spicy, healthy, veg food)
        example: pizza
      - name: limit
        in: query
        type: integer
        required: false
        default: 10
        description: Max number of results
      - name: type
        in: query
        type: string
        required: false
        description: Filter by type (veg / non-veg)
        enum: [veg, non-veg]
    responses:
      200:
        description: List of matching food items with scores and explanations
    """
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    food_type = request.args.get('type')
    
    if not query:
        return jsonify({"results": []})
    
    query_lower = query.lower()
    search_history.append(query)
    if len(search_history) > 5: search_history.pop(0)
    
    # 1. Detect diet type from query text + explicit param
    detected_type = detect_diet_type(query, food_type)
    print(f"[FILTER DEBUG] Query: '{query}' | Explicit type: {food_type} | Detected diet: {detected_type}")
    
    # 2. Detect Vague Queries
    vague_keywords = ["food", "dish", "something", "tasty", "delicious", "eat", "good", "hungry", "anything", "meal"]
    is_vague = any(word == query_lower for word in vague_keywords) or query_lower in ["something tasty", "something delicious", "anything good", "veg food", "non-veg food", "veg", "vegetarian"]
    
    if is_vague:
        search_vector = model.encode(["popular trending delicious top-rated food"])[0]
    else:
        search_vector = model.encode([query])[0]
    
    # 3. Endee Search — fetch extra results so we have enough after filtering
    fetch_limit = 25 if detected_type else 15
    res = client.search(COLLECTION_NAME, search_vector, limit=fetch_limit)
    results = res.get('results', [])
    
    # 4. POST-RETRIEVAL FILTERING (CRITICAL FIX)
    # NEVER return non-veg when user asks for veg, and vice versa
    if detected_type:
        filtered = []
        for item in results:
            try:
                payload = json.loads(item.get('meta', '{}'))
            except:
                payload = {}
            if payload.get('type') == detected_type:
                filtered.append(item)
        print(f"[FILTER DEBUG] Before filter: {len(results)} | After filter ({detected_type}): {len(filtered)}")
        results = filtered
    
    # 5. Fallback if no results found — respect diet type
    if not results:
        if detected_type == "veg":
            fallback_vector = model.encode(["popular vegetarian paneer healthy food"])[0]
        elif detected_type == "non-veg":
            fallback_vector = model.encode(["popular chicken biryani meat food"])[0]
        else:
            fallback_vector = model.encode(["delicious popular food"])[0]
        res = client.search(COLLECTION_NAME, fallback_vector, limit=15)
        results = res.get('results', [])
        # Re-apply filter for fallback too
        if detected_type:
            results = [item for item in results if json.loads(item.get('meta', '{}')).get('type') == detected_type]
        print(f"[FILTER DEBUG] Fallback results ({detected_type}): {len(results)}")

    # 6. Hybrid Scoring and Re-ranking
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
        
        # Diet match boost
        if detected_type and payload.get('type') == detected_type:
            tag_boost += 0.3
        
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

    # Sort by enhanced_score descending
    processed_results.sort(key=lambda x: x['enhanced_score'], reverse=True)

    # Prepare final flat results
    final_results = []
    for item in processed_results[:limit]:
        p = item['payload']
        flat_item = {
            **p,
            "match_score": item['score'],
            "enhanced_score": item['enhanced_score'],
            "explanation": item['explanation']
        }
        final_results.append(flat_item)
    
    return jsonify({"results": final_results, "detected_type": detected_type})

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Personalized food recommendations
    ---
    tags:
      - Recommendations
    parameters:
      - name: city
        in: query
        type: string
        required: false
        default: Mumbai
        description: City for context (used in response metadata)
      - name: type
        in: query
        type: string
        required: false
        description: Filter recommendations by diet type
        enum: [veg, non-veg]
    responses:
      200:
        description: Recommended food items based on your search history
    """
    city = request.args.get('city', 'Mumbai')
    food_type = request.args.get('type')
    
    # Build context vector from search history for personalization
    context_text = " ".join(search_history) if search_history else "best popular top-rated dishes"
    context_vector = model.encode([context_text])[0]
    
    # Fetch extra results if filtering will be applied
    fetch_limit = 15 if food_type else 6
    res = client.search(COLLECTION_NAME, context_vector, limit=fetch_limit)
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
    
    # POST-RETRIEVAL FILTERING for recommendations
    if food_type and food_type in ["veg", "non-veg"]:
        processed_items = [item for item in processed_items if item.get('payload', {}).get('type') == food_type]
        print(f"[FILTER DEBUG] /recommend filter ({food_type}): {len(processed_items)} results")
    
    final_items = []
    for r in processed_items[:6]:
        p = r.get('payload', {})
        flat_item = {
            **p,
            "match_score": r.get('score', 0),
            "enhanced_score": r.get('enhanced_score', 0)
        }
        final_items.append(flat_item)

    top_item_name = final_items[0].get('name', 'Biryani') if final_items else 'Biryani'
    personalized = bool(search_history)
    msg = f"Personalized recommendations based on your history" if personalized else f"Top rated food recommendations"

    return jsonify({
        "message": msg,
        "city": city,
        "items": final_items,
        "popular_near_you": top_item_name
    })

@app.route('/history', methods=['GET'])
def get_history():
    """
    Get search history
    ---
    tags:
      - General
    responses:
      200:
        description: List of recent search queries
    """
    return jsonify({"history": search_history})

@app.route('/suggestions', methods=['GET'])
def suggestions():
    """
    Autocomplete suggestions
    ---
    tags:
      - Search
    parameters:
      - name: q
        in: query
        type: string
        required: true
        description: Partial search term for autocomplete
        example: bi
    responses:
      200:
        description: List of matching food name suggestions
    """
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
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(port=port, debug=debug)
