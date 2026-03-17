# Smart Food Recommendation App

A web app that lets you search for food using natural language and get personalized recommendations. Built as part of a hands-on project to learn how vector databases work in real applications.

---

## What it does

You type something like "spicy street food" or "light healthy breakfast" and the app finds the closest matching dishes from its database — not just by keyword, but by meaning. It also explains why each result was recommended.

---

## Problem Statement

Most food apps use keyword filters and categories. If you type "something warm and filling", they return nothing useful. This project explores how semantic search can make that kind of query actually work, by converting text into vector embeddings and finding similar items in a vector database (Endee).

---

## Features

- Semantic search powered by sentence-transformers and Endee vector database
- Personalized food recommendations based on your search history
- Each result includes an AI-generated explanation of why it matched
- Handles vague queries like "something tasty" by falling back to trending results
- 30 food items with images, ratings, cuisines, and tags
- Clean React frontend with autocomplete suggestions and category filters

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Vector Database | Endee (self-hosted via Docker) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Frontend | React + Vite |
| Data | JSON (30 hand-curated food items) |

---

## How Endee is Used

Endee is an open-source vector database. Here is how it fits into this project:

1. **Storing data** — When the backend starts, it converts each food item's name, description, and tags into a 384-dimensional vector and stores it in Endee along with the full metadata (name, image, rating, etc.).

2. **Searching** — When a user submits a query, the app converts that query into a vector using the same model, then asks Endee to return the most similar food items.

3. **Retrieving results** — Endee returns the closest matches with similarity scores. The backend then re-ranks them using tag matching, user history, and ratings before sending the final list to the frontend.

---

## Project Structure

```
smart-food-app/
├── backend/
│   ├── app.py              # Flask API server
│   ├── endee_client.py     # Endee HTTP client wrapper
│   ├── reseed_db.py        # Script to populate Endee with food data
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   └── main.jsx
│   └── package.json
├── data/
│   └── sample_data.json    # 30 food items with metadata
└── README.md
```

---

## How to Run

### Step 1 — Start Endee (Vector Database)

Endee runs as a Docker container. Make sure Docker is installed and running, then:

```bash
docker run -p 8080:8080 endeeio/endee-server:latest
```

Wait a few seconds until you see the server start message. Endee should now be accessible at `http://localhost:8080`.

### Step 2 — Set up and run the Backend

Open a terminal and navigate to the backend folder:

```bash
cd smart-food-app/backend
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Seed the database with food data (run this once):

```bash
python reseed_db.py
```

You should see: `Data successfully stored in Endee`

Start the Flask server:

```bash
python app.py
```

The backend will be running at `http://localhost:5000`.

### Step 3 — Run the Frontend

Open a new terminal and navigate to the frontend folder:

```bash
cd smart-food-app/frontend
```

Install Node dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The app will open at `http://localhost:5173`.

---

## Example Usage

- Search **"spicy food"** → returns Hyderabadi Biryani, Misal Pav, Szechuan Noodles, etc.
- Search **"healthy breakfast"** → returns Masala Dosa, Avocado Toast, Quinoa Salad, etc.
- Search **"street food"** → returns Pav Bhaji, Chicken Shawarma, Falafel Wrap, etc.
- Search **"something sweet"** → falls back to trending, returns top-rated desserts

Each result shows the dish name, image, rating, location, and a short explanation of why it matched your query.

---

## Future Improvements

- Add user login so recommendations can be personalized per user over time
- Let users add their own food items through the UI
- Use a larger, more diverse dataset
- Add filters for price range and dietary restrictions
- Improve the recommendation model with collaborative filtering

---

## Notes

- Make sure Endee is running before you start the backend, otherwise the app will print a warning and search will not work.
- The `reseed_db.py` script clears and repopulates the Endee collection every time it runs. Only run it once, or when you want to reset the data.
- The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically on first run. This may take a minute.
