import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = "http://localhost:5000";

const App = () => {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [results, setResults] = useState([]);
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    fetchRecommendations();
  }, [filter]);

  // Debounced search for suggestions
  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      if (query.length > 1) {
        fetchSuggestions();
      } else {
        setSuggestions([]);
      }
    }, 300);

    return () => clearTimeout(delayDebounceFn);
  }, [query]);

  const fetchRecommendations = async () => {
    try {
      const res = await axios.get(`${API_BASE}/recommend?city=Mumbai`);
      setRecommendation(res.data);
      if (!query) setResults(res.data.items);
    } catch (err) {
      console.error("Backend unreachable.");
    }
  };

  const fetchSuggestions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/suggestions?q=${query}`);
      setSuggestions(res.data.suggestions);
    } catch (err) {
      console.error(err);
    }
  };

  const handleSearch = async (e, forcedQuery = null) => {
    if (e) e.preventDefault();
    const finalQuery = forcedQuery !== null ? forcedQuery : query;
    if (!finalQuery && !filter) return;
    
    setLoading(true);
    setShowSuggestions(false);
    try {
      const res = await axios.get(`${API_BASE}/search?q=${finalQuery || 'food'}&type=${filter}`);
      setResults(res.data.results);
      fetchRecommendations();
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const highlightMatch = (text, term) => {
    if (!term) return text;
    const parts = text.split(new RegExp(`(${term})`, 'gi'));
    return parts.map((part, i) => 
      part.toLowerCase() === term.toLowerCase() ? <strong key={i}>{part}</strong> : part
    );
  };

  // Quick categories for trending section
  const trendingItems = [
    { name: "Best Biryani", img: "https://images.unsplash.com/photo-1563379091339-03b21bc4a4f8?w=300", query: "spicy biryani" },
    { name: "Healthy Bowls", img: "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=300", query: "healthy salad quinoa" },
    { name: "Coffee Breaks", img: "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=300", query: "breakfast coffee" },
    { name: "Street Food", img: "https://images.unsplash.com/photo-1567337710282-00832b415979?w=300", query: "street food chat pav" },
    { name: "Desserts", img: "https://images.unsplash.com/photo-1563729784474-d77dbb933a9e?w=300", query: "sweet dessert chocolate" },
  ];

  return (
    <div className="app">
      <nav className="navbar">
        <div className="container nav-content">
          <div className="logo" onClick={() => window.location.reload()}>SmartFood.ai</div>
          <div className="nav-links" style={{display: 'flex', gap: '25px', fontWeight: '500'}}>
             <span>Discover</span>
             <span>Restaurants</span>
             <span style={{color: 'var(--primary)'}}>Sign In</span>
          </div>
        </div>
      </nav>

      <section className="hero">
        <div className="container">
          <h1>Find the perfect meal <br/><span style={{color: 'var(--primary)'}}>with AI precision.</span></h1>
          <p>Powered by Endee Vector DB for semantic flavor search.</p>
          
          <div className="search-wrapper">
            <form onSubmit={handleSearch}>
              <input 
                type="text" 
                className="search-bar" 
                placeholder="Try 'something spicy with rice' or 'light healthy breakfast'..."
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setShowSuggestions(true);
                }}
                onFocus={() => setShowSuggestions(true)}
              />
            </form>
            
            {showSuggestions && suggestions.length > 0 && (
              <div className="suggestions-dropdown">
                {suggestions.map((s, i) => (
                  <div 
                    key={i} 
                    className="suggestion-item"
                    onClick={() => {
                      setQuery(s);
                      handleSearch(null, s);
                    }}
                  >
                    <span>🔍</span> {highlightMatch(s, query)}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>

      <main className="container">
        {recommendation && (
          <div className="notification-banner" style={{
            background: 'linear-gradient(90deg, #fff0f3, #ffe5e9)',
            border: '1px solid var(--primary)',
            color: 'var(--primary-dark)',
            padding: '15px 20px',
            borderRadius: '15px',
            marginBottom: '30px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <span style={{fontSize: '1.2rem'}}>✨</span> <strong>AI Insight:</strong> {recommendation.message}
          </div>
        )}

        <div className="section-header">
          <h2>Trending Flavors</h2>
        </div>
        <div className="trending-container">
          {trendingItems.map((cat, i) => (
            <div key={i} className="trending-card" onClick={() => {setQuery(cat.query); handleSearch();}}>
              <img src={cat.img} alt={cat.name} />
              <div className="overlay">
                <h4>{cat.name}</h4>
              </div>
            </div>
          ))}
        </div>

        <div className="section-header">
          <h2>Semantic Results</h2>
          <div className="filters-bar" style={{margin: 0}}>
            {["", "veg", "non-veg"].map((t) => (
              <div 
                key={t}
                className={`filter-chip ${filter === t ? 'active' : ''}`}
                onClick={() => setFilter(t)}
              >
                {t === "" ? "All" : t === "veg" ? "Vegetarian" : "Non-Veg"}
              </div>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="loader-container">
            <div className="spinner"></div>
            <p style={{marginTop: '15px', color: 'var(--text-muted)'}}>Consulting the flavor vectors...</p>
          </div>
        ) : (
          <>
            {results.length > 0 ? (
              <div className="food-grid">
                {results.map((item, idx) => (
                  <FoodCard key={item.id} item={item} score={item.match_score} index={idx} />
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <img src="https://cdni.iconscout.com/illustration/premium/thumb/empty-cart-2130356-1800917.png" alt="Empty" />
                <h3>Oops! We couldn't find that specific flavor.</h3>
                <p>Try broad terms like 'spicy', 'healthy', or 'street food'.</p>
              </div>
            )}
          </>
        )}
      </main>
      
      <footer style={{padding: '60px 0', textAlign: 'center', borderTop: '1px solid #eee', marginTop: '40px', color: 'var(--text-muted)'}}>
        <p>© 2026 SmartFood.ai • Powered by Endee Vector Database</p>
      </footer>
    </div>
  );
};

const FoodCard = ({ item, score, index }) => {
  // Use backend image directly with a smart fallback
  const foodImage = item.image_url || `https://source.unsplash.com/400x300/?${encodeURIComponent(item.name)}`;

  return (
    <div className="food-card" style={{animationDelay: `${index * 0.1}s`}}>
      <div className="food-img-wrapper">
        <img src={foodImage} alt={item.name} className="food-img" onError={(e) => {
          e.target.src = `https://source.unsplash.com/400x300/?food,${encodeURIComponent(item.name)}`;
        }} />
        <div style={{
          position: 'absolute', 
          top: '15px', 
          right: '15px', 
          background: 'rgba(255,255,255,0.9)', 
          padding: '5px 10px', 
          borderRadius: '10px',
          fontSize: '0.8rem',
          fontWeight: '700',
          boxShadow: '0 4px 10px rgba(0,0,0,0.1)'
        }}>
          {score ? `${(score*100).toFixed(0)}% Match` : item.cuisine}
        </div>
      </div>
      <div className="food-info">
        <div className="food-header">
          <span className="food-name">{item.name}</span>
          <span className="badge-rating">★ {item.rating}</span>
        </div>
        <p style={{color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.5', height: '45px', overflow: 'hidden'}}>
          {item.description}
        </p>
        
        {/* RAG Explanation Section */}
        {score && (
          <div style={{
            background: '#fdf0f2',
            padding: '10px',
            borderRadius: '10px',
            marginTop: '15px',
            fontSize: '0.85rem',
            borderLeft: '3px solid var(--primary)',
            color: '#444',
            fontStyle: 'italic'
          }}>
            "{(item.explanation || `Matches your search for "${item.name}" with a ${(score*100).toFixed(0)}% semantic score.`)}"
          </div>
        )}

        <div style={{marginTop: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <span style={{fontSize: '0.85rem', fontWeight: '600', color: '#888'}}>
             📍 {item.location}
          </span>
          <button style={{
            background: 'var(--primary)',
            color: 'white',
            border: 'none',
            padding: '8px 18px',
            borderRadius: '25px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'background 0.3s'
          }}>Order Now</button>
        </div>
      </div>
    </div>
  );
};

export default App;
