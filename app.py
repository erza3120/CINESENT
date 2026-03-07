"""
╔══════════════════════════════════════════════════════════════════╗
║         CINESENT — Cinema Sentiment Analyzer                     ║
║         Flask Backend · TMDB API · ML Sentiment Model            ║
╚══════════════════════════════════════════════════════════════════╝

Architecture:
  ┌─────────┐     ┌──────────┐     ┌──────────────┐     ┌─────────┐
  │ Browser │────▶│  Flask   │────▶│  TMDB API    │     │  SQLite │
  │   UI    │◀────│  Routes  │     │  (movies +   │     │  Users  │
  └─────────┘     └──────────┘     │   reviews)   │     └─────────┘
                       │           └──────────────┘
                       ▼
                  ┌──────────┐
                  │ ML Model │  ← sentiment_model.pkl + vectorizer.pkl
                  │  (.pkl)  │    (loaded ONCE at startup, never retrained)
                  └──────────┘
"""

import os
import pickle
import sqlite3
import requests
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, g)
from werkzeug.security import generate_password_hash, check_password_hash

# ─────────────────────────────────────────────────────────────────
#  App Configuration
# ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "cinesent_secret_2024_change_in_production"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATABASE    = os.path.join(BASE_DIR, "database.db")
TMDB_KEY    = "f8081478f446d417bf51c96c116e2109"
TMDB_BASE   = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# ─────────────────────────────────────────────────────────────────
#  ML Model Loading (done ONCE at startup — never retrained)
# ─────────────────────────────────────────────────────────────────
print("\n⏳  Loading sentiment model...")
with open(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"), "rb") as f:
    sentiment_model = pickle.load(f)

print("⏳  Loading TF-IDF vectorizer...")
with open(os.path.join(BASE_DIR, "model", "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

print("✅  ML model ready.\n")


# ─────────────────────────────────────────────────────────────────
#  Database Helpers
# ─────────────────────────────────────────────────────────────────
def get_db():
    """
    Open a SQLite connection and store it on Flask's 'g' object
    so the same connection is reused within a single request.
    """
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row   # allows dict-like access: row["col"]
    return g.db


@app.teardown_appcontext
def close_db(error):
    """Automatically close the DB connection after each request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """
    Create the users table if it doesn't already exist.
    Called once when the app starts.
    """
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()
    print("✅  Database initialised.")


# ─────────────────────────────────────────────────────────────────
#  Auth Decorator
# ─────────────────────────────────────────────────────────────────
def login_required(f):
    """
    Decorator that redirects unauthenticated users to /login.
    Wrap any route that needs authentication with @login_required.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────
#  Sentiment Analysis Helper
# ─────────────────────────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """
    Run the pre-trained model on a single review text.

    Steps:
      1. Vectorize the text using the pre-fitted TF-IDF vectorizer
      2. Predict the sentiment class (positive / negative)
      3. Get class probabilities for the confidence score

    Returns a dict:
      {
        "sentiment":   "Positive" | "Negative",
        "label_class": "positive" | "negative",
        "confidence":  float (0–100)
      }
    """
    # Transform raw text into a TF-IDF feature vector
    X = vectorizer.transform([text])

    # Predict the label
    raw_label: str = sentiment_model.predict(X)[0]

    # Get the highest probability across both classes
    proba      = sentiment_model.predict_proba(X)[0]
    confidence = round(float(max(proba)) * 100, 1)

    return {
        "sentiment":   raw_label.strip().capitalize(),
        "label_class": raw_label.lower(),
        "confidence":  confidence,
    }


# ─────────────────────────────────────────────────────────────────
#  TMDB API Helpers
# ─────────────────────────────────────────────────────────────────
def tmdb_get(endpoint: str, extra_params: dict = None) -> dict | None:
    """
    Centralised TMDB API caller.
    Adds the API key automatically; returns the JSON response or None on error.
    """
    params = {"api_key": TMDB_KEY}
    if extra_params:
        params.update(extra_params)
    try:
        r = requests.get(f"{TMDB_BASE}{endpoint}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[TMDB ERROR] {endpoint}: {e}")
        return None


def fetch_trending_movies(count: int = 12) -> list[dict]:
    """
    Fetch today's trending movies from TMDB for the login page preview.
    Returns up to *count* movies with poster URLs attached.
    """
    data = tmdb_get("/trending/movie/day")
    if not data:
        return []
    movies = data.get("results", [])[:count]
    for m in movies:
        m["poster_url"] = (
            f"{POSTER_BASE}{m['poster_path']}"
            if m.get("poster_path")
            else "/static/img/no_poster.png"
        )
    return movies


def search_movies(query: str, page: int = 1) -> list[dict]:
    """
    Search TMDB for movies matching *query*.
    Returns list of movie dicts with poster URLs.
    """
    data = tmdb_get("/search/movie", {"query": query, "page": page})
    if not data:
        return []
    movies = data.get("results", [])
    for m in movies:
        m["poster_url"] = (
            f"{POSTER_BASE}{m['poster_path']}"
            if m.get("poster_path")
            else "/static/img/no_poster.png"
        )
    return movies


def fetch_popular_movies(page: int = 1) -> list[dict]:
    """
    Fetch the current page of popular movies from TMDB.
    Adds the full poster URL to each movie dict.
    """
    data = tmdb_get("/movie/popular", {"page": page})
    if not data:
        return []
    movies = data.get("results", [])
    for m in movies:
        m["poster_url"] = (
            f"{POSTER_BASE}{m['poster_path']}"
            if m.get("poster_path")
            else "/static/img/no_poster.png"
        )
    return movies


def fetch_movie_details(movie_id: int) -> dict | None:
    """
    Fetch full details for a single movie by its TMDB ID.
    Also attaches the full poster URL.
    """
    data = tmdb_get(f"/movie/{movie_id}")
    if data and data.get("poster_path"):
        data["poster_url"] = f"{POSTER_BASE}{data['poster_path']}"
    else:
        if data:
            data["poster_url"] = "/static/img/no_poster.png"
    return data


def fetch_movie_reviews(movie_id: int) -> list[dict]:
    """
    Fetch reviews for a movie from TMDB, then run every review
    through the ML model to attach a sentiment result.

    Returns a list of dicts with: author, content, sentiment,
    label_class, confidence, content_preview (first 400 chars).
    """
    data = tmdb_get(f"/movie/{movie_id}/reviews")
    if not data:
        return []

    reviews = []
    for r in data.get("results", []):
        content = r.get("content", "").strip()
        if not content:
            continue

        # Run sentiment analysis on the review text
        sentiment_result = analyze_sentiment(content)

        reviews.append({
            "author":          r.get("author", "Anonymous"),
            "content":         content,
            "content_preview": content[:420] + ("…" if len(content) > 420 else ""),
            "sentiment":       sentiment_result["sentiment"],
            "label_class":     sentiment_result["label_class"],
            "confidence":      sentiment_result["confidence"],
        })
    return reviews


# ─────────────────────────────────────────────────────────────────
#  Routes — Authentication
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Root route: redirect to dashboard if logged in, else to login."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """
    GET  → render the registration form
    POST → validate input, hash password, insert into DB, redirect to login
    """
    # If already logged in, skip straight to dashboard
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        # ── Input validation ────────────────────────────────────────
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        # ── Insert user into DB ─────────────────────────────────────
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, generate_password_hash(password))
            )
            db.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already registered.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    GET  → render the login form
    POST → verify credentials, create session, redirect to dashboard
    """
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        db   = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            # Store minimal user info in the session
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    # Fetch trending movies to display on the login page
    trending = fetch_trending_movies(count=12)
    return render_template("login.html", trending=trending)


@app.route("/logout")
def logout():
    """Clear the session and redirect to login."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/search")
@login_required
def search():
    """
    Search movies via TMDB.
    Accepts ?q=query and returns results page.
    If no query, redirects back to dashboard.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return redirect(url_for("dashboard"))

    results = search_movies(query)
    return render_template(
        "dashboard.html",
        hero=results[0] if results else None,
        movies=results[1:] if results else [],
        page=1,
        username=session.get("username"),
        search_query=query,
        search_total=len(results),
    )


# ─────────────────────────────────────────────────────────────────
#  Routes — Dashboard
# ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    """
    Main movie discovery page.
    Fetches popular movies from TMDB (up to 3 pages = ~60 movies)
    and renders them in a Netflix-style grid.
    """
    page   = request.args.get("page", 1, type=int)
    movies = fetch_popular_movies(page=page)

    # Separate the first movie as a featured hero banner
    hero   = movies[0] if movies else None
    rest   = movies[1:] if movies else []

    return render_template(
        "dashboard.html",
        hero=hero,
        movies=rest,
        page=page,
        username=session.get("username"),
    )


# ─────────────────────────────────────────────────────────────────
#  Routes — Movie Detail + Sentiment Analysis
# ─────────────────────────────────────────────────────────────────

@app.route("/movie/<int:movie_id>")
@login_required
def movie_detail(movie_id: int):
    """
    Movie detail page:
      1. Fetch full movie details from TMDB
      2. Fetch all reviews for that movie
      3. Run each review through the ML sentiment model
      4. Calculate summary stats (% positive, % negative)
      5. Render the movie detail template
    """
    movie   = fetch_movie_details(movie_id)
    if not movie:
        flash("Movie not found.", "error")
        return redirect(url_for("dashboard"))

    reviews = fetch_movie_reviews(movie_id)

    # ── Sentiment summary statistics ────────────────────────────
    total     = len(reviews)
    pos_count = sum(1 for r in reviews if r["label_class"] == "positive")
    neg_count = total - pos_count
    pos_pct   = round((pos_count / total * 100), 1) if total else 0
    neg_pct   = round((neg_count / total * 100), 1) if total else 0

    # Average confidence across all reviews
    avg_conf  = (
        round(sum(r["confidence"] for r in reviews) / total, 1)
        if total else 0
    )

    sentiment_summary = {
        "total":     total,
        "positive":  pos_count,
        "negative":  neg_count,
        "pos_pct":   pos_pct,
        "neg_pct":   neg_pct,
        "avg_conf":  avg_conf,
        "overall":   "Positive" if pos_pct >= 50 else "Negative",
    }

    return render_template(
        "movie.html",
        movie=movie,
        reviews=reviews,
        summary=sentiment_summary,
        username=session.get("username"),
    )


# ─────────────────────────────────────────────────────────────────
#  Routes — API (JSON endpoints for optional AJAX use)
# ─────────────────────────────────────────────────────────────────

@app.route("/api/movies")
@login_required
def api_movies():
    """Return popular movies as JSON (for potential AJAX pagination)."""
    page   = request.args.get("page", 1, type=int)
    movies = fetch_popular_movies(page=page)
    return jsonify({"movies": movies, "page": page})


@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    """
    Accepts JSON: { "text": "review text here" }
    Returns:      { "sentiment": ..., "confidence": ..., "label_class": ... }
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    result = analyze_sentiment(data["text"])
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    # Read PORT from environment (Render sets this automatically)
    # Falls back to 5000 for local development
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    print(f"\n🎬  CINESENT running on port {port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
