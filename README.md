# 🎬 CINESENT — Cinema Sentiment Analyzer

> A full-stack Flask web application that fetches real movies from TMDB and
> uses a pre-trained machine learning model to analyze review sentiment in real time.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     BROWSER (User)                       │
│  login / register → dashboard → movie detail            │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼─────────────────────────────────┐
│                   FLASK (app.py)                         │
│                                                          │
│  Routes:                                                 │
│   /login  /register  /logout   → Auth                   │
│   /dashboard                   → Movie grid             │
│   /movie/<id>                  → Detail + sentiment      │
│   /api/movies  /api/analyze    → JSON endpoints         │
└───────┬──────────────────┬─────────────────────────────-─┘
        │                  │
┌───────▼──────┐    ┌──────▼──────────────────────────────┐
│  SQLite DB   │    │         TMDB REST API                │
│  (users)     │    │  /movie/popular → poster, title...   │
│              │    │  /movie/{id}    → full details       │
│  id          │    │  /movie/{id}/reviews → raw text      │
│  username    │    └──────────────────┬──────────────────-┘
│  email       │                       │ review text
│  password_   │            ┌──────────▼──────────────────┐
│  hash        │            │     ML Model (pkl files)    │
└──────────────┘            │                             │
                            │  vectorizer.pkl             │
                            │   → TF-IDF transform        │
                            │                             │
                            │  sentiment_model.pkl        │
                            │   → Logistic Regression     │
                            │   → predict() + proba()     │
                            └─────────────────────────────┘
```

---

## Project Structure

```
CINESENT/
│
├── app.py                    ← Main Flask application (all routes + logic)
├── database.db               ← SQLite DB (auto-created on first run)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
├── model/
│   ├── sentiment_model.pkl   ← Pre-trained Logistic Regression
│   └── vectorizer.pkl        ← Pre-trained TF-IDF vectorizer
│
├── templates/
│   ├── login.html            ← Login page
│   ├── register.html         ← Registration page
│   ├── dashboard.html        ← Movie grid (Netflix-style)
│   └── movie.html            ← Movie detail + sentiment cards
│
└── static/
    └── css/
        └── style.css         ← Global styles (dark streaming theme)
```

---

## Setup & Run (Windows)

### Step 1 — Create virtual environment
```cmd
python -m venv venv
```

### Step 2 — Activate it
```cmd
venv\Scripts\activate
```
You'll see `(venv)` appear in your prompt.

### Step 3 — Install dependencies
```cmd
pip install -r requirements.txt
```

### Step 4 — Run the app
```cmd
python app.py
```

### Step 5 — Open in browser
```
http://127.0.0.1:5000
```

The database (`database.db`) is created automatically on first run.

---

## Feature Walkthrough

### Feature 1 — Authentication
- Register with username, email, password
- Passwords are hashed with `werkzeug.security.generate_password_hash`
- Sessions managed via Flask's signed cookie session
- `@login_required` decorator protects dashboard + movie routes

### Feature 2 — Movie Dashboard
- Fetches popular movies from TMDB API (paginated)
- First movie shown as a full-width hero banner
- Rest shown in responsive grid with hover animations
- Each card links to `/movie/<id>` for full analysis

### Feature 3 — Movie Detail Page
- Full movie info: poster, title, overview, rating, runtime, genres
- Blurred backdrop creates cinematic atmosphere

### Feature 4 — Sentiment Analysis (ML)
- Reviews fetched from TMDB `/movie/{id}/reviews`
- Each review passed through `vectorizer.pkl` → `sentiment_model.pkl`
- Returns: sentiment label + confidence score
- Summary bar shows % positive / % negative across all reviews
- Each review card has a colored confidence bar

### Feature 5 — UI Design
- Dark streaming platform aesthetic (Outfit + Playfair Display fonts)
- Tailwind CSS + custom CSS variables
- Animated confidence bars, hover effects, staggered card reveals

---

## API Routes

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| GET | `/` | No | Redirect to dashboard or login |
| GET/POST | `/login` | No | Login page |
| GET/POST | `/register` | No | Register page |
| GET | `/logout` | Yes | Clear session |
| GET | `/dashboard` | Yes | Movie grid |
| GET | `/movie/<id>` | Yes | Movie + reviews + sentiment |
| GET | `/api/movies` | Yes | JSON list of movies |
| POST | `/api/analyze` | Yes | JSON sentiment of a text |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11+ · Flask 3.x |
| Authentication | Werkzeug password hashing · Flask sessions |
| Database | SQLite (via Python sqlite3 module) |
| ML Model | scikit-learn Logistic Regression (pre-trained) |
| Vectorizer | TF-IDF (pre-trained, loaded from pkl) |
| Movie Data | TMDB REST API |
| Frontend | HTML5 · Tailwind CSS CDN · Vanilla JS |
| Fonts | Playfair Display (titles) · Outfit (UI) |

---

## Notes for College Presentation

1. **Model is NEVER retrained** — `sentiment_model.pkl` is loaded once at startup
2. **All reviews come from TMDB** — no user-typed reviews
3. **SQLite is zero-config** — no database server needed
4. **Single file backend** — entire app lives in `app.py` for clarity
5. **Real-time inference** — each movie page runs live ML predictions
