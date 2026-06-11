# 🎬 IMDb 2024 Movie Recommendation System

> A content-based movie recommendation engine powered by **TF-IDF** and **Cosine Similarity** — built with Python, Streamlit, and NLP techniques on a dataset of **5,099 IMDb 2024 movies**.

---

## 📌 Project Overview

This project collects movie data from IMDb for the year 2024, processes the storylines using Natural Language Processing, and recommends the **Top N most similar movies** based on a user-provided plot description. The entire system is wrapped in an interactive **Streamlit web application** with an adaptive light/dark UI, analytics dashboards, and a browse interface.

---

## 🖥️ App Preview

| Tab | Description |
|-----|-------------|
| 🔍 **Recommend Movies** | Input any storyline → get Top N similar 2024 movies with similarity scores and bar chart |
| 📊 **Dataset Overview** | KPI metrics, storyline length histogram, category breakdown, box plot, raw data preview |
| 🗂️ **Browse Movies** | Search and explore all 5,099 movies by title or keyword |
| 📈 **NLP Analytics** | Word cloud, top 20 frequent words, TF-IDF model summary, cosine similarity heatmap |

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|----------|------------------|
| Language | Python 3.x |
| NLP | NLTK, Scikit-learn (TF-IDF Vectorizer) |
| Recommendation | Cosine Similarity |
| Web Framework | Streamlit |
| Data Manipulation | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn, WordCloud |
| Environment | VS Code / Any Python Environment |

---

## 🗂️ Project Structure

```
imdb-movie-recommendation/
│
├── preprocess.py            # Step 1 — NLP text cleaning pipeline
├── recommender.py           # Step 2 — TF-IDF model + cosine similarity logic
├── app.py                   # Step 3 — Streamlit multi-tab application
│
├── imdb_movies_raw.csv      # Input  → 2 columns (Movie Name, Storyline)
├── imdb_movies_2024.csv     # Output → 3 columns (adds Cleaned_Storyline)
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/imdb-movie-recommendation.git
cd imdb-movie-recommendation
```

### 2. Install Dependencies
```bash
pip install streamlit scikit-learn pandas numpy nltk matplotlib seaborn plotly wordcloud
```

### 3. Run Preprocessing (Step 1)
```bash
python preprocess.py
```
This reads `imdb_movies_raw.csv` (2 columns) and generates `imdb_movies_2024.csv` (3 columns).

### 4. Run the Streamlit App (Step 2)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 🧹 Preprocessing Pipeline (`preprocess.py`)

The `Cleaned_Storyline` column is generated from the raw `Storyline` column through 6 sequential steps:

```
Raw Storyline Text
      ↓  Step 1 — Lowercase conversion
      ↓  Step 2 — Remove punctuation
      ↓  Step 3 — Remove numbers
      ↓  Step 4 — Remove extra whitespace
      ↓  Step 5 — Remove stopwords  (NLTK English stopwords)
      ↓  Step 6 — Remove short tokens  (length ≤ 1)
Cleaned Storyline Text
```

**Before:** `A fading celebrity takes a black-market drug: a cell-replicating substance that helps her create a younger, better version of herself.`

**After:** `fading celebrity takes blackmarket drug cellreplicating substance helps create younger better version`

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| `Movie Name` | Title of the 2024 IMDb movie |
| `Storyline` | Original raw plot summary |
| `Cleaned_Storyline` | Pre-processed text (output of `preprocess.py`) |

- **Source:** [IMDb 2024 Feature Films](https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31)
- **Total Records:** 5,099 movies
- **Format:** CSV

---

## 🧠 How the Recommendation Works

```
User Input (Raw Storyline)
         │
         ▼
  Text Cleaning (preprocess pipeline)
         │
         ▼
  TF-IDF Vectorization
  (5,000 features · unigrams + bigrams)
         │
         ▼
  Cosine Similarity
  (input vector vs all 5,099 movie vectors)
         │
         ▼
  Ranked Top N Results
  (sorted by similarity score descending)
```

**TF-IDF** converts each storyline into a numerical vector. Words frequent in one movie but rare across all movies receive higher scores — capturing what makes each storyline unique.

**Cosine Similarity** measures the angle between two vectors. Score of `1.0` = identical; `0.0` = no overlap at all.

---

## 💡 Example

**Input:**
```
A detective investigates a series of mysterious murders in a small town,
uncovering dark secrets about the community.
```

**Output (Top 5):**

| Rank | Movie Name | Similarity Score |
|------|-----------|-----------------|
| 1 | Bhimaa | 40.0% |
| 2 | He's Watching You | 37.9% |
| 3 | Vampire Zombies From Space | 37.5% |
| 4 | Deadly Culture: Species X | 32.5% |
| 5 | Touch | 32.3% |

---

## 🎨 UI Theme

The app automatically adapts between **dark mode** and **light mode** based on the browser/OS preference using CSS variables and Streamlit's theme detection — works seamlessly on Chrome, Edge, and Firefox.

---

## 📦 requirements.txt

```
streamlit
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
plotly
wordcloud
```

---

## ✅ Evaluation Checklist

- [x] GitHub repository — public
- [x] Proper `README.md` with full workflow explanation
- [x] Clean modular Python code (PEP8 compliant)
- [x] Standalone preprocessing script (`preprocess.py`)
- [x] TF-IDF + Cosine Similarity recommendation engine
- [x] Streamlit app — 4 tabs with full visualizations
- [x] Adaptive light/dark UI theme
- [x] Works on Windows / Mac / Linux (portable)
- [ ] Demo video posted on LinkedIn

---

## 👨‍💻 Author

**Vignesh R K**  
Researcher, Centre for Post Harvest Technology  
Tamil Nadu Agricultural University (TNAU), Coimbatore  
GUVI Data Science Program — Capstone Project 4

---

## 📄 References

- [IMDb 2024 Movies](https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31)
- [Streamlit Documentation](https://docs.streamlit.io/library/api-reference)
- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [NLTK Stopwords](https://www.nltk.org/)
- [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
