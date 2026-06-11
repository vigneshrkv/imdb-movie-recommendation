"""
app.py
------
IMDb 2024 Movie Recommendation System
Streamlit Multi-Tab Application
Run with: streamlit run app.py
"""

import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
from recommender import load_data, build_tfidf_matrix, get_recommendations

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IMDb 2024 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>

/* ═══════════════════════════════════════════════════════════
   CSS VARIABLES — Dark Mode (default)
   ═══════════════════════════════════════════════════════════ */
:root {
    --bg-main:        #0d0d0d;
    --bg-sidebar:     #111111;
    --bg-card:        #1a1a1a;
    --bg-card-alt:    #151515;
    --bg-browse:      #161616;
    --bg-metric:      #1a1a1a;
    --bg-tip:         #1a1a0a;
    --bg-tabbar:      #111111;
    --bg-input:       #1a1a1a;

    --text-primary:   #ffffff;
    --text-secondary: #cccccc;
    --text-muted:     #aaaaaa;
    --text-plot:      #bbbbbb;
    --text-tip:       #cccc88;

    --border-main:    #2a2a2a;
    --border-light:   #222222;
    --border-sidebar: #222222;
    --border-tip:     #3a3000;

    --accent:         #F5C518;
    --accent-hover:   #e6b800;
    --accent-text:    #000000;

    --shadow:         rgba(0, 0, 0, 0.45);
    --tab-inactive:   #888888;
}

/* ═══════════════════════════════════════════════════════════
   CSS VARIABLES — Light Mode (Edge / light browser theme)
   ═══════════════════════════════════════════════════════════ */
@media (prefers-color-scheme: light) {
    :root {
        --bg-main:        #f5f5f5;
        --bg-sidebar:     #ffffff;
        --bg-card:        #ffffff;
        --bg-card-alt:    #fafafa;
        --bg-browse:      #ffffff;
        --bg-metric:      #ffffff;
        --bg-tip:         #fffbea;
        --bg-tabbar:      #eeeeee;
        --bg-input:       #ffffff;

        --text-primary:   #111111;
        --text-secondary: #333333;
        --text-muted:     #555555;
        --text-plot:      #444444;
        --text-tip:       #7a6000;

        --border-main:    #dddddd;
        --border-light:   #e0e0e0;
        --border-sidebar: #e0e0e0;
        --border-tip:     #f0d060;

        --accent:         #c9a800;
        --accent-hover:   #a88c00;
        --accent-text:    #000000;

        --shadow:         rgba(0, 0, 0, 0.10);
        --tab-inactive:   #555555;
    }
}

/* ═══════════════════════════════════════════════════════════
   GLOBAL
   ═══════════════════════════════════════════════════════════ */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-sidebar);
}
h1, h2, h3, h4 { color: var(--accent) !important; }
p, li, span, label { color: var(--text-secondary) !important; }

/* ═══════════════════════════════════════════════════════════
   METRIC TILES
   ═══════════════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-metric);
    border: 1px solid var(--border-main);
    border-top: 3px solid var(--accent);
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px var(--shadow);
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.8rem;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.6rem;
    font-weight: 700;
}

/* ═══════════════════════════════════════════════════════════
   MOVIE RESULT CARD
   ═══════════════════════════════════════════════════════════ */
.movie-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-alt) 100%);
    border-left: 5px solid var(--accent);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px var(--shadow);
    transition: transform 0.15s;
}
.movie-card:hover { transform: translateX(4px); }
.movie-rank {
    font-size: 0.78rem; color: var(--accent); font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px;
}
.movie-title {
    font-size: 1.25rem; font-weight: 700; color: var(--text-primary);
    margin: 5px 0 10px 0;
}
.movie-plot  { font-size: 0.92rem; color: var(--text-plot); line-height: 1.65; }
.score-pill  {
    display: inline-block; background: var(--accent); color: var(--accent-text);
    font-size: 0.75rem; font-weight: 700; padding: 3px 12px;
    border-radius: 20px; margin-top: 10px;
}
.no-match    { color: var(--text-muted); font-style: italic; }

/* ═══════════════════════════════════════════════════════════
   BROWSE CARD
   ═══════════════════════════════════════════════════════════ */
.browse-card {
    background: var(--bg-browse);
    border: 1px solid var(--border-main);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px var(--shadow);
}
.browse-title { font-size: 1rem; font-weight: 700; color: var(--accent); }
.browse-plot  {
    font-size: 0.85rem; color: var(--text-muted);
    margin-top: 6px; line-height: 1.55;
}

/* ═══════════════════════════════════════════════════════════
   SECTION HEADER
   ═══════════════════════════════════════════════════════════ */
.section-header {
    font-size: 1.5rem; font-weight: 700; color: var(--accent);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 6px; margin-bottom: 20px;
}

/* ═══════════════════════════════════════════════════════════
   INPUT FIELDS
   ═══════════════════════════════════════════════════════════ */
textarea, .stTextInput input {
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 8px !important;
}
textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(245,197,24,0.2) !important;
}

/* ═══════════════════════════════════════════════════════════
   PRIMARY BUTTON
   ═══════════════════════════════════════════════════════════ */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: var(--accent-text) !important;
    font-weight: 700; border-radius: 8px;
    border: none !important; width: 100%;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover) !important;
}

/* ═══════════════════════════════════════════════════════════
   TAB BAR
   ═══════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-tabbar);
    border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--tab-inactive) !important;
    font-weight: 600; border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: var(--accent-text) !important;
}

/* ═══════════════════════════════════════════════════════════
   DIVIDER
   ═══════════════════════════════════════════════════════════ */
.divider { border-top: 1px solid var(--border-light); margin: 22px 0; }

/* ═══════════════════════════════════════════════════════════
   SIDEBAR TIP BOX
   ═══════════════════════════════════════════════════════════ */
.tip-box {
    background: var(--bg-tip);
    border: 1px solid var(--border-tip);
    border-radius: 8px; padding: 12px 14px; margin-top: 10px;
    font-size: 0.82rem; color: var(--text-tip) !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🎬 Loading movie database...")
def load_model():
    df = load_data("imdb_movies_2024.csv")
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)
    return df, vectorizer, tfidf_matrix


@st.cache_data
def compute_stats(_df):
    """Pre-compute all analytics so tabs load instantly."""
    df = _df.copy()
    df["word_count"] = df["Storyline"].str.split().str.len()

    # Length category
    bins   = [0, 15, 25, 35, 50, 100, 500]
    labels = ["Very Short", "Short", "Medium", "Long", "Very Long", "Epic"]
    df["length_cat"] = pd.cut(df["word_count"], bins=bins, labels=labels)

    # Top words from cleaned storylines
    all_words  = " ".join(df["Cleaned_Storyline"].dropna()).split()
    word_freq  = Counter(all_words).most_common(20)

    return df, word_freq


df, vectorizer, tfidf_matrix = load_model()
df_stats, word_freq = compute_stats(df)


# ─────────────────────────────────────────────────────────────────────────────
# THEME DETECTION  (light / dark — drives Plotly + Matplotlib colours)
# ─────────────────────────────────────────────────────────────────────────────

def get_theme() -> dict:
    """Return colour tokens for Plotly/Matplotlib based on active Streamlit theme."""
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = "dark"

    if base == "light":
        return dict(
            bg        = "#f5f5f5",
            plot_bg   = "#ffffff",
            font      = "#333333",
            title     = "#c9a800",
            grid      = "#dddddd",
            accent    = "#c9a800",
            fig_bg    = "#f5f5f5",
            ax_bg     = "#ffffff",
            mpl_text  = "#333333",
            wc_bg     = "#f5f5f5",
            colorscale= ["#eeeeee", "#c9a800"],
        )
    return dict(
        bg        = "#0d0d0d",
        plot_bg   = "#111111",
        font      = "#cccccc",
        title     = "#F5C518",
        grid      = "#222222",
        accent    = "#F5C518",
        fig_bg    = "#0d0d0d",
        ax_bg     = "#111111",
        mpl_text  = "#cccccc",
        wc_bg     = "#0d0d0d",
        colorscale= ["#333333", "#F5C518"],
    )

T = get_theme()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎬 IMDb 2024")
    st.markdown("**Movie Recommendation System**")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📊 Dataset")
    st.metric("Total Movies", f"{len(df):,}")
    st.metric("Year", "2024")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    1. Input cleaned → **TF-IDF** vector  
    2. **Cosine Similarity** vs all 5,099 movies  
    3. Top N matches returned by score  
    """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 💡 Try These")
    examples = [
        "Astronauts face danger in deep space",
        "Detective solves mysterious murders in a small town",
        "Two strangers fall in love against all odds",
        "A superhero battles a powerful villain to save the world",
        "Family struggles with secrets from the past",
    ]
    for ex in examples:
        st.caption(f"› {ex}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tip-box">💡 Tip: Longer, more descriptive inputs give better recommendations!</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding: 10px 0 4px 0;'>
    <span style='font-size:2.6rem; font-weight:800; color:#F5C518;'>🎬 IMDb 2024 Movie Recommender</span><br>
    <span style='font-size:1.05rem; color:#888;'>
        Describe a plot — discover your next favourite movie from 5,099 titles
    </span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Recommend Movies",
    "📊 Dataset Overview",
    "🗂️ Browse Movies",
    "📈 NLP Analytics",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATION
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">🔍 Find Similar Movies</div>', unsafe_allow_html=True)

    col_input, col_ctrl = st.columns([3, 1])

    with col_input:
        user_input = st.text_area(
            "📝 Enter a Movie Storyline",
            placeholder=(
                "e.g. A young soldier discovers a dark secret during a war "
                "mission and must choose between duty and conscience..."
            ),
            height=140,
            help="Describe any plot. The more detail you give, the better the match.",
        )

    with col_ctrl:
        st.markdown("<br>", unsafe_allow_html=True)
        top_n = st.slider("Results", min_value=3, max_value=10, value=5, key="top_n_slider")
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍 Find Movies", type="primary", key="search_btn")

    if search_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter a storyline to search.")
        else:
            with st.spinner("Analysing storyline..."):
                results = get_recommendations(
                    user_input, df, vectorizer, tfidf_matrix, top_n=top_n
                )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Top {top_n} Recommendations")

            if results.empty or results["Similarity Score"].max() == 0:
                st.info("No strong matches found. Try a more descriptive storyline.")
            else:
                for _, row in results.iterrows():
                    pct = f"{row['Similarity Score'] * 100:.1f}%"
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-rank">#{int(row['Rank'])} &nbsp;·&nbsp; Similarity Match</div>
                        <div class="movie-title">{row['Movie Name']}</div>
                        <div class="movie-plot">{row['Storyline']}</div>
                        <span class="score-pill">🎯 {pct} match</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                with st.expander("📋 Summary Table"):
                    display_df = results[["Rank", "Movie Name", "Similarity Score"]].copy()
                    display_df["Similarity Score"] = (
                        display_df["Similarity Score"] * 100
                    ).round(2).astype(str) + "%"
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Bar chart of scores
                fig = px.bar(
                    results,
                    x="Similarity Score",
                    y="Movie Name",
                    orientation="h",
                    color="Similarity Score",
                    color_continuous_scale=T["colorscale"],
                    labels={"Similarity Score": "Cosine Similarity", "Movie Name": ""},
                    title="Similarity Scores",
                )
                fig.update_layout(
                    paper_bgcolor=T["bg"], plot_bgcolor=T["bg"],
                    font_color=T["font"], title_font_color=T["title"],
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    height=320,
                )
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    # ── KPI Row ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🎬 Total Movies",   f"{len(df):,}")
    k2.metric("📝 Avg. Plot Words", f"{int(df_stats['word_count'].mean())}")
    k3.metric("📏 Longest Plot",    f"{int(df_stats['word_count'].max())} words")
    k4.metric("📏 Shortest Plot",   f"{int(df_stats['word_count'].min())} words")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ── Storyline Length Distribution ──
    with col_a:
        st.markdown("#### 📏 Storyline Length Distribution")
        fig_hist = px.histogram(
            df_stats,
            x="word_count",
            nbins=40,
            color_discrete_sequence=[T["accent"]],
            labels={"word_count": "Word Count", "count": "Number of Movies"},
        )
        fig_hist.update_layout(
            paper_bgcolor=T["bg"], plot_bgcolor=T["plot_bg"],
            font_color=T["font"], bargap=0.05,
            xaxis=dict(gridcolor=T["grid"]), yaxis=dict(gridcolor=T["grid"]),
            showlegend=False, height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Length Category Breakdown ──
    with col_b:
        st.markdown("#### 🏷️ Storyline Length Categories")
        cat_counts = df_stats["length_cat"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        order = ["Very Short", "Short", "Medium", "Long", "Very Long", "Epic"]
        cat_counts["Category"] = pd.Categorical(cat_counts["Category"], categories=order, ordered=True)
        cat_counts = cat_counts.sort_values("Category")

        fig_bar = px.bar(
            cat_counts, x="Category", y="Count",
            color="Count",
            color_continuous_scale=T["colorscale"],
            labels={"Count": "Number of Movies"},
        )
        fig_bar.update_layout(
            paper_bgcolor=T["bg"], plot_bgcolor=T["plot_bg"],
            font_color=T["font"], coloraxis_showscale=False,
            xaxis=dict(gridcolor=T["grid"]), yaxis=dict(gridcolor=T["grid"]),
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Word Count Box Plot ──
    st.markdown("#### 📦 Word Count Box Plot")
    fig_box = px.box(
        df_stats, y="word_count",
        color_discrete_sequence=[T["accent"]],
        labels={"word_count": "Word Count"},
        points="outliers",
    )
    fig_box.update_layout(
        paper_bgcolor=T["bg"], plot_bgcolor=T["plot_bg"],
        font_color=T["font"],
        yaxis=dict(gridcolor=T["grid"]),
        height=320,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Raw Data Preview ──
    st.markdown("#### 🗃️ Raw Data Preview")
    st.dataframe(
        df[["Movie Name", "Storyline"]].head(20),
        use_container_width=True,
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BROWSE MOVIES
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">🗂️ Browse All Movies</div>', unsafe_allow_html=True)

    col_s, col_n = st.columns([3, 1])
    with col_s:
        search_term = st.text_input(
            "🔎 Search by movie name or keyword in storyline",
            placeholder="e.g. love, detective, space, revenge...",
            key="browse_search",
        )
    with col_n:
        show_n = st.selectbox("Show per page", [10, 20, 50], index=0, key="show_n")

    # Filter
    if search_term.strip():
        mask = (
            df["Movie Name"].str.contains(search_term, case=False, na=False)
            | df["Storyline"].str.contains(search_term, case=False, na=False)
        )
        filtered = df[mask].reset_index(drop=True)
    else:
        filtered = df.copy()

    st.markdown(
        f"<small style='color:#888;'>Showing {min(show_n, len(filtered))} of {len(filtered):,} movies</small>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    for i, row in filtered.head(show_n).iterrows():
        st.markdown(f"""
        <div class="browse-card">
            <div class="browse-title">🎬 {row['Movie Name']}</div>
            <div class="browse-plot">{row['Storyline']}</div>
        </div>
        """, unsafe_allow_html=True)

    if len(filtered) == 0:
        st.info("No movies found for that search term.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — NLP ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">📈 NLP Analytics</div>', unsafe_allow_html=True)

    col_w, col_f = st.columns(2)

    # ── Word Cloud ──
    with col_w:
        st.markdown("#### ☁️ Most Common Storyline Words")
        text_for_cloud = " ".join(df["Cleaned_Storyline"].dropna())
        wc = WordCloud(
            width=700, height=380,
            background_color=T["wc_bg"],
            colormap="YlOrBr" if T["wc_bg"] == "#0d0d0d" else "autumn",
            max_words=120,
            collocations=False,
        ).generate(text_for_cloud)

        fig_wc, ax = plt.subplots(figsize=(7, 3.8))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig_wc.patch.set_facecolor(T["fig_bg"])
        st.pyplot(fig_wc)
        plt.close(fig_wc)

    # ── Top 20 Words Bar ──
    with col_f:
        st.markdown("#### 🔤 Top 20 Most Frequent Words")
        words_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
        fig_words = px.bar(
            words_df,
            x="Frequency", y="Word",
            orientation="h",
            color="Frequency",
            color_continuous_scale=T["colorscale"],
        )
        fig_words.update_layout(
            paper_bgcolor=T["bg"], plot_bgcolor=T["plot_bg"],
            font_color=T["font"], coloraxis_showscale=False,
            yaxis=dict(autorange="reversed", gridcolor=T["grid"]),
            xaxis=dict(gridcolor=T["grid"]),
            height=400,
        )
        st.plotly_chart(fig_words, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── TF-IDF Info ──
    st.markdown("#### 🧮 TF-IDF Model Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("📚 Vocabulary Size",    "5,000 features")
    c2.metric("🔗 N-gram Range",       "Unigrams + Bigrams")
    c3.metric("🎯 Similarity Metric",  "Cosine Similarity")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Cosine Similarity heatmap (sample 15 movies) ──
    st.markdown("#### 🔥 Cosine Similarity Heatmap (Top 15 Movies Sample)")
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sample_df  = df.head(15).reset_index(drop=True)
    sample_vec = vectorizer.transform(sample_df["Cleaned_Storyline"])
    sim_matrix = cos_sim(sample_vec).round(2)

    fig_heat, ax_heat = plt.subplots(figsize=(10, 7))
    fig_heat.patch.set_facecolor(T["fig_bg"])
    ax_heat.set_facecolor(T["ax_bg"])
    sns.heatmap(
        sim_matrix,
        xticklabels=sample_df["Movie Name"].str[:18],
        yticklabels=sample_df["Movie Name"].str[:18],
        annot=True, fmt=".2f",
        cmap="YlOrBr",
        linewidths=0.5,
        linecolor="#222",
        ax=ax_heat,
    )
    ax_heat.tick_params(colors=T["mpl_text"], labelsize=7)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig_heat)
    plt.close(fig_heat)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#444; font-size:0.82rem;'>
    Built with Streamlit &nbsp;·&nbsp; TF-IDF &nbsp;·&nbsp; Cosine Similarity
    &nbsp;·&nbsp; IMDb 2024 Data &nbsp;·&nbsp; 5,099 Movies
</div>
""", unsafe_allow_html=True)