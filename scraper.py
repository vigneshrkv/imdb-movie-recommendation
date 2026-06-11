"""
scraper.py
----------
IMDb 2024 Movie Scraper using Selenium.
Extracts Movie Name + Storyline for up to 10,000 movies.
Saves output to: imdb_movies_2024.csv

Run with:
    python scraper.py

Requirements:
    pip install selenium webdriver-manager pandas nltk
"""

import re
import time
import random
import logging
import pandas as pd
import nltk

from nltk.corpus import stopwords
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)
from webdriver_manager.chrome import ChromeDriverManager

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = (
    "https://www.imdb.com/search/title/"
    "?title_type=feature&release_date=2024-01-01,2024-12-31"
)

TARGET_MOVIES  = 10_000    # how many movies to collect
SAVE_EVERY     = 250       # save CSV every N movies (checkpoint)
OUTPUT_FILE    = "imdb_movies_2024.csv"
PAGE_LOAD_WAIT = 10        # seconds to wait for page elements
SCROLL_PAUSE   = 1.5       # seconds between scrolls
CLICK_PAUSE    = 2.5       # seconds after clicking Load More

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────────────────────────────────────

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Replicate the same cleaning used in the provided dataset:
      - lowercase
      - remove punctuation and numbers
      - remove stopwords
      - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# DRIVER SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_driver() -> webdriver.Chrome:
    """
    Create a headless Chrome driver.
    webdriver-manager handles ChromeDriver version automatically.
    """
    options = Options()
    options.add_argument("--headless=new")          # headless Chrome
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)

    # Mask webdriver property to avoid bot detection
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_movies_from_page(driver: webdriver.Chrome) -> list[dict]:
    """
    Parse all visible movie cards on the current page.
    Returns a list of dicts: {Movie Name, Storyline, Cleaned_Storyline}
    """
    movies = []

    # IMDb renders each result as <li class="ipc-metadata-list-summary-item ...">
    cards = driver.find_elements(
        By.CSS_SELECTOR,
        "li.ipc-metadata-list-summary-item",
    )

    for card in cards:
        try:
            # ── Movie Name ──
            title_el = card.find_element(
                By.CSS_SELECTOR, "h3.ipc-title__text"
            )
            raw_title = title_el.text.strip()

            # Remove leading rank number  "1. Movie Name" → "Movie Name"
            name = re.sub(r"^\d+\.\s*", "", raw_title).strip()
            if not name:
                continue

            # ── Storyline ──
            try:
                plot_el = card.find_element(
                    By.CSS_SELECTOR,
                    "div.ipc-html-content-inner-div",
                )
                storyline = plot_el.text.strip()
            except NoSuchElementException:
                # Some cards have no plot summary — skip them
                continue

            if len(storyline) < 10:
                continue

            cleaned = clean_text(storyline)

            movies.append({
                "Movie Name":         name,
                "Storyline":          storyline,
                "Cleaned_Storyline":  cleaned,
            })

        except (NoSuchElementException, StaleElementReferenceException):
            continue

    return movies


def scroll_to_bottom(driver: webdriver.Chrome) -> None:
    """Scroll to the bottom of the page so lazy-loaded content renders."""
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE)


def click_load_more(driver: webdriver.Chrome, wait: WebDriverWait) -> bool:
    """
    Click the '50 more' / 'Load More' button.
    Returns True if clicked successfully, False if button not found.
    """
    # IMDb uses a button with class containing 'load-more' or text '50 more'
    selectors = [
        "button.ipc-btn--half-rounded-right",          # newer IMDb layout
        "button[data-testid='adv-search-get-next-page']",
        "//button[contains(text(),'50 more')]",
        "//button[contains(text(),'Load more')]",
        "//button[contains(@class,'load-more')]",
    ]

    for sel in selectors:
        try:
            if sel.startswith("//"):
                btn = wait.until(EC.element_to_be_clickable((By.XPATH, sel)))
            else:
                btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))

            driver.execute_script("arguments[0].scrollIntoView(true);", btn)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(CLICK_PAUSE + random.uniform(0, 1))
            return True

        except (TimeoutException, NoSuchElementException):
            continue

    return False  # No button found — end of results


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(data: list[dict], filepath: str) -> None:
    """Save current data to CSV (deduplicated)."""
    df = pd.DataFrame(data)
    df.drop_duplicates(subset=["Movie Name"], inplace=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    log.info("💾  Checkpoint saved → %s  (%d movies)", filepath, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCRAPER
# ─────────────────────────────────────────────────────────────────────────────

def scrape_imdb(target: int = TARGET_MOVIES) -> pd.DataFrame:
    """
    Main scraping loop.
    Strategy:
      1. Open BASE_URL
      2. Extract visible movies
      3. Click 'Load More' → scroll → extract new movies
      4. Repeat until target reached or no more results
      5. Save checkpoint every SAVE_EVERY movies
    """
    log.info("🎬  Starting IMDb 2024 scraper  |  Target: %d movies", target)

    driver = build_driver()
    wait   = WebDriverWait(driver, PAGE_LOAD_WAIT)

    all_movies: list[dict] = []
    seen_titles: set[str]  = set()
    load_more_clicks = 0

    try:
        # ── Initial page load ──
        log.info("🌐  Loading IMDb search page...")
        driver.get(BASE_URL)
        time.sleep(3)

        # Accept cookie banner if present
        try:
            accept_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button[data-testid='accept-button']")
                )
            )
            accept_btn.click()
            time.sleep(1)
            log.info("🍪  Cookie banner accepted")
        except TimeoutException:
            pass  # No cookie banner

        # ── Scraping loop ──
        while len(all_movies) < target:

            scroll_to_bottom(driver)

            # Extract movies currently on page
            page_movies = extract_movies_from_page(driver)

            new_count = 0
            for m in page_movies:
                if m["Movie Name"] not in seen_titles:
                    seen_titles.add(m["Movie Name"])
                    all_movies.append(m)
                    new_count += 1

            log.info(
                "📋  Load #%d  |  New: %d  |  Total: %d / %d",
                load_more_clicks,
                new_count,
                len(all_movies),
                target,
            )

            # Checkpoint save
            if len(all_movies) % SAVE_EVERY < new_count or len(all_movies) >= target:
                save_checkpoint(all_movies, OUTPUT_FILE)

            if len(all_movies) >= target:
                log.info("🎯  Target reached!")
                break

            # Click 'Load More'
            clicked = click_load_more(driver, wait)
            if not clicked:
                log.warning("⚠️  'Load More' button not found. End of results.")
                break

            load_more_clicks += 1

            # Polite random delay to avoid rate-limiting
            time.sleep(random.uniform(1.5, 3.0))

    except KeyboardInterrupt:
        log.info("⛔  Interrupted by user. Saving collected data...")

    except Exception as exc:
        log.error("❌  Unexpected error: %s", exc, exc_info=True)

    finally:
        driver.quit()
        log.info("🔒  Browser closed")

    # ── Final save ──
    df = pd.DataFrame(all_movies)
    df.drop_duplicates(subset=["Movie Name"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    log.info("✅  Scraping complete  |  Total movies saved: %d", len(df))
    log.info("📂  Output file: %s", OUTPUT_FILE)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = scrape_imdb(target=TARGET_MOVIES)

    print("\n" + "=" * 55)
    print(f"  {'Scraping Summary':^50}")
    print("=" * 55)
    print(f"  Total Movies Scraped : {len(df):,}")
    print(f"  Columns              : {list(df.columns)}")
    print(f"  Output File          : {OUTPUT_FILE}")
    print("=" * 55)
    print("\nSample Output:")
    print(df.head(3).to_string(index=False))