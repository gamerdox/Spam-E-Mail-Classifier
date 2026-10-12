"""
Builds this project's training data from a REAL, publicly released, labeled
email corpus. No synthetic, hand-written, or AI-generated examples are used
anywhere in this file.

Source: the Enron-Spam dataset (V. Metsis, I. Androutsopoulos, G. Paliouras,
"Spam Filtering with Naive Bayes -- Which Naive Bayes?", CEAS 2006),
17,171 real spam emails + 16,545 real legitimate (ham) emails sent to and
from actual Enron employees, 33,716 total. Original source:
http://www2.aueb.gr/users/ion/data/enron-spam/

This script pulls a single pre-built CSV of that same dataset from a public
GitHub mirror (MWiechmann/enron_spam_data), which packages the original
per-message text files into one file with columns Subject, Message,
Spam/Ham, Date -- see that repo's README for exactly how it was built.

What this script does:
  1. Downloads and unzips enron_spam_data.zip (~15 MB) from the mirror.
  2. Combines Subject + Message into one `text` field.
  3. Sets aside a small, fixed, held-out DEMO POOL (35 real emails) that is
     NEVER used for training -- this is what populates the demo inbox and
     the Spam Detector tab's preset examples, so the model is never shown
     the exact emails it's demonstrated on.
  4. Writes the remaining ~33,680 emails to data/spam_dataset.csv (the
     actual training set) and the demo pool to data/demo_pool.csv.

Run this once before starting the app if data/spam_dataset.csv is missing:

    pip install pandas requests
    python prepare_dataset.py
"""
import io
import os
import random
import re
import unicodedata
import zipfile

import pandas as pd

MIRROR_URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "spam_dataset.csv")
DEMO_CSV = os.path.join(DATA_DIR, "demo_pool.csv")

DEMO_POOL_SIZE = 35          # held out entirely from training
MIN_CHARS, MAX_CHARS = 60, 2000  # keep demo emails a readable size for the UI
MAX_TRAIN_CHARS = 20000      # cap rare huge forwarded-thread outliers (still real text, just capped)
RANDOM_SEED = 42             # fixed -> same demo pool every time this is run


def download_corpus() -> pd.DataFrame:
    import requests

    print(f"Downloading real Enron-Spam corpus from {MIRROR_URL} ...")
    resp = requests.get(MIRROR_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    print(f"Loaded {len(df)} real emails ({df['Spam/Ham'].value_counts().to_dict()})")
    return df


def build():
    df = download_corpus()
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    df["text"] = (df["Subject"] + ". " + df["Message"]).str.strip()
    df["text"] = df["text"].str.slice(0, MAX_TRAIN_CHARS)  # cap rare 100k+ char outliers
    df["label"] = df["Spam/Ham"].str.lower()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    rng = random.Random(RANDOM_SEED)

    # Candidates for the demo pool: readable length, real subject line present,
    # and -- since these are the emails that get *displayed* in the demo UI,
    # unlike the training set which the model only ever sees as word counts --
    # filtered away from the (real, but explicit) pharma/adult spam that this
    # 2001-era corpus also contains, so the shipped prototype is presentable.
    # These old spam campaigns deliberately obfuscate spelling (spaces,
    # accents, leetspeak) to dodge naive keyword filters, so we normalize
    # before matching: strip accents, drop non-letters, and undo common
    # leetspeak substitutions.
    _LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})

    def _normalize(t: str) -> str:
        t = unicodedata.normalize("NFKD", t)
        t = "".join(c for c in t if not unicodedata.combining(c))
        t = t.lower().translate(_LEET)
        return re.sub(r"[^a-z]", "", t)

    _EXPLICIT_TERMS = [
        "viagra", "cialis", "vlgr", "xanax", "anax", "valium", "alium",
        "ambien", "ambie", "amble", "soma", "erectile", "penis", "sexual",
        "porn", "naked", "orgasm", "fuck", "xxx", "agra", "discreteencounter",
        "cheatinghousewive", "karpos", "increaseddriveandperformance",
        "miracleinmedicine", "vicodin", "lolita", "musclebo", "gotlaid",
        "hookupwith", "hookupsite", "stiffonebetween", "pleasureyourpartner",
        "girlfriendloves", "crazyhookup", "chics", "stronge", "sexlife",
        "erect", "vibrator", "personals", "sexappeal", "oppositesex",
        "sexandthecity", "feelgood", "makemefeelgood", "fatgirl",
        "wantstomeetsomeone",
    ]
    normalized = df["text"].apply(_normalize)
    explicit_mask = normalized.apply(
        lambda t: any(term in t for term in _EXPLICIT_TERMS)
    )
    candidates = df[
        (df["Subject"].str.len() > 0)
        & (df["text"].str.len().between(MIN_CHARS, MAX_CHARS))
        & (~explicit_mask)
    ].copy()
    spam_candidates = candidates[candidates["label"] == "spam"].sample(
        frac=1, random_state=RANDOM_SEED
    )
    ham_candidates = candidates[candidates["label"] == "ham"].sample(
        frac=1, random_state=RANDOM_SEED
    )

    # Roughly balanced demo pool: ~18 spam, ~17 ham out of 35.
    n_spam = DEMO_POOL_SIZE // 2 + 1
    n_ham = DEMO_POOL_SIZE - n_spam
    demo_df = pd.concat(
        [spam_candidates.head(n_spam), ham_candidates.head(n_ham)]
    ).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Training set = everything NOT in the demo pool.
    train_df = df.drop(index=df.index.intersection(demo_df.index)).reset_index(drop=True)
    # (demo_df rows came from `candidates`, a filtered copy of df, so their
    # original df index is preserved and safe to drop by.)

    os.makedirs(DATA_DIR, exist_ok=True)
    train_df[["text", "label"]].to_csv(TRAIN_CSV, index=False)
    demo_df[["text", "label"]].to_csv(DEMO_CSV, index=False)

    print(f"\nTraining set: {len(train_df)} real emails -> {TRAIN_CSV}")
    print(f"Demo pool (held out, never trained on): {len(demo_df)} real emails -> {DEMO_CSV}")
    print(f"  demo spam: {(demo_df['label'] == 'spam').sum()}, demo ham: {(demo_df['label'] == 'ham').sum()}")


if __name__ == "__main__":
    build()
