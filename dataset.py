"""
Loads this project's real data into the three things the app needs:

  TRAINING_DATA             -- every email the model is trained on
                                (data/spam_dataset.csv, ~33.7k real Enron-Spam
                                emails -- see prepare_dataset.py).

  INBOX_EMAILS              -- 30 real emails randomly sampled, on each
                                process start, from data/demo_pool.csv -- a
                                fixed set of 35 real emails that were set
                                aside BEFORE training and are never part of
                                TRAINING_DATA. The model has never seen these
                                during training, so classifying them for the
                                inbox view is an honest, out-of-sample demo.

  SPAM_DETECTOR_PRESETS     -- 5 fixed real emails from that same held-out
                                pool, used as one-click examples on the
                                Spam Detector tab.

No email body or subject here is written or edited by hand -- every word of
every email's content comes verbatim from the real corpus. The only things
this file invents are *display metadata* the raw corpus doesn't include
(a synthetic sender display name/address and inbox timestamp), which never
feed into the classifier and are cosmetic only -- clearly commented below.

Run `python prepare_dataset.py` once before starting the app if
data/spam_dataset.csv or data/demo_pool.csv don't exist yet.
"""
import csv
import os
import random

from textclean import detokenize

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_TRAIN_PATH = os.path.join(_DATA_DIR, "spam_dataset.csv")
_DEMO_PATH = os.path.join(_DATA_DIR, "demo_pool.csv")

INBOX_SIZE = 30

# Fixed row indices into demo_pool.csv (stable because demo_pool.csv itself
# is regenerated deterministically with RANDOM_SEED=42 in prepare_dataset.py)
# chosen to show a diverse, presentable mix: a financial/phishing scam, a
# pharma/diet-pill scam, a software-piracy scam, and two genuine business
# emails. Picked by hand after reading the real held-out pool end to end.
_PRESET_INDICES = [33, 4, 13, 27, 2]


def _load_csv_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python prepare_dataset.py` first to download "
            "the real Enron-Spam corpus and build this project's data files -- "
            "see prepare_dataset.py for details."
        )
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["text"], row["label"].strip().lower()) for row in csv.DictReader(f)]


TRAINING_DATA = _load_csv_rows(_TRAIN_PATH)
_DEMO_POOL = _load_csv_rows(_DEMO_PATH)

# ---------------------------------------------------------------------------
# Cosmetic-only display metadata. The raw Enron-Spam corpus has no sender
# field, so something has to appear in the inbox's "from" column. These
# names/addresses are invented for display, are NOT real people, and are
# never seen by the classifier -- classify() only ever runs on the real
# `text` column loaded above. Ham gets plausible internal-coworker-style
# senders; spam gets generic external sender patterns typical of the real
# scams in this corpus.
# ---------------------------------------------------------------------------
_HAM_SENDERS = [
    ("Vince Kaminski", "vince.kaminski@enron.com"),
    ("Louise Kitchen", "louise.kitchen@enron.com"),
    ("David Martinez", "david.martinez@enron.com"),
    ("Sarah Jenkins", "sarah.jenkins@enron.com"),
    ("HR Department", "hr-notices@enron.com"),
    ("Chris Bowling", "chris.bowling@enron.com"),
    ("Scheduling Desk", "scheduling@enron.com"),
    ("IT Helpdesk", "it-helpdesk@enron.com"),
    ("Legal Department", "legal-dept@enron.com"),
    ("Amy Kaminski", "amy.kaminski@enron.com"),
]
_SPAM_SENDERS = [
    ("Account Services", "no-reply@secure-billing-notice.net"),
    ("Promotions Team", "offers@mega-deals-today.biz"),
    ("Pharmacy Direct", "orders@discount-rx-online.info"),
    ("Finance Dept.", "approval@fast-loan-confirm.com"),
    ("Software Deals", "sales@cheap-license-store.ru"),
    ("Rewards Center", "winner@claim-your-prize-now.xyz"),
    ("Marketing Dept", "campaign@bulk-email-blast.net"),
    ("Customer Support", "support@verify-account-now.cc"),
]


def _synth_sender(label: str, rng: random.Random):
    """Deterministic given rng -- picks a display sender for the inbox UI only."""
    pool = _HAM_SENDERS if label == "ham" else _SPAM_SENDERS
    return rng.choice(pool)


def _subject_and_snippet(text: str):
    """The corpus stores "Subject. Message" joined in `text` (see
    prepare_dataset.py); split that back apart for display, falling back
    gracefully if an email has no ". " separator.

    The raw corpus text is also pre-tokenized (space-separated punctuation,
    all lowercase) by whoever built the mirror -- detokenize() cleans that
    up for display only; the classifier never sees this cleaned version
    (TRAINING_DATA is untouched), and cleaning up punctuation spacing/case
    doesn't change what the model outputs either, since classifier.py's own
    tokenizer already strips that same punctuation and lowercases."""
    if ". " in text:
        subject, rest = text.split(". ", 1)
    else:
        subject, rest = text[:60], text
    subject = detokenize(subject).strip() or "(no subject)"
    body = detokenize(rest)
    snippet = " ".join(body.split())[:140]
    return subject, snippet, body


_INBOX_TIMES = [
    "9:14 AM", "9:47 AM", "10:02 AM", "10:28 AM", "11:05 AM", "11:41 AM",
    "12:18 PM", "1:03 PM", "1:52 PM", "2:30 PM", "3:11 PM", "3:49 PM",
    "4:22 PM", "Yesterday", "Yesterday", "Yesterday", "Oct 9", "Oct 9",
    "Oct 8", "Oct 8", "Oct 7", "Oct 6", "Oct 5", "Oct 3", "Oct 1",
    "Sep 29", "Sep 27", "Sep 24", "Sep 20", "Sep 15",
]


def build_inbox(seed=None):
    """30 real emails sampled fresh from the held-out demo pool (never the
    training set). `seed=None` -> a different random sample each run."""
    rng = random.Random(seed)
    sample = rng.sample(_DEMO_POOL, min(INBOX_SIZE, len(_DEMO_POOL)))
    # If the pool is smaller than INBOX_SIZE, top up with repeats (won't
    # happen at the current pool size of 35, but keeps this robust).
    while len(sample) < INBOX_SIZE:
        sample.append(rng.choice(_DEMO_POOL))

    inbox = []
    for i, (text, label) in enumerate(sample):
        sender_name, sender_email = _synth_sender(label, rng)
        subject, snippet, body = _subject_and_snippet(text)
        inbox.append({
            "id": f"msg-{i + 1:03d}",
            "sender_name": sender_name,
            "sender_email": sender_email,
            "subject": subject,
            "snippet": snippet,
            "body": body,
            "date": _INBOX_TIMES[i % len(_INBOX_TIMES)],
            "is_read": rng.random() < 0.55,
            "is_starred": rng.random() < 0.12,
            "true_label": label,  # real corpus label, for reference/debugging only
        })
    return inbox


def build_presets():
    """5 fixed real emails from the held-out pool for the Spam Detector tab."""
    presets = []
    labels = ["Scam / Phishing", "Pharma / Diet Scam", "Software Piracy Scam",
              "Genuine Business Email", "Genuine Business Email"]
    for n, idx in enumerate(_PRESET_INDICES):
        text, label = _DEMO_POOL[idx]
        subject, _snippet, body = _subject_and_snippet(text)
        presets.append({
            "id": f"preset-{n + 1}",
            "title": labels[n],
            "subject": subject,
            "body": body,
            "true_label": label,
        })
    return presets
