"""
Flask backend for the Gmail-style spam classifier prototype.

Routes:
  GET  /                      -> the single-page app shell
  GET  /api/inbox             -> 30 real emails (fresh random sample each
                                  server start), each pre-classified and
                                  spam-highlighted for the inbox list
  GET  /api/email/<id>        -> one inbox email's full detail (classified)
  POST /api/email/<id>/star   -> toggle starred, persists for this server run
  POST /api/email/<id>/read   -> mark an email read (called when opened)
  POST /api/compose/send      -> "send" a composed email -- NO spam check
                                  here on purpose (the user asked for
                                  composing to no longer auto-check spam;
                                  checking now lives only on the Spam
                                  Detector tab). Sent mail is tracked
                                  server-side so a "Sent" folder is real,
                                  not just a toast that goes nowhere.
  GET  /api/sent              -> everything sent this server run
  GET  /api/detector/presets  -> the 5 fixed example emails for the Spam
                                  Detector tab
  POST /api/detector/check    -> classify arbitrary text typed/pasted into
                                  the Spam Detector tab, returning the full
                                  Naive Bayes breakdown for the explanation
                                  panel
  POST /api/detector/feedback -> record agreement/disagreement with a
                                  verdict (see note in the handler on why
                                  this doesn't retrain the model live)
  GET  /api/stats             -> training-set size, vocabulary size, and a
                                  REAL held-out accuracy/confusion matrix
                                  computed fresh at startup -- for the
                                  "About this model" panel
"""
import os

from flask import Flask, jsonify, request, render_template

from classifier import BayesianSpamClassifier
from evaluate import evaluate_holdout
import dataset

app = Flask(__name__)

# --- Train once at startup on the real training set -----------------------
classifier = BayesianSpamClassifier(alpha=1.0)
classifier.train(dataset.TRAINING_DATA)

# --- Honest held-out accuracy, computed fresh every startup (80/20 split
# of the real training data, trained/tested completely separately from the
# `classifier` above) -- not a number typed into source code.
_HOLDOUT_METRICS = evaluate_holdout()

# --- Build a fresh random inbox sample + the fixed presets at startup -----
_INBOX = dataset.build_inbox()          # list of dicts, see dataset.py
_INBOX_BY_ID = {e["id"]: e for e in _INBOX}
_PRESETS = dataset.build_presets()

# --- Session-lifetime state (in-memory; resets when the server restarts) --
_SENT_MAIL = []           # emails sent from the Compose window
_FEEDBACK = {"agree": 0, "disagree": 0}


def _classify_for_ui(text: str) -> dict:
    result = classifier.classify(text)
    result["highlighted_html"] = classifier.highlight_text(text)
    return result


def _inbox_summary(email: dict) -> dict:
    # Classify on the full subject + body (not just the list-view snippet) so
    # the inbox's spam flag matches what you'd see on the email detail page --
    # using only the truncated snippet would throw away most of the email's
    # real signal before the model ever sees it.
    verdict = _classify_for_ui(email["subject"] + ". " + email["body"])
    return {
        "id": email["id"],
        "sender_name": email["sender_name"],
        "sender_email": email["sender_email"],
        "subject": email["subject"],
        "snippet": email["snippet"],
        "date": email["date"],
        "is_read": email["is_read"],
        "is_starred": email["is_starred"],
        "is_spam": verdict["is_spam"],
        "spam_probability": verdict["spam_probability"],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/inbox")
def api_inbox():
    return jsonify([_inbox_summary(e) for e in _INBOX])


@app.route("/api/email/<email_id>")
def api_email_detail(email_id):
    email = _INBOX_BY_ID.get(email_id)
    if not email:
        return jsonify({"error": "not found"}), 404
    full_text = email["subject"] + ". " + email["body"]
    verdict = _classify_for_ui(full_text)
    return jsonify({
        **{k: email[k] for k in (
            "id", "sender_name", "sender_email", "subject", "body", "date",
            "is_read", "is_starred",
        )},
        "classification": verdict,
    })


@app.route("/api/email/<email_id>/star", methods=["POST"])
def api_email_star(email_id):
    email = _INBOX_BY_ID.get(email_id)
    if not email:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    email["is_starred"] = bool(data.get("starred", not email["is_starred"]))
    return jsonify({"id": email_id, "is_starred": email["is_starred"]})


@app.route("/api/email/<email_id>/read", methods=["POST"])
def api_email_read(email_id):
    email = _INBOX_BY_ID.get(email_id)
    if not email:
        return jsonify({"error": "not found"}), 404
    email["is_read"] = True
    return jsonify({"id": email_id, "is_read": True})


@app.route("/api/compose/send", methods=["POST"])
def api_compose_send():
    """Intentionally does NOT run the spam classifier. The user asked for
    composing a new email to no longer auto-check for spam -- that check now
    lives only on the Spam Detector tab (see /api/detector/check). The sent
    message IS kept, in a real "Sent" folder, so Send goes somewhere."""
    data = request.get_json(silent=True) or {}
    to = (data.get("to") or "").strip()
    subject = (data.get("subject") or "").strip()
    body = (data.get("body") or "").strip()
    if not to:
        return jsonify({"error": "A recipient is required."}), 400

    sent_email = {
        "id": f"sent-{len(_SENT_MAIL) + 1:03d}",
        "to": to,
        "subject": subject or "(no subject)",
        "body": body,
        "snippet": " ".join(body.split())[:140],
        "date": "Just now",
    }
    _SENT_MAIL.insert(0, sent_email)
    return jsonify({"status": "sent", "message": f"Message sent to {to}.", "email": sent_email})


@app.route("/api/sent")
def api_sent():
    return jsonify(_SENT_MAIL)


@app.route("/api/detector/presets")
def api_detector_presets():
    return jsonify(_PRESETS)


@app.route("/api/detector/check", methods=["POST"])
def api_detector_check():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()
    body = (data.get("body") or "").strip()
    text = f"{subject}. {body}".strip(". ").strip()
    if not text:
        return jsonify({"error": "Write or paste some email text to check."}), 400

    result = _classify_for_ui(text)
    return jsonify(result)


@app.route("/api/detector/feedback", methods=["POST"])
def api_detector_feedback():
    """Records whether the person agreed with a verdict. This is a Naive
    Bayes model trained once, in batch, at server startup -- it has no
    online-learning step, so a disagreement here can't retrain it on the
    spot. What it DOES do: count the feedback, and tell you exactly what
    retraining it for real would involve (add the corrected example to
    data/spam_dataset.csv and rerun), rather than faking an instant fix."""
    data = request.get_json(silent=True) or {}
    agree = bool(data.get("agree"))
    _FEEDBACK["agree" if agree else "disagree"] += 1
    return jsonify({
        "recorded": True,
        "agree_count": _FEEDBACK["agree"],
        "disagree_count": _FEEDBACK["disagree"],
    })


@app.route("/api/stats")
def api_stats():
    m = _HOLDOUT_METRICS
    return jsonify({
        "training_examples": len(dataset.TRAINING_DATA),
        "vocabulary_size": len(classifier.vocabulary),
        "num_spam_docs": classifier.num_spam_docs,
        "num_ham_docs": classifier.num_ham_docs,
        "p_spam_prior": round(classifier.p_spam_prior, 4),
        "p_ham_prior": round(classifier.p_ham_prior, 4),
        "alpha": classifier.alpha,
        # Computed fresh at startup by evaluate_holdout() -- a real 80/20
        # split, trained and tested completely separately from the live
        # `classifier` above. Not hardcoded.
        "held_out_train_count": m["train_count"],
        "held_out_test_count": m["test_count"],
        "held_out_accuracy": round(m["accuracy"], 2),
        "held_out_precision": round(m["precision"], 2),
        "held_out_recall": round(m["recall"], 2),
        "held_out_f1": round(m["f1"], 2),
        "confusion_matrix": {"tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]},
        "feedback": dict(_FEEDBACK),
        "dataset_source": "Enron-Spam corpus (Metsis, Androutsopoulos, Paliouras, CEAS 2006) — 33,716 real emails",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
