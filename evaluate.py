"""
Honest accuracy check: trains on 80% of the real training set and tests on
the other 20% (never seen during training), then prints accuracy, spam
precision/recall/F1, and a confusion matrix.

Run after prepare_dataset.py:
    python evaluate.py
"""
import csv
import random
import sys
from typing import Dict, List, Tuple

from classifier import BayesianSpamClassifier

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # safety net for any long email field

DATA_PATH = "data/spam_dataset.csv"
SEED = 42
TEST_FRACTION = 0.2


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["text"], row["label"]) for row in csv.DictReader(f)]


def stratified_split(rows, test_fraction, seed):
    """Stratified 80/20 split, grouped by exact email text first.

    The real corpus has thousands of verbatim duplicate emails (the same
    forwarded thread, the same bulk spam template sent to many Enron
    addresses). Splitting row-by-row would let a duplicate land on both
    sides, so the model could see the *exact* text of a "held-out" test
    email during training -- that's leakage, not generalization, and it
    quietly inflates the reported accuracy. Grouping by text first means
    every copy of a duplicate email goes to the same side of the split, so
    nothing in the test set was literally present in training."""
    rng = random.Random(seed)

    groups: Dict[str, List[Tuple[str, str]]] = {}
    for text, label in rows:
        groups.setdefault(text, []).append((text, label))

    by_label: Dict[str, List[List[Tuple[str, str]]]] = {"spam": [], "ham": []}
    for group in groups.values():
        by_label[group[0][1]].append(group)

    train, test = [], []
    for label, label_groups in by_label.items():
        label_groups = label_groups[:]
        rng.shuffle(label_groups)
        cut = int(len(label_groups) * (1 - test_fraction))
        for group in label_groups[:cut]:
            train.extend(group)
        for group in label_groups[cut:]:
            test.extend(group)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def evaluate_holdout(data_path=DATA_PATH, test_fraction=TEST_FRACTION, seed=SEED):
    """Runs the honest 80/20 held-out evaluation and returns every number as
    a plain dict -- used by both `python evaluate.py` (prints it) and the
    live app's /api/stats (so the About tab shows real, freshly-computed
    numbers instead of a number typed into source code)."""
    rows = load_rows(data_path)
    train_rows, test_rows = stratified_split(rows, test_fraction, seed)

    clf = BayesianSpamClassifier(alpha=1.0)
    clf.train(train_rows)

    tp = fp = tn = fn = 0
    for text, label in test_rows:
        pred_spam = clf.classify(text)["is_spam"]
        actual_spam = label == "spam"
        if pred_spam and actual_spam:
            tp += 1
        elif pred_spam and not actual_spam:
            fp += 1
        elif not pred_spam and actual_spam:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "vocabulary_size": len(clf.vocabulary),
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main():
    m = evaluate_holdout()
    print(f"Training examples: {m['train_count']}")
    print(f"Test examples (held out, never trained on): {m['test_count']}")
    print(f"Vocabulary size: {m['vocabulary_size']}")
    print()
    print(f"Accuracy:        {m['accuracy']:.2f}%")
    print(f"Spam precision:  {m['precision']:.2f}%   (of emails flagged spam, % that really were)")
    print(f"Spam recall:     {m['recall']:.2f}%   (of real spam, % that got caught)")
    print(f"F1 score:        {m['f1']:.2f}%")
    print()
    print("Confusion matrix:")
    print(f"               predicted spam   predicted ham")
    print(f"  actual spam  {m['tp']:>14}   {m['fn']:>13}")
    print(f"  actual ham   {m['fp']:>14}   {m['tn']:>13}")


if __name__ == "__main__":
    main()
