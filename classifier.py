"""
A Multinomial Naive Bayes spam classifier, implemented from first principles
(no scikit-learn) so every step of the math is visible and explainable.

HOW NAIVE BAYES IS APPLIED TO CLASSIFYING AN EMAIL
---------------------------------------------------
We want P(Spam | email), the probability the email is spam given its words.
Bayes' theorem rewrites that as:

    P(Spam | words) = P(words | Spam) * P(Spam) / P(words)

P(words) is the same for both classes, so we only need to compare
P(words | Spam)*P(Spam) against P(words | Ham)*P(Ham), and pick whichever
is bigger. "Naive" Bayes makes one simplifying assumption: that, given the
class, each word's presence is independent of the others (untrue in real
language, but it works remarkably well for spam filtering). That turns
P(words | Spam) into a simple product over every word in the email:

    P(words | Spam) = P(w_1 | Spam) * P(w_2 | Spam) * ... * P(w_n | Spam)

Each P(w | Spam) is learned from training data as:

    P(w | Spam) = (how many times w appeared in spam + alpha)
                  / (total words in all spam emails + alpha * vocabulary size)

The "+ alpha" (Laplace/additive smoothing, alpha = 1.0 here) stops a word
that never appeared in spam during training from forcing P(words|Spam) to
exactly zero the first time it shows up -- every word gets a small,
non-zero probability under both classes.

Multiplying tens of thousands of small probabilities together underflows
floating point fast, so we work in log-space instead (sums instead of
products), then convert the final log-odds back into a 0-100% probability
with a sigmoid at the end. The math is equivalent either way.

P(Spam) and P(Ham), the priors, are just how common each class was in the
training data (e.g. roughly half spam / half ham in the real corpus this
is trained on).
"""
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

# Common English stopwords to ignore -- they carry almost no spam/ham signal
# and would otherwise dilute the words that do.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


class BayesianSpamClassifier:
    """Multinomial Naive Bayes with Laplace smoothing, trained on (text, label) pairs."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.spam_word_counts: Counter = Counter()
        self.ham_word_counts: Counter = Counter()
        self.total_spam_words: int = 0
        self.total_ham_words: int = 0
        self.num_spam_docs: int = 0
        self.num_ham_docs: int = 0
        self.vocabulary: set = set()
        self.p_spam_prior: float = 0.5
        self.p_ham_prior: float = 0.5

    def tokenize(self, text: str) -> List[str]:
        """Lowercase, extract alphanumeric tokens (keeping $ ! % as they carry
        spam signal), drop stopwords."""
        tokens = re.findall(r"[a-zA-Z0-9$!%]{2,}", text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def train(self, documents: List[Tuple[str, str]]):
        """documents: list of (text, label) where label is 'spam' or 'ham'."""
        self.spam_word_counts.clear()
        self.ham_word_counts.clear()
        self.total_spam_words = self.total_ham_words = 0
        self.num_spam_docs = self.num_ham_docs = 0
        self.vocabulary.clear()

        for text, label in documents:
            tokens = self.tokenize(text)
            if label.lower() == "spam":
                self.num_spam_docs += 1
                self.spam_word_counts.update(tokens)
                self.total_spam_words += len(tokens)
            else:
                self.num_ham_docs += 1
                self.ham_word_counts.update(tokens)
                self.total_ham_words += len(tokens)
            self.vocabulary.update(tokens)

        total_docs = self.num_spam_docs + self.num_ham_docs
        if total_docs > 0:
            self.p_spam_prior = self.num_spam_docs / total_docs
            self.p_ham_prior = self.num_ham_docs / total_docs

    def word_prob_spam(self, word: str) -> float:
        vocab_size = len(self.vocabulary)
        return (self.spam_word_counts[word] + self.alpha) / (
            self.total_spam_words + self.alpha * vocab_size
        )

    def word_prob_ham(self, word: str) -> float:
        vocab_size = len(self.vocabulary)
        return (self.ham_word_counts[word] + self.alpha) / (
            self.total_ham_words + self.alpha * vocab_size
        )

    def get_word_spamminess(self, word: str) -> float:
        """P(Spam | this one word), via Bayes' rule applied to a single word --
        used only for the UI's word-level explanation, not the document decision."""
        p_w_spam = self.word_prob_spam(word)
        p_w_ham = self.word_prob_ham(word)
        numerator = p_w_spam * self.p_spam_prior
        denominator = numerator + (p_w_ham * self.p_ham_prior)
        return numerator / denominator if denominator else 0.5

    def classify(self, text: str) -> Dict[str, Any]:
        tokens = self.tokenize(text)

        log_prob_spam = math.log(self.p_spam_prior or 1e-12)
        log_prob_ham = math.log(self.p_ham_prior or 1e-12)

        word_analysis = []
        for word in tokens:
            p_w_spam = self.word_prob_spam(word)
            p_w_ham = self.word_prob_ham(word)
            log_prob_spam += math.log(p_w_spam)
            log_prob_ham += math.log(p_w_ham)

            if word in self.vocabulary:
                word_score = self.get_word_spamminess(word)
                ratio = p_w_spam / p_w_ham if p_w_ham > 0 else 10.0
                word_analysis.append(
                    {
                        "word": word,
                        "spam_count": self.spam_word_counts[word],
                        "ham_count": self.ham_word_counts[word],
                        "p_w_given_spam": round(p_w_spam, 8),
                        "p_w_given_ham": round(p_w_ham, 8),
                        "spamminess": round(word_score, 4),
                        "likelihood_ratio": round(ratio, 2),
                    }
                )

        log_diff = log_prob_spam - log_prob_ham
        log_diff_clipped = max(-50.0, min(50.0, log_diff))
        posterior_spam = 1.0 / (1.0 + math.exp(-log_diff_clipped))
        posterior_ham = 1.0 - posterior_spam
        is_spam = posterior_spam >= 0.5

        spam_indicators = [
            w for w in word_analysis if w["spamminess"] > 0.6 and w["spam_count"] > 0
        ]
        spam_indicators.sort(key=lambda w: w["spamminess"], reverse=True)
        ham_indicators = [
            w for w in word_analysis if w["spamminess"] < 0.4 and w["ham_count"] > 0
        ]
        ham_indicators.sort(key=lambda w: w["spamminess"])

        return {
            "is_spam": is_spam,
            "spam_probability": round(posterior_spam * 100, 2),
            "ham_probability": round(posterior_ham * 100, 2),
            "log_prob_spam": round(log_prob_spam, 4),
            "log_prob_ham": round(log_prob_ham, 4),
            "p_spam_prior": round(self.p_spam_prior, 4),
            "p_ham_prior": round(self.p_ham_prior, 4),
            "token_count": len(tokens),
            "known_token_count": len(word_analysis),
            "vocabulary_size": len(self.vocabulary),
            "top_spam_words": spam_indicators[:10],
            "top_ham_words": ham_indicators[:10],
        }

    def highlight_text(self, text: str) -> str:
        """Wrap strong spam-indicator words in <mark> tags for red highlighting in the UI."""
        tokens = set(re.findall(r"[a-zA-Z0-9$!%]{2,}", text.lower()))
        flagged, details = set(), {}

        for token in tokens:
            if token in self.vocabulary and self.spam_word_counts[token] > 0:
                spamminess = self.get_word_spamminess(token)
                if spamminess >= 0.65:
                    flagged.add(token)
                    details[token] = {
                        "spamminess": round(spamminess * 100, 1),
                        "count": self.spam_word_counts[token],
                    }

        if not flagged:
            return text

        def replace_match(match):
            word = match.group(0)
            clean = word.lower()
            if clean in flagged:
                info = details[clean]
                return (
                    f'<mark class="spam-highlight" '
                    f'title="P(Spam|word): {info["spamminess"]}% | seen in {info["count"]} spam emails">'
                    f"{word}</mark>"
                )
            return word

        pattern = re.compile(r"\b[a-zA-Z0-9$!%]+\b")
        return pattern.sub(replace_match, text)
