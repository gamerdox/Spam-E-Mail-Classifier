"""
Cosmetic-only cleanup for display.

The underlying Enron-Spam corpus mirror ships text that has already been
run through an NLTK-style tokenizer by whoever built it: punctuation is
space-separated ("don ' t", "re : var for cob 2 nd aug 2000", "$ 30 , 000"),
and everything is lowercased. That's fine for a bag-of-words classifier
(classifier.py re-tokenizes from scratch and ignores spacing/case anyway),
but it looks broken to a human reading it in an inbox UI.

detokenize() undoes that formatting for DISPLAY ONLY -- it is never applied
to TRAINING_DATA, and classification always runs on real corpus text
(detokenized or not makes no difference to the classifier's output, since
its own tokenizer strips exactly the punctuation this function is fixing
the spacing of).
"""
import re

_ORDINAL_RE = re.compile(r"(\d)\s+(st|nd|rd|th)\b", re.IGNORECASE)
_THOUSANDS_RE = re.compile(r"(\d)\s*,\s*(\d{3})(?=\D|$)")


def _capitalize(match: "re.Match") -> str:
    return match.group(1) + match.group(2).upper()


def detokenize(text: str) -> str:
    if not text:
        return text

    # Collapse runs of spaces/tabs, and runs of blank lines down to one.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # No space before closing punctuation: "units ." -> "units."
    text = re.sub(r"\s+([.,!?;:%)\]])", r"\1", text)
    # No space after opening punctuation: "( see" -> "(see"
    text = re.sub(r"([(\[])\s+", r"\1", text)
    # Contractions/possessives: "don ' t" -> "don't", "everyone ' s" -> "everyone's"
    text = re.sub(r"\s*'\s*", "'", text)
    # Currency: "$ 30714" -> "$30714"
    text = re.sub(r"\$\s+", "$", text)
    # Ordinal suffixes: "2 nd" -> "2nd"
    text = _ORDINAL_RE.sub(r"\1\2", text)
    # Thousands separators: "300 , 000" -> "300,000"
    text = _THOUSANDS_RE.sub(r"\1,\2", text)
    # Decimals: "4 . 3 %" (already "4. 3%" after the punctuation-spacing
    # step above) -> "4.3%"
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
    # Hyphenated compounds: "pre - approved" -> "pre-approved". Also tidies
    # the long "- - - - -" divider lines some of these emails use.
    text = re.sub(r"(\w)\s+-\s+(\w)", r"\1-\2", text)
    text = re.sub(r"(?:-\s+){2,}-", "----", text)

    # Capitalize "i" as a standalone word/contraction ("i ' m" -> already
    # "i'm" by this point -> "I'm"; bare "i" -> "I").
    text = re.sub(r"(?<![A-Za-z'])i(?=['’]|\b)", "I", text)

    # Capitalize the first letter after sentence-ending punctuation, a
    # newline, or the start of the text.
    text = re.sub(r"(^|[.!?]\s+|\n)([a-z])", _capitalize, text)

    return text.strip()
