import re


DIGIT_RE = re.compile(r"\d")


def normalize_token(token: str, config: dict) -> str:
    preprocessing = config.get("preprocessing", {})

    if preprocessing.get("lowercase_tokens", False):
        token = token.lower()

    if preprocessing.get("normalize_digits", False):
        replacement = preprocessing.get("digit_replacement", "0")
        token = DIGIT_RE.sub(replacement, token)

    return token


def is_stopword(token: str, stopwords: set[str]) -> bool:
    return token.lower() in stopwords