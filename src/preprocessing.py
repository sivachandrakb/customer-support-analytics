import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


def clean_text(text: str) -> str:
    """Basic cleanup."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)      # URLs
    text = re.sub(r"<.*?>", "", text)                  # HTML
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_pii(text: str) -> str:
    """Remove emails, phone numbers, names, etc."""
    try:
        results = analyzer.analyze(text=text, language="en")
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    except Exception:
        return text


def preprocess(text: str) -> str:
    return remove_pii(clean_text(text))
