import re
import string
from nltk.stem.porter import PorterStemmer

SUPPORTED_LANGUAGES = ['english', 'hindi', 'marathi']
LANGUAGE_LABELS = {
    'english': 'English',
    'hindi': 'Hindi',
    'marathi': 'Marathi'
}
LANGUAGE_FILE_SUFFIX = {
    'english': 'en',
    'hindi': 'hi',
    'marathi': 'mr'
}

# Devanagari range includes Hindi and Marathi scripts
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')
MARATHI_HINT = re.compile(r'[ळॉ]')

ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'could', "couldn't", 'did', "didn't",
    'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few',
    'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't",
    'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers',
    'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm",
    "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
    "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not',
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll",
    "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's",
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these',
    'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll",
    "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
    "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with',
    "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves'
}

ps = PorterStemmer()


def detect_language(text: str) -> str:
    """Detect English, Hindi, or Marathi based on script characters."""
    if not text or not isinstance(text, str):
        return 'english'

    if DEVANAGARI_RANGE.search(text):
        if MARATHI_HINT.search(text):
            return 'marathi'
        return 'hindi'

    return 'english'


def preprocess_text(text: str, language: str = 'english') -> str:
    """Normalize text for the detected language."""
    if not text:
        return ''

    text = str(text).strip()

    if language == 'english':
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        filtered_words = [word for word in tokens if word not in ENGLISH_STOPWORDS]
        stemmed_words = [ps.stem(word) for word in filtered_words]
        return ' '.join(stemmed_words)

    # Hindi / Marathi preprocessing: preserve Devanagari text, digits, and Latin tokens
    # so fraud URLs and English-link tokens can still contribute to the model.
    text = text.lower()
    text = re.sub(r'[^0-9\u0900-\u097F a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
