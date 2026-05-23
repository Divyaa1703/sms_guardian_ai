import re
import urllib.parse

URL_REGEX = re.compile(r'https?://[^\s]+|www\.[^\s]+', re.IGNORECASE)
SHORTENER_DOMAINS = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'tiny.cc', 'is.gd', 'buff.ly',
    'adf.ly', 'bit.do', 'shorturl.at', 'shorte.st'
]

SUSPICIOUS_URL_PATTERNS = [
    'login', 'secure', 'verify', 'account', 'update', 'confirm', 'bank', 'payment',
    'webscr', 'signin', 'reset', 'urgent', 'reward', 'free', 'claim'
]

FORCED_SPAM_URL_PATTERNS = [
    'secure-bank-login', 'bank-login', 'secure-bank', 'login-bank', 'verify-bank',
    'confirm-bank', 'bank.verify', 'account.verify', 'secure-login', 'login.verify',
    'secure-account', 'bank-update', 'update-bank', 'secure-bank-login.com'
]


def extract_urls(text: str) -> list:
    if not text or not isinstance(text, str):
        return []
    return re.findall(URL_REGEX, text)


def _normalize_url(url: str) -> str:
    if url.startswith('www.'):
        return 'http://' + url
    return url


def _suspicious_domain(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(_normalize_url(url))
        domain = parsed.netloc.lower()
        return any(short in domain for short in SHORTENER_DOMAINS)
    except Exception:
        return False


def _suspicious_url_pattern(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(_normalize_url(url))
        text = parsed.netloc.lower() + parsed.path.lower() + parsed.query.lower()
        return any(pattern in text for pattern in SUSPICIOUS_URL_PATTERNS)
    except Exception:
        return False


def _forced_spam_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(_normalize_url(url))
        text = parsed.netloc.lower() + parsed.path.lower() + parsed.query.lower()
        return any(pattern in text for pattern in FORCED_SPAM_URL_PATTERNS)
    except Exception:
        return False


def assess_link_spam(urls: list) -> dict:
    result = {
        'urls': urls,
        'score': 0.0,
        'reasons': [],
        'force_spam': False
    }

    if not urls:
        result['reasons'].append('No links detected in the message.')
        return result

    suspicious_count = 0
    for url in urls:
        if _forced_spam_url(url):
            suspicious_count += 1
            result['force_spam'] = True
            result['reasons'].append(f'Forced spam URL pattern detected: {url}')
        elif _suspicious_domain(url):
            suspicious_count += 1
            result['reasons'].append(f'Suspicious shortener or redirect link: {url}')
        elif _suspicious_url_pattern(url):
            suspicious_count += 1
            result['reasons'].append(f'Suspicious keyword detected in URL: {url}')
        elif len(url) < 30:
            suspicious_count += 1
            result['reasons'].append(f'Short or unusual URL: {url}')

    if suspicious_count > 0:
        if result['force_spam']:
            result['score'] = 1.0
        else:
            result['score'] = min(1.0, 0.3 + 0.25 * suspicious_count)
    else:
        result['score'] = 0.05
        result['reasons'].append('No suspicious link patterns detected.')

    return result
