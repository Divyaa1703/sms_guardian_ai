import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer
from multilingual import detect_language, preprocess_text
from link_detector import extract_urls, assess_link_spam

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

ps = PorterStemmer()


def transform_text(text, language='english'):
    return preprocess_text(text, language)


def parse_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()
    try:
        uploaded_file.seek(0)
        if filename.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        if filename.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            messages = [line.strip() for line in content.splitlines() if line.strip()]
            return pd.DataFrame({'message': messages})
    except Exception:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception:
            return None
    return None


def find_message_column(df):
    candidates = [
        col for col in df.columns
        if col.lower() in ['message', 'text', 'sms', 'msg', 'body', 'content']
    ]
    if candidates:
        return candidates[0]
    if len(df.columns) == 1:
        return df.columns[0]
    return None


def classify_single_message(text):
    if not isinstance(text, str):
        text = str(text)
    language = detect_language(text)
    transformed_text = transform_text(text, language)
    urls = extract_urls(text)
    link_info = assess_link_spam(urls)

    vector_input = tfidf.transform([transformed_text])
    result = model.predict(vector_input)[0]
    probability = model.predict_proba(vector_input)[0]
    spam_probability = probability[1]

    spam_keywords = ['won', 'win', 'winner', 'prize', 'free', 'congratulations',
                     'selected', 'urgent', 'click', 'call now', 'limited time',
                     'offer', 'deal', 'money', 'cash', 'reward', 'claim']
    high_risk_patterns = ['you won', 'you win', 'you have won', 'winner!',
                          'congratulations!', 'you are selected', 'free money',
                          'click here', 'call now', 'urgent!']

    message_lower = text.lower()
    keyword_found = any(keyword in message_lower for keyword in spam_keywords)
    high_risk_found = any(pattern in message_lower for pattern in high_risk_patterns)
    suspicious_link_found = link_info['score'] >= 0.3
    suspicious_link_force = link_info.get('force_spam', False)

    if high_risk_found:
        spam_threshold = 0.1
    elif keyword_found:
        spam_threshold = 0.3
    elif suspicious_link_found or suspicious_link_force:
        spam_threshold = 0.25
    else:
        spam_threshold = 0.5

    if spam_probability > spam_threshold or str(result).lower() == 'spam' or suspicious_link_found or suspicious_link_force:
        prediction = 'Spam'
        confidence = max(spam_probability, link_info['score']) * 100
    else:
        prediction = 'Not Spam'
        confidence = probability[0] * 100

    return {
        'prediction': prediction,
        'confidence': confidence,
        'language': language,
        'processed_text': transformed_text,
        'link_score': link_info['score'],
        'link_reasons': '; '.join(link_info['reasons'])
    }


def classify_batch(df, message_column):
    rows = []
    for _, row in df.iterrows():
        message = row[message_column]
        record = classify_single_message(message)
        record['message'] = str(message)
        rows.append(record)
    return pd.DataFrame(rows)

# Load models with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.error("Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

uploaded_file = st.file_uploader(
    "Upload SMS messages for bulk classification",
    type=['csv', 'txt'],
    help="Upload a CSV with a message column or a TXT file with one message per line."
)

if uploaded_file:
    uploaded_df = parse_uploaded_file(uploaded_file)
    if uploaded_df is None:
        st.error("Unable to read the uploaded file. Please upload a valid CSV or TXT file.")
    else:
        message_column = find_message_column(uploaded_df)
        if message_column:
            st.info(f"Detected message column: {message_column}")
        else:
            message_column = st.selectbox("Select message column", uploaded_df.columns)

        st.write(f"Loaded {len(uploaded_df)} messages.")
        st.dataframe(uploaded_df.head())

        if st.button("Classify uploaded file", key='classify_uploaded'):
            try:
                result_df = classify_batch(uploaded_df, message_column)
                st.success(f"Classified {len(result_df)} messages.")
                st.dataframe(result_df)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download classification results",
                    csv,
                    file_name='bulk_classification_results.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Error classifying uploaded file: {e}")

if st.button('Predict'):
    if input_sms.strip():  # Check if input is not empty
        try:
            # 1. detect language, preprocess, and analyze links
            language = detect_language(input_sms)
            transformed_sms = transform_text(input_sms, language)
            urls = extract_urls(input_sms)
            link_info = assess_link_spam(urls)

            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            probability = model.predict_proba(vector_input)[0]

            # 4. Enhanced spam detection
            spam_probability = probability[1]

            # Check for common spam keywords
            spam_keywords = ['won', 'win', 'winner', 'prize', 'free', 'congratulations',
                           'selected', 'urgent', 'click', 'call now', 'limited time',
                           'offer', 'deal', 'money', 'cash', 'reward', 'claim']

            # High-risk patterns that are almost always spam
            high_risk_patterns = ['you won', 'you win', 'you have won', 'winner!',
                                'congratulations!', 'you are selected', 'free money',
                                'click here', 'call now', 'urgent!']

            forced_spam_patterns = ['congratulations', 'selected for', 'claim your',
                                    'click here', 'urgent', 'winner', 'prize', 'free money',
                                    'you won', 'you have won', 'won a', 'win a',
                                    'bank login', 'secure login', 'verify account',
                                    'account update', 'reset password', 'confirm your']

            message_lower = input_sms.lower()
            keyword_found = any(keyword in message_lower for keyword in spam_keywords)
            high_risk_found = any(pattern in message_lower for pattern in high_risk_patterns)
            forced_spam_found = any(pattern in message_lower for pattern in forced_spam_patterns)
            suspicious_link_found = link_info['score'] >= 0.3
            suspicious_link_force = link_info.get('force_spam', False)

            # Enhanced detection with multiple thresholds
            if forced_spam_found or suspicious_link_force:
                spam_threshold = 0.0  # Force spam for very strong indicators
            elif high_risk_found:
                spam_threshold = 0.1  # Very low threshold for high-risk patterns
            elif keyword_found:
                spam_threshold = 0.3  # Lower threshold for spam keywords
            elif suspicious_link_found:
                spam_threshold = 0.25  # Lower threshold for suspicious URLs
            else:
                spam_threshold = 0.5  # Normal threshold

            if spam_probability > spam_threshold or str(result).lower() == 'spam' or suspicious_link_found or suspicious_link_force or forced_spam_found:
                st.header("Spam")
                st.write(f"Confidence: {max(spam_probability, link_info['score'])*100:.1f}%")
                if suspicious_link_force:
                    st.warning("🚨 Forced suspicious URL detected in this message")
                elif forced_spam_found:
                    st.warning("🚨 Strong spam phrase detected in message")
                elif suspicious_link_found:
                    st.warning("🚨 Suspicious URL detected in this message")
                elif high_risk_found:
                    st.warning("🚨 High-risk spam pattern detected!")
                elif keyword_found and spam_probability <= 0.5:
                    st.info("🔍 Detected spam keywords in message")
            else:
                st.header("Not Spam")
                st.write(f"Confidence: {probability[0]*100:.1f}%")

            st.markdown("---")
            st.markdown("**Detected language:**")
            st.write(language.title())
            st.markdown("**Processed text:**")
            st.code(transformed_sms)
            st.markdown("**Extracted URLs:**")
            st.write(urls if urls else "No links found.")
            st.markdown("**Link spam score:**")
            st.write(f"{link_info['score']:.2f}")
            for reason in link_info['reasons']:
                st.write(f"- {reason}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter a message to classify.")
