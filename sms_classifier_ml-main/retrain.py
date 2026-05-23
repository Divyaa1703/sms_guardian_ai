import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from multilingual import detect_language, preprocess_text, LANGUAGE_FILE_SUFFIX

EXTRA_DATASETS = ['multi_language_spam_samples.csv']


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1')


def _normalize_label(value):
    if pd.isna(value):
        return None
    label = str(value).strip().lower()
    if label in ['ham', 'not spam', 'not_spam', '0', 'false', 'no', 'n', 'legit', 'legitimate']:
        return 'ham'
    if label in ['spam', '1', 'true', 'yes', 'y']:
        return 'spam'
    return None


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})

    if 'label' not in df.columns or 'message' not in df.columns:
        raise ValueError('Dataset must contain label and message columns.')

    df = df[['label', 'message'] + ([col for col in ['language'] if col in df.columns])].dropna(subset=['label', 'message'])
    df['label'] = df['label'].apply(_normalize_label)
    df = df[df['label'].isin(['ham', 'spam'])]
    df['message'] = df['message'].astype(str)
    if 'language' in df.columns:
        df['language'] = df['language'].astype(str).str.lower().str.strip()
    return df


def load_dataset(csv_path='spam.csv') -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = _prepare_dataset(_read_csv(csv_path))

    for extra_path in EXTRA_DATASETS:
        if extra_path != csv_path and os.path.exists(extra_path):
            try:
                extra_df = _prepare_dataset(_read_csv(extra_path))
                df = pd.concat([df, extra_df], ignore_index=True)
            except Exception:
                pass

    return df


def build_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['language'] = df.apply(
        lambda row: row['language'] if pd.notna(row.get('language')) and str(row.get('language')).strip() else detect_language(row['message']),
        axis=1
    )
    df['processed_text'] = df.apply(lambda row: preprocess_text(row['message'], row['language']), axis=1)
    df['target'] = df['label'].map({'ham': 'Not Spam', 'spam': 'Spam'})
    return df


def retrain_model(csv_path='spam.csv', model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    df = load_dataset(csv_path)
    df = build_training_data(df)

    # Train a default multilingual model on the full dataset
    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['target']

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save language-specific models when available
    for language in sorted(df['language'].unique()):
        subset = df[df['language'] == language]
        if subset.empty:
            continue

        suffix = LANGUAGE_FILE_SUFFIX.get(language, language[:2])
        vectorizer_name = f'vectorizer_{suffix}.pkl'
        model_name = f'model_{suffix}.pkl'

        lang_vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
        X_lang = lang_vectorizer.fit_transform(subset['processed_text'])
        lang_model = LogisticRegression(max_iter=2000)
        lang_model.fit(X_lang, subset['target'])

        with open(vectorizer_name, 'wb') as f:
            pickle.dump(lang_vectorizer, f)
        with open(model_name, 'wb') as f:
            pickle.dump(lang_model, f)

    return model, vectorizer


if __name__ == '__main__':
    print('Retraining spam detection model from spam.csv...')
    try:
        model, vectorizer = retrain_model()
        print('Retraining finished successfully.')
        print('Saved model to model.pkl and vectorizer to vectorizer.pkl')
    except Exception as e:
        print(f'Error during retraining: {e}')
