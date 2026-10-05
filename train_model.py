from pathlib import Path
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from download_dataset import download_and_prepare

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
CSV_PATH = DATA_DIR / 'sms_spam.csv'
MODEL_PATH = MODEL_DIR / 'spam_classifier.joblib'


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[^a-z\s_]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    if not CSV_PATH.exists():
        df = download_and_prepare()
    else:
        df = pd.read_csv(CSV_PATH)

    df = df.dropna(subset=['label', 'message']).copy()
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df['message'] = df['message'].astype(str)
    df = df[df['label'].isin(['ham', 'spam'])].copy()
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    df['cleaned'] = df['message'].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['target'], test_size=0.20, random_state=42, stratify=df['target']
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=50000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True, zero_division=0)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }

    dataset_info = {
        'rows': int(len(df)),
        'ham': int((df['target'] == 0).sum()),
        'spam': int((df['target'] == 1).sum()),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'test_size': 0.20,
        'features': int(X_train_tfidf.shape[1]),
    }

    # Useful examples for the UI.
    sample_rows = df[['label', 'message']].head(12).to_dict(orient='records')

    artifact = {
        'vectorizer': vectorizer,
        'model': model,
        'metrics': metrics,
        'dataset_info': dataset_info,
        'sample_rows': sample_rows,
        'cleaner_version': 'basic_regex_v1',
    }
    joblib.dump(artifact, MODEL_PATH)

    print('\n=== TRAINING COMPLETE ===')
    print(f'Dataset rows     : {dataset_info["rows"]}')
    print(f'Ham messages      : {dataset_info["ham"]}')
    print(f'Spam messages     : {dataset_info["spam"]}')
    print(f'Training rows     : {dataset_info["train_rows"]}')
    print(f'Test rows         : {dataset_info["test_rows"]}')
    print(f'TF-IDF features   : {dataset_info["features"]}')
    print(f'Accuracy          : {metrics["accuracy"]:.4f}')
    print(f'Precision         : {metrics["precision"]:.4f}')
    print(f'Recall            : {metrics["recall"]:.4f}')
    print(f'F1-score          : {metrics["f1"]:.4f}')
    print(f'Confusion matrix  :\n{cm}')
    print(f'\nSaved model to: {MODEL_PATH}')


if __name__ == '__main__':
    main()
