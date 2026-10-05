from pathlib import Path
import re
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'spam_classifier.joblib'

st.set_page_config(page_title='Spam Email/SMS Classifier', page_icon='🛡️', layout='wide')


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[^a-z\s_]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


artifact = load_artifact()

st.title('🛡️ Spam Email / SMS Classifier')
st.caption('UCI SMS Spam Collection • TF-IDF • Logistic Regression • Streamlit')

if artifact is None:
    st.warning('The trained model is not available yet.')
    st.markdown('### Run these commands in the VS Code terminal')
    st.code('pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py', language='bash')
    st.info('The training script automatically downloads and prepares the official UCI dataset.')
    st.stop()

vectorizer = artifact['vectorizer']
model = artifact['model']
metrics = artifact['metrics']
info = artifact['dataset_info']

classifier_tab, performance_tab, dataset_tab, method_tab = st.tabs([
    '🔍 Classifier', '📊 Performance', '🗂️ Dataset', '📚 Methodology'
])

with classifier_tab:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader('Enter a message')
        examples = {
            'Custom message': '',
            'Normal example': 'Hey, are we still meeting for lunch at 1 pm today?',
            'Spam example': 'Congratulations! You have won a free prize. Call now to claim your reward.'
        }
        choice = st.selectbox('Example', list(examples))
        message = st.text_area('SMS / Email text', value=examples[choice], height=190,
                               placeholder='Paste a message here...')
        predict = st.button('Classify Message', type='primary', use_container_width=True)

        if predict:
            if not message.strip():
                st.error('Please enter a message first.')
            else:
                cleaned = clean_text(message)
                X = vectorizer.transform([cleaned])
                pred = int(model.predict(X)[0])
                probs = model.predict_proba(X)[0]
                spam_prob = float(probs[1])
                ham_prob = float(probs[0])

                st.divider()
                if pred == 1:
                    st.error('🚨 SPAM DETECTED')
                else:
                    st.success('✅ HAM / NOT SPAM')

                c1, c2 = st.columns(2)
                c1.metric('Spam probability', f'{spam_prob * 100:.2f}%')
                c2.metric('Ham probability', f'{ham_prob * 100:.2f}%')
                st.progress(spam_prob, text='Spam probability')

                with st.expander('Show text preprocessing'):
                    st.write('**Original:**')
                    st.code(message)
                    st.write('**Cleaned:**')
                    st.code(cleaned or '[empty after preprocessing]')

    with right:
        st.subheader('Project summary')
        st.metric('Messages', f"{info['rows']:,}")
        c1, c2 = st.columns(2)
        c1.metric('Ham', f"{info['ham']:,}")
        c2.metric('Spam', f"{info['spam']:,}")
        st.write(f"TF-IDF vocabulary: **{info['features']:,} features**")
        st.write('Model: **Logistic Regression**')
        st.write('Split: **80% train / 20% test**')

with performance_tab:
    st.subheader('Model evaluation')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Accuracy', f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric('Precision', f"{metrics['precision'] * 100:.2f}%")
    c3.metric('Recall', f"{metrics['recall'] * 100:.2f}%")
    c4.metric('F1-score', f"{metrics['f1'] * 100:.2f}%")

    st.markdown('### Confusion Matrix')
    cm = metrics['confusion_matrix']
    cm_df = pd.DataFrame(cm, index=['Actual Ham', 'Actual Spam'], columns=['Predicted Ham', 'Predicted Spam'])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown('### Class-wise report')
    report = metrics['classification_report']
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

with dataset_tab:
    st.subheader('Dataset overview')
    st.write('This project uses the **UCI SMS Spam Collection**, containing 5,574 labeled SMS messages.')
    d1, d2, d3 = st.columns(3)
    d1.metric('Total', f"{info['rows']:,}")
    d2.metric('Ham', f"{info['ham']:,}")
    d3.metric('Spam', f"{info['spam']:,}")

    chart_df = pd.DataFrame({'Class': ['Ham', 'Spam'], 'Messages': [info['ham'], info['spam']]}).set_index('Class')
    st.bar_chart(chart_df)

    if artifact.get('sample_rows'):
        st.markdown('### Sample rows')
        st.dataframe(pd.DataFrame(artifact['sample_rows']), use_container_width=True)

    st.info('The original UCI distribution is stored as a tab-separated file named SMSSpamCollection. The training script also creates data/sms_spam.csv for convenient viewing.')

with method_tab:
    st.subheader('Machine learning pipeline')
    st.markdown('''
**1. Dataset**  
The UCI SMS Spam Collection provides two labels: `ham` (normal) and `spam` (unwanted message).

**2. Text preprocessing**  
Convert to lowercase, replace URLs/email addresses/numbers, remove non-letter noise, and normalize spaces.

**3. Tokenization**  
`TfidfVectorizer` tokenizes the cleaned text into word features and also creates 1-word and 2-word features (unigrams and bigrams).

**4. TF-IDF**  
TF-IDF converts the text into a numerical sparse matrix where informative terms receive greater weight.

**5. Logistic Regression**  
The classifier learns a binary decision boundary: `0 = ham`, `1 = spam`.

**6. Evaluation**  
The model is tested on data it did not see during training. We report accuracy, precision, recall, F1-score, and a confusion matrix.
''')
    st.code('SMS text → preprocessing → tokenization → TF-IDF → Logistic Regression → ham/spam + probability')
    st.markdown('### Why Logistic Regression?')
    st.write('It is a standard and efficient baseline for sparse, high-dimensional TF-IDF text features, and it provides class probabilities that are useful for an interactive classifier UI.')
    st.markdown('### Dataset source')
    st.write('UCI Machine Learning Repository — SMS Spam Collection (Dataset 228).')
    st.write('https://archive.ics.uci.edu/dataset/228/sms-spam-collection')
