# Spam Email / SMS Classifier

A college machine-learning project for binary spam classification using the **UCI SMS Spam Collection**, **TF-IDF**, and **Logistic Regression**, with a Streamlit UI.

## Official dataset

UCI Machine Learning Repository — SMS Spam Collection:
https://archive.ics.uci.edu/dataset/228/sms-spam-collection

Official direct download archive:
https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip

The UCI page states that the dataset contains **5,574 instances** and that the original corpus is one text file where each line contains the class (`ham` or `spam`) followed by the raw SMS. The dataset is CC BY 4.0. Cite Almeida & Hidalgo (2011), DOI 10.24432/C5CC84.

## Project structure

```text
spam_email_sms_classifier_project/
├── app.py
├── train_model.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── data/
│   └── (dataset is downloaded here)
└── models/
    └── (trained model is created here)
```

## Arch Linux / Linux setup

Open a terminal in this folder.

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download + prepare dataset + train model

```bash
python train_model.py
```

This script will automatically download the official UCI archive, extract `SMSSpamCollection`, create `data/sms_spam.csv`, train the model, evaluate it, and save:

```text
models/spam_classifier.joblib
```

### 5. Start the UI

```bash
streamlit run app.py
```

Open the URL shown by Streamlit, usually `http://localhost:8501`.

## Manual dataset download

If automatic download fails:

1. Open the official UCI page:
   https://archive.ics.uci.edu/dataset/228/sms-spam-collection
2. Download the dataset ZIP.
3. Extract the file named `SMSSpamCollection`.
4. Place it inside this project's `data/` directory.
5. Run:

```bash
python train_model.py
```

The raw file format is:

```text
ham<TAB>message text
spam<TAB>message text
```

## What the UI demonstrates

- Real-time spam/ham prediction
- Spam probability
- Text preprocessing preview
- Accuracy, precision, recall, F1-score
- Confusion matrix
- Dataset statistics and sample rows
- Explanation of tokenization, TF-IDF, and Logistic Regression

## Suggested viva explanation

**Why this dataset?**

It contains raw SMS text plus a spam/ham label, so it directly supports text preprocessing, tokenization, TF-IDF and binary classification.

**Why TF-IDF?**

A machine-learning model needs numbers. TF-IDF converts text into weighted numerical features and reduces the importance of very common words.

**Why Logistic Regression?**

It performs well on high-dimensional sparse text features and produces class probabilities.

**What is a false positive?**

A legitimate message incorrectly classified as spam.

**What is a false negative?**

A spam message incorrectly classified as legitimate.

## Academic citation

Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
