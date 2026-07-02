"""Task 1 baseline classifier: TF-IDF + multinomial logistic regression.

A linear model over sparse TF-IDF features is fast to train on CPU, reaches
~91-92% test accuracy on AG News, and its ``predict_proba`` gives the
confidence scores the deferral system (Tasks 3-5) compares against the
expert-competence model.
"""

import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from . import data

CLASSIFIER_FILE = "classifier.joblib"


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            max_features=300_000,
        )),
        ("logreg", LogisticRegression(
            C=4.0,
            max_iter=1000,
            random_state=data.RANDOM_STATE,
        )),
    ])


def train_classifier(train_df):
    pipe = build_pipeline()
    pipe.fit(train_df["text"], train_df["label"])
    return pipe


def save_classifier(pipe, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(models_dir, CLASSIFIER_FILE))


def load_classifier(models_dir):
    return joblib.load(os.path.join(models_dir, CLASSIFIER_FILE))


def classifier_available(models_dir):
    return os.path.exists(os.path.join(models_dir, CLASSIFIER_FILE))


def evaluate_classifier(pipe, df):
    """Accuracy, per-class precision/recall/F1 and confusion matrix as plain lists."""
    pred = pipe.predict(df["text"])
    y = df["label"].values
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(data.N_CLASSES), zero_division=0
    )
    return {
        "accuracy": float((pred == y).mean()),
        "per_class": [
            {
                "class": data.CLASS_NAMES[k],
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k in range(data.N_CLASSES)
        ],
        "confusion_matrix": confusion_matrix(
            y, pred, labels=range(data.N_CLASSES)
        ).tolist(),
    }
