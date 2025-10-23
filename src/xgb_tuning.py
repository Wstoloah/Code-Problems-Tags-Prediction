import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
from joblib import dump
from scipy.sparse import hstack

from utils.data_loader import load_all_splits
from baseline_model import BaselineTagPredictor, FeatureExtractor, TARGET_TAGS

# Config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

VECTOR_TYPE = "count"  # or "tfidf"
USE_STATS_FEATURES = False
MODEL_OUT = "models/xgb_tuned_model.pkl"

TARGET_TAGS = [
    'math', 'graphs', 'strings', 'number theory',
    'trees', 'geometry', 'games', 'probabilities'
]

# Load data
train_df, val_df, _ = load_all_splits(DATA_ROOT)

def build_texts(df, use_code=True):
    if use_code:
        return (df["description"].fillna("") + "\n\n" + df["code"].fillna("")).tolist()
    else:
        return df["description"].fillna("").tolist()

train_texts = build_texts(train_df, use_code=True)
val_texts = build_texts(val_df, use_code=True)
y = train_df["tags"]

# Prepare multi-label targets
mlb = MultiLabelBinarizer(classes=TARGET_TAGS)
Y_train = mlb.fit_transform(train_df["tags"])
Y_val = mlb.transform(val_df["tags"])

# Vectorizer

vectorizer = FeatureExtractor(vectorizer_type=VECTOR_TYPE, max_features_text=20000)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

# Optional: add statistical features
if USE_STATS_FEATURES:
    stat_train = vectorizer.extract_statistical_features(train_texts)
    from scipy.sparse import hstack, csr_matrix
    X_train = hstack([X_train, csr_matrix(stat_train)])
    
    stat_val = vectorizer.extract_statistical_features(val_texts)
    X_val = hstack([X_val, csr_matrix(stat_val)])


# --- RandomizedSearchCV---
param_grid = {
    "estimator__n_estimators": [300, 500, 800],
    "estimator__max_depth": [4, 6, 8, 10],
    "estimator__learning_rate": [0.05, 0.1, 0.2],
    "estimator__subsample": [0.7, 0.9, 1.0],
    "estimator__colsample_bytree": [0.6, 0.8, 1.0],
    "estimator__min_child_weight": [1, 3, 5],
    "estimator__gamma": [0, 0.1, 0.2],
}



base_clf = OneVsRestClassifier(XGBClassifier(eval_metric="logloss"), n_jobs=-1)
search = RandomizedSearchCV(base_clf, param_grid, cv=3, n_iter=20, scoring="f1_micro", n_jobs=-1, verbose=1)
search.fit(X_train, Y_train)

print("\n=== Best Parameters ===")
print(search.best_params_)
print("\n=== Best F1 (micro) ===")
print(search.best_score_)

# Evaluate on validation
y_pred = search.best_estimator_.predict(X_val)
print("\n=== Validation Results ===")
print(classification_report(Y_val, y_pred, target_names=TARGET_TAGS, digits=4))

# --- Wrap into BaselineTagPredictor and save ---
predictor = BaselineTagPredictor(target_tags=TARGET_TAGS, classifier=search.best_estimator_, vectorizer_type=VECTOR_TYPE)
predictor.feature_extractor = vectorizer # keep vectorizer & stats
predictor.use_code = True  # if code concatenation is used

os.makedirs(os.path.dirname(MODEL_OUT) or ".", exist_ok=True)
predictor.save(MODEL_OUT)
print(f"\n==== Saved tuned predictor to: {MODEL_OUT} ====")
