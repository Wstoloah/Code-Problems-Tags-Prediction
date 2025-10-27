"""
Baseline Tag Predictor for Codeforces Problems
Multi-label classification system for algorithmic problem tags
"""

import json
import pickle
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

from src.utils.logger import ExperimentLogger
from src.utils.data_loader import load_all_splits, load_dataset_split
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    hamming_loss, precision_recall_fscore_support,
    classification_report, multilabel_confusion_matrix
)
import click

# Target tags (these are the ones we train/evaluate for)
TARGET_TAGS = [
    'math', 'graphs', 'strings', 'number theory',
    'trees', 'geometry', 'games', 'probabilities'
]

# max features number in vectorizer
MAX_FEATURES_TEXT = 20000 # based on EDA using CountVectorizer and TF-IDF Vectorizer

class FeatureExtractor:
    """Text feature extractor using TF-IDF or CountVectorizer, optional statistical features."""

    def __init__(self, vectorizer_type: str = "tfidf", max_features_text: int = MAX_FEATURES_TEXT):
        self.vectorizer_type = vectorizer_type.lower()
        if self.vectorizer_type == "tfidf":
            self.text_vectorizer = TfidfVectorizer(
                max_features=max_features_text,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2,
                max_df=0.85,
                sublinear_tf=True
            )
        elif self.vectorizer_type == "count":
            self.text_vectorizer = CountVectorizer(
                max_features=max_features_text,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2,
                max_df=0.85
            )
        else:
            raise ValueError("vectorizer_type must be either 'tfidf' or 'count'")

        self.fitted = False

    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Statistical and heuristics features (works on combined text or on description/code)."""
        features = []
        for text in texts:
            if not text or pd.isna(text):
                text = ""
            text_lower = text.lower()

            feature_dict = {
                # Math indicators
                'has_math_kw': int(any(kw in text_lower for kw in
                                      ['prime', 'divisor', 'factorial', 'gcd', 'lcm', 'modulo', 'fibonacci', 'math'])),
                'has_formula': int(any(kw in text_lower for kw in
                                       ['formula', 'equation', 'calculate', 'sum', 'product'])),

                # Graph indicators
                'has_graph_kw': int(any(kw in text_lower for kw in
                                        ['graph', 'node', 'edge', 'vertex', 'tree', 'path', 'cycle'])),
                'has_graph_algo': int(any(kw in text_lower for kw in
                                          ['dfs', 'bfs', 'dijkstra', 'shortest path', 'spanning tree', 'graph'])),

                # String indicators
                'has_string_kw': int(any(kw in text_lower for kw in
                                         ['string', 'substring', 'palindrome', 'character', 'word', 'text'])),
                'has_string_algo': int(any(kw in text_lower for kw in
                                           ['pattern', 'match', 'search', 'replace', 'concatenate'])),

                # Geometry indicators
                'has_geometry_kw': int(any(kw in text_lower for kw in
                                           ['point', 'line', 'circle', 'triangle', 'polygon', 'distance', 'angle',
                                            'coordinate', 'geometry'])),

                # Probability indicators
                'has_prob_kw': int(any(kw in text_lower for kw in
                                       ['probability', 'expected', 'random', 'chance', 'outcome'])),

                # Game theory indicators
                'has_game_kw': int(any(kw in text_lower for kw in
                                       ['game', 'player', 'win', 'lose', 'strategy', 'turn', 'move'])),

                # Code structure indicators
                'has_loop': text.count('for') + text.count('while'),
                'has_recursion': int('def' in text and any(
                    kw in text_lower for kw in ['recursive', 'recursion'])),
                'has_sort': int('sort' in text_lower),
                'num_functions': text.count('def '),

                # # Text statistics
                # 'text_length': len(text),
                # 'num_numbers': sum(c.isdigit() for c in text),
                # 'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            }
            features.append(list(feature_dict.values()))

        return np.array(features)

    def fit(self, texts: List[str]):
        """Fit vectorizer on training texts."""
        self.text_vectorizer.fit(texts)
        self.fitted = True

    def transform(self, texts: List[str], use_stats_features: bool = False):
        """Transform texts to vectorized features (sparse matrix)."""
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        X_text = self.text_vectorizer.transform(texts)

        if use_stats_features:
            stat_features = self.extract_statistical_features(texts)
            from scipy.sparse import hstack, csr_matrix
            combined = hstack([X_text, csr_matrix(stat_features)])
            return combined

        return X_text

    def fit_transform(self, texts: List[str], use_stats_features: bool = False):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts, use_stats_features)


class BaselineTagPredictor:
    """Multi-label classifier for algorithm problem tags"""

    def __init__(self, target_tags: Optional[List[str]] = None,
                 classifier=None, vectorizer_type: str = "tfidf", use_code: bool = False, use_stats_features: bool = False):
        """
        Args:
            target_tags: list of tags to train/predict (columns order).
            classifier: sklearn-style classifier (OneVsRestClassifier wrapper recommended).
            vectorizer_type: 'tfidf' or 'count'.
            use_code: whether code is included as part of text features.
        """
        self.TARGET_TAGS = target_tags or TARGET_TAGS
        self.feature_extractor = FeatureExtractor(vectorizer_type=vectorizer_type)
        self.embedding_name = vectorizer_type.upper()
        self.classifier = classifier
        self.use_code = use_code
        self.use_stats_features = use_stats_features
        self.all_tags = None  # all tags seen in dataset (kept for EDA / future use)
        self.tag_to_idx = None
        self.idx_to_tag = None

    # Labels: build binary matrix only for TARGET_TAGS
    def prepare_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Build y matrix of shape (n_samples, len(TARGET_TAGS)).
        df expected to have either 'filtered_tags' or 'tags' column (list).
        We **do not** drop samples that lack target tags — we include them with all-zero labels.
        """
        self.all_tags = sorted({t for tags in df["tags"] for t in (tags or [])})
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.TARGET_TAGS)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}

        y = np.zeros((len(df), len(self.TARGET_TAGS)), dtype=int)
        for i, tags in enumerate(df["tags"]):
            if not isinstance(tags, list):
                continue
            for t in tags:
                if t in self.tag_to_idx:
                    y[i, self.tag_to_idx[t]] = 1
        return y, self.TARGET_TAGS

    # Helper to build text features (description +/- code)
    def _build_texts_from_df(self, df: pd.DataFrame) -> List[str]:
        """
        Build the string list that will be passed to vectorizer.
        Uses 'description' column and optionally 'code' column.
        If df has a 'text' column already, prefer that for backward compatibility.
        """
        if "text" in df.columns:
            return df["text"].fillna("").tolist()

        descs = df["description"].fillna("").tolist() if "description" in df.columns else [""] * len(df)
        if self.use_code and "code" in df.columns:
            codes = df["code"].fillna("").tolist()
            return [d.strip() + "\n\n" + c.strip() for d, c in zip(descs, codes)]
        else:
            return [d.strip() for d in descs]

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the classifier on training DataFrame (keeps samples even if no TARGET_TAGS)."""
        print("==== Preparing data... ====")

        # Build texts according to use_code flag
        train_texts = self._build_texts_from_df(train_df)
        y_train, tags = self.prepare_labels(train_df)

        print(f"===> Training set: {len(train_df)} examples, {len(tags)} tags (target tags)")
        print(f"===> Target tags present in dataset: {sum(y_train.sum(axis=0) > 0)}/{len(tags)}")

        # # Tag distribution (for target tags only)
        # tag_counts = y_train.sum(axis=0)
        # print("\n==== Tag distribution (Target tags): ====")
        # for tag in self.TARGET_TAGS:
        #     idx = self.tag_to_idx[tag]
        #     count = int(tag_counts[idx])
        #     print(f"  {tag:<20} {count:>5} ({count/len(train_df)*100:.1f}%)")

        print("\n==== Extracting features... ====")
        X_train = self.feature_extractor.fit_transform(train_texts, use_stats_features=self.use_stats_features)
        print(f"==== Feature vector size: {X_train.shape[1]} ====")

        # Default classifier fallback
        if self.classifier is None:
            print("\n[!] No classifier specified — using default Logistic Regression (One-vs-rest).")
            self.classifier = OneVsRestClassifier(
                LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42, solver="lbfgs"),
                n_jobs=-1
            )

        print(f"\n==== Training model: {type(self.classifier.estimator).__name__} ====")
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"==== Training completed in {train_time:.2f}s ====")

        # Evaluate on validation set if provided
        if val_df is not None:
            print("\n==== Evaluation on validation set: ====")
            val_texts = self._build_texts_from_df(val_df)
            y_val, _ = self.prepare_labels(val_df)
            X_val = self.feature_extractor.transform(val_texts, use_stats_features=self.use_stats_features)
            self.evaluate(X_val, y_val)

        return X_train, y_train

    def predict(self, texts: List[str], threshold: float = 0.5) -> List[List[str]]:
        """Predict tags for raw text inputs (list of strings). Returns list of predicted target tags."""
        if self.classifier is None:
            raise ValueError("Model must be trained before prediction")
        X = self.feature_extractor.transform(texts, use_stats_features=self.use_stats_features)
        y_prob = self.classifier.predict_proba(X)

        predictions = []
        for probs in y_prob:
            pred_tags = [self.idx_to_tag[idx] for idx, prob in enumerate(probs) if prob >= threshold]
            predictions.append(pred_tags)
        return predictions

    def predict_from_df(self, df: pd.DataFrame, threshold: float = 0.5) -> List[List[str]]:
        texts = self._build_texts_from_df(df)
        return self.predict(texts, threshold=threshold, use_stats_features=self.use_stats_features)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.classifier is None:
            raise ValueError("Model must be trained before prediction")
        X = self.feature_extractor.transform(texts, use_stats_features=self.use_stats_features)
        return self.classifier.predict_proba(X)

    def evaluate(self, X, y_true, threshold: float = 0.5):
        """Evaluate and print metrics. y_true must match TARGET_TAGS order."""
        y_prob = self.classifier.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)

        # Global metrics
        hamming = hamming_loss(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="samples", zero_division=0
        )

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        print(f"  Hamming Loss: {hamming:.4f}")
        print(f"  Precision (samples avg): {precision:.4f}")
        print(f"  Recall (samples avg): {recall:.4f}")
        print(f"  F1 (samples avg): {f1:.4f}")
        print(f"  F1 (macro avg): {f1_macro:.4f}")

        # Per-tag detail for TARGET_TAGS
        print("\n==== Target Tags Performance ====")
        print(f"{'Tag':<20}{'Prec':>10}{'Rec':>10}{'F1':>10}{'Support':>10}{'Pred':>10}")
        for tag in self.TARGET_TAGS:
            idx = self.tag_to_idx[tag]
            yt, yp = y_true[:, idx], y_pred[:, idx]
            support, predicted = int(yt.sum()), int(yp.sum())
            if support > 0:
                p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
                print(f"{tag:<20}{p:>10.4f}{r:>10.4f}{f:>10.4f}{support:>10d}{predicted:>10d}")

        return {
            "hamming_loss": hamming,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        }

    def evaluate_per_tag(self, X, y_true, threshold: float = 0.5):
        """
        Evaluate model and print per-tag metrics table.
        Returns detailed metrics per tag.
        """

        # Predict probabilities and threshold
        y_prob = self.classifier.predict_proba(X)
        if isinstance(y_prob, list):
            y_prob = np.vstack([p[:, 1] for p in y_prob]).T
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics per tag
        from sklearn.metrics import precision_recall_fscore_support, hamming_loss
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        support = y_true.sum(axis=0)
        predicted_count = y_pred.sum(axis=0)

        # Print table
        print(f"\n{'Tag':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}{'Predicted':>10}")
        print("="*70)
        for idx, tag in enumerate(self.TARGET_TAGS):
            print(f"{tag:<20}{precision[idx]:>10.3f}{recall[idx]:>10.3f}{f1[idx]:>10.3f}{support[idx]:>10d}{predicted_count[idx]:>10d}")

        # Global metrics
        hamming = hamming_loss(y_true, y_pred)
        precision_s, recall_s, f1_s, _ = precision_recall_fscore_support(y_true, y_pred, average="samples", zero_division=0)
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        print(f"\nHamming Loss: {hamming:.4f}")
        print(f"Sample-average Precision: {precision_s:.4f}, Recall: {recall_s:.4f}, F1: {f1_s:.4f}")
        print(f"Macro-average Precision: {precision_m:.4f}, Recall: {recall_m:.4f}, F1: {f1_m:.4f}")

        return {
            "per_tag": [{"tag": t, "precision": float(p), "recall": float(r), "f1": float(f),
                        "support": int(s), "predicted": int(pc)}
                        for t, p, r, f, s, pc in zip(self.TARGET_TAGS, precision, recall, f1, support, predicted_count)],
            "hamming_loss": float(hamming),
            "precision_samples": float(precision_s),
            "recall_samples": float(recall_s),
            "f1_samples": float(f1_s),
            "precision_macro": float(precision_m),
            "recall_macro": float(recall_m),
            "f1_macro": float(f1_m)
        }

    def save(self, path: str):
        model_data = {
            "feature_extractor": self.feature_extractor,
            "classifier": self.classifier,
            "all_tags": self.all_tags,
            "tag_to_idx": self.tag_to_idx,
            "idx_to_tag": self.idx_to_tag,
            "TARGET_TAGS": self.TARGET_TAGS,
            "embedding_name": self.embedding_name,
            "use_code": self.use_code,
            "use_stats_features": self.use_stats_features
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"==== Model saved to {path} ====")

    def load(self, path: str):
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        self.feature_extractor = model_data["feature_extractor"]
        self.classifier = model_data["classifier"]
        self.all_tags = model_data.get("all_tags", None)
        self.tag_to_idx = model_data.get("tag_to_idx", None)
        self.idx_to_tag = model_data.get("idx_to_tag", None)
        self.TARGET_TAGS = model_data.get("TARGET_TAGS", self.TARGET_TAGS)
        self.embedding_name = model_data.get("embedding_name", getattr(self, "embedding_name", "TFIDF"))
        self.use_code = model_data.get("use_code", getattr(self, "use_code", False))
        self.use_stats_features = model_data.get("use_stats_features", getattr(self, "use_code", False))
        print(f"==== Model loaded from {path} (embedding={self.embedding_name}, use_code={self.use_code})")


# classifier choice 
CLASSIFIERS = {
    "logistic": lambda: OneVsRestClassifier(
        LogisticRegression(max_iter=2000),
        n_jobs=-1
    ),
    "random_forest": lambda: OneVsRestClassifier(
        RandomForestClassifier(n_estimators=500, max_depth=10),
        n_jobs=-1
    ),
    "xgboost": lambda: OneVsRestClassifier(
        XGBClassifier(
            n_estimators=200, max_depth=6,
            eval_metric="logloss"
        ),
        n_jobs=-1
    ),
    "naive_bayes": lambda: OneVsRestClassifier(
        MultinomialNB(),
        n_jobs=-1
    ),
    "mlp": lambda: OneVsRestClassifier(
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=30,
            random_state=42,
        ),
        n_jobs=-1
    ),
}

# CLI
@click.group()
def cli():
    """Tag Predictor CLI for Codeforces Problems"""
    pass


@cli.command()
@click.argument('data_root', type=click.Path(exists=True))
@click.option('--classifier', type=click.Choice(list(CLASSIFIERS.keys())), default='logistic',
              help='Which classifier to use for training')
@click.option('--model-path', default='models/baseline/baseline_model.pkl', help='Path to save the trained model')
@click.option('--use-val', is_flag=True, help='Use validation set for evaluation during training')
@click.option('--use-stats-features', is_flag=True, help='Include statistical features in training')
@click.option('--vectorizer', type=click.Choice(['tfidf', 'count']), default='tfidf', help='Text embedding type')
@click.option('--use-code', is_flag=True, help='Include code field concatenated with description as feature')
def train(data_root, classifier, model_path, use_val, use_stats_features, vectorizer, use_code):
    """Train the tag prediction model"""
    click.echo(f"==== Training Tag Predictor ({classifier}, vectorizer={vectorizer.upper()}, use_code={use_code}), use_stats_features={use_stats_features} ====\n")

    # Load data (expects JSONs with fields: description, code, original_tags / tags)
    train_df, val_df, _ = load_all_splits(data_root)

    # if load_all_splits returns 'text' instead of 'description'/'code', try to adapt
    for df_ in (train_df, val_df):
        if df_ is not None and "text" in df_.columns and "description" not in df_.columns:
            df_["description"] = df_["text"]

    # Initialize classifier and predictor
    clf = CLASSIFIERS[classifier]()
    predictor = BaselineTagPredictor(classifier=clf, vectorizer_type=vectorizer, use_code=use_code, use_stats_features=use_stats_features)

    if use_val and val_df is not None:
        predictor.train(train_df, val_df=val_df)
    else:
        predictor.train(train_df)

    predictor.save(model_path)
    click.echo(f"\n==== Training complete! Model saved to {model_path} ====")


@cli.command()
@click.argument('data_root', type=click.Path(exists=True))
@click.option('--model-path', default='models/baseline/baseline_model.pkl', help='Path to the trained model')
@click.option('--split', default='test', type=click.Choice(['train', 'val', 'test']))
@click.option('--output', default='outputs/predictions/predictions.json')
@click.option('--threshold', default=0.5)
def predict(data_root, model_path, split, output, threshold):
    """Predict tags for problems in a dataset split"""
    click.echo(f"==== Predicting Tags on {split} set ====\n")

    predictor = BaselineTagPredictor()
    predictor.load(model_path)

    df = load_dataset_split(os.path.join(data_root, split))

    texts = predictor._build_texts_from_df(df)

    click.echo(f"Making predictions on {len(df)} examples...")
    start = time.time()
    predictions = predictor.predict(texts, threshold=threshold)
    elapsed = time.time() - start

    # Save results
    results = []
    for i, (pred_tags, (_, row)) in enumerate(zip(predictions, df.iterrows())):
        results.append({
            "index": i,
            "predicted_tags": pred_tags,
            "true_tags": row.get('tags', row.get('original_tags', [])),
            "description_preview": (row.get('description') or "")[:300] + "...",
            "code_preview": (row.get('code') or "")[:300] + "..."
        })

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    click.echo(f"\n===> Predictions saved to {output}")
    click.echo(f"===> Avg prediction time: {elapsed/len(df)*1000:.2f} ms/sample")
    click.echo("\n==== Prediction complete! ====")


@cli.command()
@click.argument('data_root', type=click.Path(exists=True))
@click.option('--model-path', default='models/baseline_model.pkl')
@click.option('--split', default='test', type=click.Choice(['train', 'val', 'test']))
@click.option('--threshold', default=0.5)
@click.option('--log-path', default='outputs/logs/results.md', help='Path to the markdown results log')
@click.option('--notes', default='', help='Optional notes for this experiment')
def evaluate(data_root, model_path, split, threshold, log_path, notes):
    """Evaluate model on a dataset split"""
    click.echo(f"==== Evaluating on {split} set ====\n")

    predictor = BaselineTagPredictor()
    predictor.load(model_path)

    df = load_dataset_split(os.path.join(data_root, split))
    texts = predictor._build_texts_from_df(df)

    X = predictor.feature_extractor.transform(texts, predictor.use_stats_features)
    y_true, _ = predictor.prepare_labels(df)

    click.echo(f"Evaluating {len(df)} examples...\n")
    # metrics = predictor.evaluate(X, y_true, threshold=threshold)
    metrics = predictor.evaluate_per_tag(X, y_true, threshold=threshold)

    # Log results
    classifier_name = type(predictor.classifier.estimator).__name__ if predictor.classifier is not None else "None"
    embedding_name = getattr(predictor, "embedding_name", "TFIDF")

    logger = ExperimentLogger(log_path)
    logger.log_result(
        model_name="BaselineTagPredictor",
        embedding=embedding_name,
        use_code=predictor.use_code,
        use_stats_features=predictor.use_stats_features,
        classifier=classifier_name,
        dataset=split,
        metrics=metrics,
        notes=notes
    )

    click.echo("\n==== Evaluation complete! ====")
    return metrics


@cli.command()
@click.argument('text', type=str)
@click.option('--model-path', default='models/baseline/baseline_model.pkl')
@click.option('--threshold', default=0.5)
def predict_one(text, model_path, threshold):
    """Predict tags for a single problem description,
    If user needs to include code, they can pass a combined string "desc\n\n<CODE>"
    """
    predictor = BaselineTagPredictor()
    predictor.load(model_path)

    click.echo(f"==== Predicting single sample (vectorizer={predictor.vectorizer}, use_code={predictor.use_code}), use_stats_features={predictor.use_stats_features} ====\n")

    start = time.time()
    pred_tags = predictor.predict([text], threshold=threshold, use_stats_features=predictor.use_stats_features)[0]
    elapsed = time.time() - start

    click.echo(f"Predicted tags: {pred_tags}")
    click.echo(f"Prediction time: {elapsed*1000:.2f} ms")


if __name__ == '__main__':
    cli()
