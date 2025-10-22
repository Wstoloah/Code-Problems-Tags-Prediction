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

from utils.logger import ExperimentLogger
from utils.data_loader import load_all_splits, load_dataset_split
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    hamming_loss, precision_recall_fscore_support,
    classification_report, multilabel_confusion_matrix
)
import click

# Target tags
TARGET_TAGS = [
    'math', 'graphs', 'strings', 'number theory', 
    'trees', 'geometry', 'games', 'probabilities'
]

class FeatureExtractor:
    """Extract and combine features from problem descriptions and code"""
    
    def __init__(self, max_features_text=8000):
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features_text,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        self.fitted = False
        
    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Extract statistical features from combined text"""
        features = []
        
        for text in texts:
            if not text or pd.isna(text):
                text = ""
            
            text_lower = text.lower()
            
            # Keywords indicating categories
            feature_dict = {
                # Math indicators
                'has_math_kw': int(any(kw in text_lower for kw in 
                    ['prime', 'divisor', 'factorial', 'gcd', 'lcm', 'modulo', 'fibonacci'])),
                'has_formula': int(any(kw in text_lower for kw in 
                    ['formula', 'equation', 'calculate', 'sum', 'product'])),
                
                # Graph indicators
                'has_graph_kw': int(any(kw in text_lower for kw in 
                    ['graph', 'node', 'edge', 'vertex', 'tree', 'path', 'cycle'])),
                'has_graph_algo': int(any(kw in text_lower for kw in 
                    ['dfs', 'bfs', 'dijkstra', 'shortest path', 'spanning tree'])),
                
                # String indicators
                'has_string_kw': int(any(kw in text_lower for kw in 
                    ['string', 'substring', 'palindrome', 'character', 'word', 'text'])),
                'has_string_algo': int(any(kw in text_lower for kw in 
                    ['pattern', 'match', 'search', 'replace', 'concatenate'])),
                
                # Geometry indicators
                'has_geometry_kw': int(any(kw in text_lower for kw in 
                    ['point', 'line', 'circle', 'triangle', 'polygon', 'distance', 'angle', 'coordinate'])),
                
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
                
                # Text statistics
                'text_length': len(text),
                'num_numbers': sum(c.isdigit() for c in text),
                'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            }
            
            features.append(list(feature_dict.values()))
        
        return np.array(features)
    
    def fit(self, texts: List[str]):
        """Fit vectorizers on training data"""
        self.text_vectorizer.fit(texts)
        self.fitted = True
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform text data to feature vectors"""
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        # TF-IDF features
        tfidf_features = self.text_vectorizer.transform(texts)
        
        # Statistical features
        stat_features = self.extract_statistical_features(texts)
        
        # Combine all features
        from scipy.sparse import hstack, csr_matrix
        combined = hstack([tfidf_features, csr_matrix(stat_features)])
        
        return combined
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)


class BaselineTagPredictor:
    """Multi-label classifier for algorithm problem tags"""

    def __init__(self, target_tags: Optional[List[str]] = None, classifier=None):
        """
        Args:
            target_tags: Optional list of tag names.
            classifier: Optional sklearn-style classifier (OneVsRestClassifier, etc.)
        """
        self.TARGET_TAGS = target_tags or TARGET_TAGS
        self.feature_extractor = FeatureExtractor()
        self.classifier = classifier  # external classifier injection
        self.all_tags = None
        self.tag_to_idx = None
        self.idx_to_tag = None

    def prepare_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Convert tag lists to binary matrix"""
        all_tags_set = set()
        for tags in df["tags"]:
            if isinstance(tags, list):
                all_tags_set.update(tags)

        self.all_tags = TARGET_TAGS
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(TARGET_TAGS)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}

        y = np.zeros((len(df), len(TARGET_TAGS)))
        for i, tags in enumerate(df["tags"]):
            if isinstance(tags, list):
                for tag in tags:
                    if tag in self.tag_to_idx:
                        y[i, self.tag_to_idx[tag]] = 1

        return y, TARGET_TAGS

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the model"""
        print("==== Preparing data... ====")
        train_texts = train_df["text"].fillna("").tolist()
        y_train, tags = self.prepare_labels(train_df)

        print(f"===> Training set: {len(train_df)} examples, {len(tags)} tags")
        print(f"===> Target tags: {len([t for t in self.TARGET_TAGS if t in tags])}/{len(self.TARGET_TAGS)}")

        # Show tag distribution
        tag_counts = y_train.sum(axis=0)
        print("\n==== Tag distribution (Target tags): ====")
        for tag in self.TARGET_TAGS:
            if tag in self.tag_to_idx:
                idx = self.tag_to_idx[tag]
                count = int(tag_counts[idx])
                print(f"  {tag:<20} {count:>5} ({count/len(train_df)*100:.1f}%)")

        print("\n==== Extracting features... ====")
        X_train = self.feature_extractor.fit_transform(train_texts)
        print(f"==== Feature vector size: {X_train.shape[1]} ====")

        # use injected classifier or default LogisticRegression
        if self.classifier is None:
            print("\n[!] No classifier specified — using default Logistic Regression.")
            self.classifier = OneVsRestClassifier(
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                ),
                n_jobs=-1,
            )

        print(f"\n==== Training model: {type(self.classifier.estimator).__name__} ====")
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"==== Training completed in {train_time:.2f}s ====")

        # Evaluate on validation set
        if val_df is not None:
            print("\n==== Evaluation on validation set: ====")
            val_texts = val_df["text"].fillna("").tolist()
            y_val, _ = self.prepare_labels(val_df)
            X_val = self.feature_extractor.transform(val_texts)
            self.evaluate(X_val, y_val)

        return X_train, y_train
    
    def predict(self, texts: List[str], threshold: float = 0.5) -> List[List[str]]:
        """Predict tags for new problems"""
        if self.classifier is None:
            raise ValueError("Model must be trained before prediction")

        X = self.feature_extractor.transform(texts)
        y_prob = self.classifier.predict_proba(X)

        predictions = []
        for probs in y_prob:
            pred_tags = [
                self.idx_to_tag[idx]
                for idx, prob in enumerate(probs)
                if prob >= threshold
            ]
            predictions.append(pred_tags)
        return predictions

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.classifier is None:
            raise ValueError("Model must be trained before prediction")

        X = self.feature_extractor.transform(texts)
        return self.classifier.predict_proba(X)

    def evaluate(self, X, y_true, threshold: float = 0.5):
        """Evaluate model performance"""
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

        # Per-tag detail
        print("\n==== Target Tags Performance ====")
        print(f"{'Tag':<20}{'Prec':>10}{'Rec':>10}{'F1':>10}{'Support':>10}{'Pred':>10}")
        for tag in self.TARGET_TAGS:
            if tag in self.tag_to_idx:
                idx = self.tag_to_idx[tag]
                yt, yp = y_true[:, idx], y_pred[:, idx]
                support, predicted = int(yt.sum()), int(yp.sum())
                if support > 0:
                    p, r, f, _ = precision_recall_fscore_support(
                        yt, yp, average="binary", zero_division=0
                    )
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
    
    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            "feature_extractor": self.feature_extractor,
            "classifier": self.classifier,
            "all_tags": self.all_tags,
            "tag_to_idx": self.tag_to_idx,
            "idx_to_tag": self.idx_to_tag,
            "TARGET_TAGS": self.TARGET_TAGS,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"==== Model saved to {path} ====")

    def load(self, path: str):
        """Load model from disk"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        self.feature_extractor = model_data["feature_extractor"]
        self.classifier = model_data["classifier"]
        self.all_tags = model_data["all_tags"]
        self.tag_to_idx = model_data["tag_to_idx"]
        self.idx_to_tag = model_data["idx_to_tag"]
        self.TARGET_TAGS = model_data["TARGET_TAGS"]
        print(f"==== Model loaded from {path} ====")

# Classifier choice
CLASSIFIERS = {
    "logistic": lambda: OneVsRestClassifier(LogisticRegression(max_iter=2000), n_jobs=-1),
    "random_forest": lambda: OneVsRestClassifier(
        RandomForestClassifier(n_estimators=500, max_depth=10), n_jobs=-1
    ),
    "xgboost": lambda: OneVsRestClassifier(
        XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        n_jobs=-1
    ),
}

@click.group()
def cli():
    """Tag Predictor CLI for Codeforces Problems"""
    pass

@cli.command()
@click.argument('data_root', type=click.Path(exists=True))
@click.option('--classifier', type=click.Choice(list(CLASSIFIERS.keys())), default='logistic',
              help='Which classifier to use for training')
@click.option('--model-path', default='models/baseline_model.pkl', help='Path to save the trained model')
@click.option('--use-val', is_flag=True, help='Use validation set for evaluation during training')
def train(data_root, classifier, model_path, use_val):
    """Train the tag prediction model"""
    click.echo(f"==== Training Tag Predictor ({classifier}) ====\n")

    # Load data
    click.echo(f"Loading data from {data_root}...")
    train_df, val_df, _ = load_all_splits(data_root)

    # Initialize classifier
    clf = CLASSIFIERS[classifier]()

    # Train model
    predictor = BaselineTagPredictor(classifier=clf)
    if use_val:
        predictor.train(train_df, val_df=val_df)
    else:
        predictor.train(train_df)

    # Save model
    predictor.save(model_path)
    click.echo(f"\n==== Training complete! Model saved to {model_path} ====")

@cli.command()
@click.argument('data_root', type=click.Path(exists=True))
@click.option('--model-path', default='models/baseline_model.pkl', help='Path to the trained model')
@click.option('--split', default='test', type=click.Choice(['train', 'val', 'test']))
@click.option('--output', default='predictions.json')
@click.option('--threshold', default=0.5)
def predict(data_root, model_path, split, output, threshold):
    """Predict tags for problems in a dataset split"""
    click.echo(f"==== Predicting Tags on {split} set ====\n")

    # Load model
    predictor = BaselineTagPredictor()
    predictor.load(model_path)

    # Load data
    df = load_dataset_split(os.path.join(data_root, split))
    texts = df['text'].fillna("").tolist()

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
            "true_tags": row.get('tags', []),
            "text_preview": row['text'][:200] + "..."
        })

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    click.echo(f"\n===> Predictions saved to {output}")
    click.echo(f"===> Average prediction time: {elapsed/len(df)*1000:.2f} ms/sample")
    click.echo("\n==== Prediction complete! ====")

@cli.command()
@click.argument('data_root', type=click.Path(exists=True))
@click.option('--model-path', default='models/baseline_model.pkl')
@click.option('--split', default='test', type=click.Choice(['train', 'val', 'test']))
@click.option('--threshold', default=0.5)
@click.option('--log-path', default='results.md', help='Path to the markdown results log')
@click.option('--notes', default='', help='Optional notes for this experiment')
def evaluate(data_root, model_path, split, threshold, log_path, notes):
    """Evaluate model on a dataset split"""
    click.echo(f"==== Evaluating on {split} set ====\n")

    predictor = BaselineTagPredictor()
    predictor.load(model_path)

    df = load_dataset_split(os.path.join(data_root, split))
    texts = df['text'].fillna("").tolist()
    X = predictor.feature_extractor.transform(texts)
    y_true, _ = predictor.prepare_labels(df)

    click.echo(f"Evaluating {len(df)} examples...\n")
    metrics = predictor.evaluate(X, y_true, threshold=threshold)

    # Log results in Markdown
    classifier_name = type(predictor.classifier.estimator).__name__
    embedding_name = getattr(predictor, "embedding_name", "TF-IDF")  # optional attribute

    logger = ExperimentLogger(log_path)
    logger.log_result(
        model_name="BaselineTagPredictor",
        embedding=embedding_name,
        classifier=classifier_name,
        dataset=split,
        metrics=metrics,
        notes=notes
    )

    click.echo("\n==== Evaluation complete! ====")
    return metrics

@cli.command()
@click.argument('text', type=str)
@click.option('--model-path', default='models/baseline_model.pkl')
@click.option('--threshold', default=0.5)
def predict_one(text, model_path, threshold):
    """Predict tags for a single problem description"""
    click.echo("==== Predicting tags for single problem ====\n")

    predictor = BaselineTagPredictor()
    predictor.load(model_path)

    start = time.time()
    pred_tags = predictor.predict([text], threshold=threshold)[0]
    elapsed = time.time() - start

    click.echo(f"Predicted tags: {pred_tags}")
    click.echo(f"Prediction time: {elapsed*1000:.2f} ms")

if __name__ == '__main__':
    cli()