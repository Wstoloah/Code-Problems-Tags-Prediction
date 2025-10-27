"""
ModernBERT / CodeBERT Tag Predictor (Embedding-Based)
----------------------------------------------------
- Multi-label tag predictor using frozen ModernBERT or CodeBERT embeddings
- Supports caching, chunking, and optional code concatenation
- No fine-tuning: uses embeddings + classic ML classifier (LogReg, RF, XGB, NB, MLP)
"""

import os
import time
import json
import pickle
import hashlib
from typing import List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import torch
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, hamming_loss, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel

from utils.data_loader import load_all_splits, load_dataset_split
from utils.logger import ExperimentLogger

# Config
TARGET_TAGS = [
    "math", "graphs", "strings", "number theory",
    "trees", "geometry", "games", "probabilities"
]

MODEL_CONFIGS = {
    "modernbert": {
        "model_name": "answerdotai/ModernBERT-base",
        "max_length": 800
    },
    "codebert": {
        "model_name": "microsoft/codebert-base",
        "max_length": 512
    }
}

# Embedder
class Embedder:
    """Embedder wrapper for ModernBERT / CodeBERT"""

    def __init__(self, model_name: str, device=None, cache_dir="cache"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.cache_dir = str(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.npy")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        normalize: bool = True,
        use_cache: bool = True,
        chunk: bool = True,
        overlap: int = 50
    ) -> np.ndarray:
        """Encode texts with optional chunking and caching"""
        cache_key = hashlib.sha1(("".join(texts) + f"{self.model.config._name_or_path}").encode()).hexdigest()
        cache_file = self._cache_path(cache_key)

        if use_cache and os.path.exists(cache_file):
            print(f"===> Loading embeddings from cache: {cache_file}")
            return np.load(cache_file)

        all_embs = []
        for text in tqdm(texts, desc="Encoding texts"):
            tokens = self.tokenizer.tokenize(text)
            if not chunk or len(tokens) <= max_length - 2:
                chunks = [text]
            else:
                chunks = []
                start = 0
                while start < len(tokens):
                    end = min(start + max_length - 2, len(tokens))
                    chunk_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                    chunks.append(chunk_text)
                    start += max_length - overlap

            chunk_embs = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = summed / counts
                if normalize:
                    mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                chunk_embs.append(mean_pooled.cpu().numpy())

            text_emb = np.mean(np.vstack(chunk_embs), axis=0)
            all_embs.append(text_emb)

        embeddings = np.vstack(all_embs)
        if use_cache:
            np.save(cache_file, embeddings)
            print(f"===> Saved embeddings to cache: {cache_file}")
        return embeddings

# Predictor
class TagPredictor:
    """Embedding-based multi-label tag predictor"""

    def __init__(self, model_name: str = None, classifier_name: str = "logistic", cache_dir: str = "cache", target_tags = None):
        self.TARGET_TAGS = target_tags or TARGET_TAGS
        self.all_tags = None
        self.tag_to_idx = None
        self.idx_to_tag = None
        self.use_code = False
        
        # Only initialize embedder and classifier if model_name is provided
        if model_name is not None:
            if model_name not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model: {model_name}")
            self.model_name = model_name
            model_cache_dir = os.path.join(cache_dir, model_name)
            self.embedder = Embedder(MODEL_CONFIGS[model_name]['model_name'], cache_dir=model_cache_dir)
            self.max_length = MODEL_CONFIGS[model_name]['max_length']
            self.mlb = MultiLabelBinarizer(classes=TARGET_TAGS)

            self.classifier = {
                "logistic": OneVsRestClassifier(LogisticRegression(max_iter=2000), n_jobs=-1),
                "random_forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=300, max_depth=10), n_jobs=-1),
                "xgboost": OneVsRestClassifier(XGBClassifier(n_estimators=200, max_depth=6, eval_metric="logloss"), n_jobs=-1),
                "naive_bayes": OneVsRestClassifier(MultinomialNB()),
                "mlp": OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(512,), max_iter=500, random_state=42))
            }[classifier_name]
        else:
            # When loading, these will be set by load() method
            self.model_name = None
            self.embedder = None
            self.max_length = None
            self.mlb = None
            self.classifier = None

    def prepare_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Build y matrix of shape (n_samples, len(TARGET_TAGS)).
        df expected to have either 'filtered_tags' or 'tags' column (list).
        We **do not** drop samples that lack target tags, we include them with all-zero labels.
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

    def _build_texts_from_df(self, df: pd.DataFrame):
        descs = df["description"].fillna("").tolist()
        if self.use_code and "code" in df.columns:
            codes = df["code"].fillna("").tolist()
            return [f"{d}\n\n{c}" for d, c in zip(descs, codes)]
        return descs

    def train(self, df_train, df_val=None, batch_size=8, use_cache=True, log_path="results.md"):

        print("==== Preparing data... ====")
        texts = self._build_texts_from_df(df_train)
        y_train, tags = self.prepare_labels(df_train)

        print(f"===> Training set: {len(df_train)} examples, {len(tags)} tags (target tags)")
        print(f"===> Target tags present in dataset: {sum(y_train.sum(axis=0) > 0)}/{len(tags)}")

        # Build texts according to use_code flag
        print("==== Encoding texts... ====")
        X_train = self.embedder.encode_texts(texts, batch_size=batch_size, use_cache=use_cache, max_length=self.max_length)

        print(f"\n==== Training model: {type(self.classifier.estimator).__name__} ====")
        start = time.time()
        self.classifier.fit(X_train, y_train)
        print(f"==== Training completed in {time.time()-start:.2f}s ====")
        if df_val is not None:
            print("\n==== Evaluation on validation set: ====")
            self.evaluate(df_val, batch_size=batch_size, use_cache=use_cache, log_path=log_path, notes="Validation set")

    def predict(self, texts, threshold=0.5, batch_size=8, use_cache=True):
        X = self.embedder.encode_texts(texts, batch_size=batch_size, use_cache=use_cache, max_length=self.max_length)
        y_prob = self.classifier.predict_proba(X)
        if isinstance(y_prob, list):
            y_prob = np.vstack([p[:, 1] for p in y_prob]).T

        predictions = []
        for probs in y_prob:
            pred_tags = [self.idx_to_tag[idx] for idx, prob in enumerate(probs) if prob >= threshold]
            predictions.append(pred_tags)
        return predictions

    def evaluate(self, df, threshold=0.5, batch_size=8, use_cache=True, log_path="results.md", notes=""):
        """Evaluate model on a dataset and print metrics."""
        texts = self._build_texts_from_df(df)

        X = self.embedder.encode_texts(texts, batch_size=batch_size, use_cache=use_cache, max_length=self.max_length)
        y_true, _ = self.prepare_labels(df)

        y_prob = self.classifier.predict_proba(X)
        if isinstance(y_prob, list):
            y_prob = np.vstack([p[:, 1] for p in y_prob]).T
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

    def evaluate_per_tag(self, X, y_true, threshold=0.5):
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
        """Save model data (classifier, embedder config, mappings) to file"""
        model_data = {
            "classifier": self.classifier,
            "all_tags": self.all_tags,
            "tag_to_idx": self.tag_to_idx,
            "idx_to_tag": self.idx_to_tag,
            "TARGET_TAGS": self.TARGET_TAGS,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "use_code": self.use_code,
            "cache_dir": self.embedder.cache_dir
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"==== Model saved to {path} ====")

    def load(self, path: str):
        """Load model data from file"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        self.classifier = model_data["classifier"]
        self.all_tags = model_data.get("all_tags", None)
        self.tag_to_idx = model_data.get("tag_to_idx", None)
        self.idx_to_tag = model_data.get("idx_to_tag", None)
        self.TARGET_TAGS = model_data.get("TARGET_TAGS", self.TARGET_TAGS)
        self.model_name = model_data.get("model_name", self.model_name)
        self.max_length = model_data.get("max_length", 512)
        self.use_code = model_data.get("use_code", False)
        cache_dir = model_data.get("cache_dir", "cache_" + self.model_name)
        
        # Reinitialize embedder with saved config
        self.embedder = Embedder(MODEL_CONFIGS[self.model_name]['model_name'], cache_dir=cache_dir)
        print(f"==== Model loaded from {path} (model={self.model_name}, use_code={self.use_code}) ====")

@click.group()
def cli():
    pass

@cli.command()
@click.argument("data_root", type=click.Path(exists=True))
@click.option("--model", type=click.Choice(["codebert", "modernbert"]), default="codebert")
@click.option("--classifier", type=click.Choice(["logistic", "random_forest", "xgboost", "naive_bayes", "mlp"]), default="logistic")
@click.option("--model-path", default="models/bert_based/embedding_model.pkl")
@click.option("--batch-size", default=8)
@click.option("--use-code", is_flag=True)
@click.option("--use-cache", is_flag=True)
def train(data_root, model, classifier, model_path, batch_size, use_code, use_cache):
    train_df, val_df, _ = load_all_splits(data_root)
    predictor = TagPredictor(model_name=model, classifier_name=classifier)
    predictor.use_code = use_code
    predictor.train(train_df, df_val=val_df, batch_size=batch_size, use_cache=use_cache)
    predictor.save(model_path)

@cli.command()
@click.argument("data_root", type=click.Path(exists=True))
@click.option("--split", default="test", type=click.Choice(["train", "val", "test"]))
@click.option("--model-path", default="models/embedding_model.pkl")
@click.option("--batch-size", default=8, type=int, help="Batch size for encoding (default: 8)")
@click.option("--threshold", default=0.5, help="Threshold for tag prediction")
@click.option('--notes', default='', help='Optional notes for this experiment')
@click.option("--use-cache", is_flag=True, help= "Embeddings loading from/to cache")
@click.option('--log-path', default='outputs/logs/results.md', help='Path to the markdown results log')
def evaluate(data_root, split, model_path, batch_size, threshold, notes, use_cache, log_path):
    """Evaluate model on a dataset split"""
    click.echo(f"==== Evaluating on {split} set ====\n")
    predictor = TagPredictor()
    predictor.load(model_path)

    click.echo(f"Model configuration: model={predictor.model_name}, use_code={predictor.use_code}, max_length={predictor.max_length}")
    click.echo(f"Encoding settings: batch_size={batch_size}, use_cache={use_cache}\n")

    df = load_dataset_split(os.path.join(data_root, split))
    texts = predictor._build_texts_from_df(df)

    X = predictor.embedder.encode_texts(texts, batch_size=batch_size, use_cache=use_cache, max_length=predictor.max_length)
    y_true, _ = predictor.prepare_labels(df)

    click.echo(f"Evaluating {len(df)} examples...\n")
    metrics = predictor.evaluate_per_tag(X, y_true, threshold=threshold)

    # Log results
    classifier_name = type(predictor.classifier.estimator).__name__ if predictor.classifier is not None else "None"
    embedding_name = predictor.model_name

    logger = ExperimentLogger(log_path)
    logger.log_result(
        model_name=predictor.model_name+"BasedTagPredictor",
        embedding=embedding_name,
        use_code=predictor.use_code,
        use_stats_features=False,
        classifier=classifier_name,
        dataset=split,
        metrics=metrics,
        notes=notes
    )

    click.echo("\n==== Evaluation complete! ====")
    return metrics


@cli.command()
@click.argument("data_root", type=click.Path(exists=True))
@click.option("--model-path", default="models/embedding_model.pkl")
@click.option("--split", default="test", type=click.Choice(["train", "val", "test"]))
@click.option("--output", default="outputs/predictions/predictions_embedding.json")
@click.option("--threshold", default=0.5, help="Threshold for tag prediction")
@click.option("--batch-size", default=8, type=int, help="Batch size for encoding (default: 8)")
@click.option("--use-cache", is_flag=True, help= "Embeddings loading from/to cache")
def predict(data_root, model_path, split, output, threshold, batch_size, use_cache):
    """Predict tags for problems in a dataset split"""
    click.echo(f"==== Predicting Tags on {split} set using embedding model ====\n")

    predictor = TagPredictor()
    predictor.load(model_path)


    click.echo(f"Model configuration: model={predictor.model_name}, use_code={predictor.use_code}, max_length={predictor.max_length}")
    click.echo(f"Encoding settings: batch_size={batch_size}, use_cache={use_cache}\n")

    df = load_dataset_split(os.path.join(data_root, split))

    texts = predictor._build_texts_from_df(df)

    click.echo(f"Making predictions on {len(df)} examples...")
    start = time.time()
    predictions = predictor.predict(texts, threshold=threshold, batch_size=batch_size, use_cache=use_cache)
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
@click.argument("text", type=str)
@click.option("--model-path", default="models/embedding_model.pkl")
@click.option("--threshold", default=0.5, help="Threshold for tag prediction")
@click.option("--use-cache", is_flag=True, help= "Embeddings loading from/to cache")
def predict_one(text, model_path, threshold, use_cache):
    """Predict tags for a single problem description"""
    predictor = TagPredictor()
    predictor.load(model_path)
    
    click.echo(f"==== Predicting single sample ====")
    click.echo(f"Model configuration: model={predictor.model_name}, use_code={predictor.use_code}, max_length={predictor.max_length}")
    click.echo(f"Encoding settings: use_cache={use_cache}\n")
    
    start = time.time()
    tags = predictor.predict([text], threshold=threshold, use_cache=use_cache)[0]
    elapsed = time.time() - start
    
    click.echo(f"Predicted tags: {tags}")
    click.echo(f"Prediction time: {elapsed*1000:.2f} ms")

if __name__ == "__main__":
    cli()
