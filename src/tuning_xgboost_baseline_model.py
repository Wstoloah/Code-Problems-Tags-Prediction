import os
import click
import joblib
from tqdm import tqdm
from contextlib import contextmanager
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

from utils.data_loader import load_all_splits, load_dataset_split
from utils.logger import ExperimentLogger
from baseline_model import BaselineTagPredictor, TARGET_TAGS

# Config
DEFAULT_PARAM_GRID = {
    "estimator__n_estimators": [300, 500, 800],
    "estimator__max_depth": [4, 6, 8, 10],
    "estimator__learning_rate": [0.05, 0.1, 0.2],
    "estimator__subsample": [0.7, 0.9, 1.0],
    "estimator__colsample_bytree": [0.6, 0.8, 1.0],
    "estimator__min_child_weight": [1, 3, 5],
    "estimator__gamma": [0, 0.1, 0.2],
}


# Train and Tune
def tune_and_train(
    data_root,
    model_out,
    use_code=True,
    vector_type="count",
    n_iter=50,
    use_stats=False,
):
    """Train + tune a multi-label XGBoost classifier with RandomizedSearchCV."""
    
    # Load data
    train_df, val_df, _ = load_all_splits(data_root)

    # Initialize predictor components
    predictor = BaselineTagPredictor(
        target_tags=TARGET_TAGS,
        vectorizer_type=vector_type,
        use_code=use_code,
        use_stats_features=use_stats
    )
    
    # Build texts and prepare labels using predictor methods
    train_texts = predictor._build_texts_from_df(train_df)
    val_texts = predictor._build_texts_from_df(val_df)
    
    # Prepare targets
    Y_train, _ = predictor.prepare_labels(train_df)
    Y_val, _ = predictor.prepare_labels(val_df)

    # Feature extraction
    X_train = predictor.feature_extractor.fit_transform(train_texts, use_stats_features=use_stats)
    X_val = predictor.feature_extractor.transform(val_texts, use_stats_features=use_stats)
    
    # Progress helper: context manager to patch joblib's BatchCompletionCallBack so
    # it updates a tqdm bar when jobs complete. This makes RandomizedSearchCV
    # progress visible when n_jobs != 1.
    @contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar."""
        old_batch_callback = joblib.parallel.BatchCompletionCallBack

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=1)
                return super().__call__(*args, **kwargs)

        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback

    # Randomized Search with progress bar
    base_clf = OneVsRestClassifier(XGBClassifier(eval_metric="logloss"), n_jobs=-1)
    cv = 3
    search = RandomizedSearchCV(
        base_clf,
        DEFAULT_PARAM_GRID,
        cv=cv,
        n_iter=n_iter,
        scoring="f1_macro",  # macro-F1 treats all labels equally (better for unbalanced multilabel)
        n_jobs=-1,
        verbose=1,
    )

    # Show progress: total jobs ~ n_iter * cv (each candidate evaluated on 'cv' folds)
    total_jobs = int(n_iter) * int(cv)
    with tqdm(total=total_jobs, desc="RandomizedSearchCV jobs", unit="job") as pbar:
        with tqdm_joblib(pbar):
            search.fit(X_train, Y_train)

    # Log training results
    best_params = search.best_params_
    best_f1 = search.best_score_

    click.echo("===> RandomizedSearchCV completed")
    click.echo(f"Best CV params: {best_params}")
    click.echo(f"Best CV F1 (macro): {best_f1:.4f}")
    click.echo(f"Features: {X_train.shape[1]}, Training examples: {len(train_df)}")

    # Save predictor
    predictor.classifier = search.best_estimator_
    predictor.save(model_out)

    return predictor

# CLI

@click.group()
def cli():
    pass


@cli.command()
@click.argument("data_root", type=click.Path(exists=True))
@click.option("--model-out", default="models/xgb_tuned_model.pkl", help="Where to save the trained model.")
@click.option("--use-code", is_flag=True, help="Include code snippets in the input.")
@click.option("--vectorizer", type=click.Choice(["tfidf", "count"]), default="tfidf")
@click.option("--n-iter", default=50, help="Number of parameter search iterations.")
@click.option('--use-stats-features', is_flag=True, help='Include statistical features')
def train(data_root, model_out, use_code, vectorizer, n_iter, use_stats_features):
    """Train and tune the tag predictor with hyperparameter search."""
    tune_and_train(data_root, model_out, use_code, vectorizer, n_iter, use_stats_features)

if __name__ == "__main__":
    cli()