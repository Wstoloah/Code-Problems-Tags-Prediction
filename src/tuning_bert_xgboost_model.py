import os
import click
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from tqdm import tqdm
from contextlib import contextmanager

from utils.data_loader import load_all_splits, load_dataset_split
from utils.logger import ExperimentLogger
from bert_based_model import TagPredictor, TARGET_TAGS, MODEL_CONFIGS

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

def tune_and_train(
    data_root,
    model_out,
    bert_model="modernbert",
    use_code=False,
    batch_size=8,
    use_cache=True,
    cache_dir="cache",
    n_iter=50,
):
    """Train + tune a multi-label XGBoost classifier with BERT embeddings using RandomizedSearchCV."""
    
    # Load data
    train_df, val_df, _ = load_all_splits(data_root)

    # Initialize predictor with BERT model
    predictor = TagPredictor(
        model_name=bert_model,
        classifier_name="xgboost",  # Initial classifier, will be replaced with tuned one
        cache_dir=cache_dir,
        target_tags=TARGET_TAGS
    )
    predictor.use_code = use_code
    
    print("==== Preparing data and extracting BERT embeddings... ====")
    
    # Get embeddings for train and validation sets
    train_texts = predictor._build_texts_from_df(train_df)
    val_texts = predictor._build_texts_from_df(val_df)
    
    X_train = predictor.embedder.encode_texts(
        train_texts, 
        batch_size=batch_size,
        use_cache=use_cache,
        max_length=MODEL_CONFIGS[bert_model]['max_length']
    )
    
    X_val = predictor.embedder.encode_texts(
        val_texts,
        batch_size=batch_size,
        use_cache=use_cache,
        max_length=MODEL_CONFIGS[bert_model]['max_length']
    )
    
    # Prepare labels
    y_train, _ = predictor.prepare_labels(train_df)
    y_val, _ = predictor.prepare_labels(val_df)

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Perform RandomizedSearchCV
    base_clf = OneVsRestClassifier(XGBClassifier(eval_metric="logloss"), n_jobs=-1)
    search = RandomizedSearchCV(
        base_clf,
        DEFAULT_PARAM_GRID,
        cv=3,
        n_iter=n_iter,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    # Progress helper: context manager to patch joblib's BatchCompletionCallBack to
    # update a tqdm bar as jobs complete. This makes RandomizedSearchCV progress visible
    # even when using parallel jobs.
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

    print("==== Starting hyperparameter search... ====")
    cv = 3
    total_jobs = int(n_iter) * int(cv)
    with tqdm(total=total_jobs, desc="RandomizedSearchCV jobs", unit="job") as pbar:
        with tqdm_joblib(pbar):
            search.fit(X_train, y_train)

    # Log results
    best_params = search.best_params_
    best_f1 = search.best_score_

    print("===> RandomizedSearchCV completed")
    print(f"Best CV params: {best_params}")
    print(f"Best CV F1 (macro): {best_f1:.4f}")
    print(f"Features: {X_train.shape[1]}, Training examples: {len(train_df)}")

    # Update predictor with best model
    predictor.classifier = search.best_estimator_
    predictor.save(model_out)

@click.group()
def cli():
    pass

@cli.command()
@click.argument("data_root", type=click.Path(exists=True))
@click.option("--model-out", default="models/bert_xgb_tuned.pkl", help="Where to save the trained model")
@click.option("--bert-model", type=click.Choice(["modernbert", "codebert"]), default="modernbert")
@click.option("--use-code", is_flag=True, help="Include code snippets in the input")
@click.option("--batch-size", default=8, help="Batch size for BERT encoding")
@click.option("--use-cache", is_flag=True, help="Use cached BERT embeddings")
@click.option("--n-iter", default=10, help="Number of parameter search iterations")
def train(data_root, model_out, bert_model, use_code, batch_size, use_cache, n_iter):
    """Train and tune XGBoost with BERT embeddings."""
    tune_and_train(
        data_root=data_root,
        model_out=model_out,
        bert_model=bert_model,
        use_code=use_code,
        batch_size=batch_size,
        use_cache=use_cache,
        n_iter=n_iter
    )

if __name__ == "__main__":
    cli()