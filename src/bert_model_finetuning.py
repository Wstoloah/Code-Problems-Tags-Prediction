"""
LoRA Fine-tuning for ModernBERT Multi-Label Tag Classification
- Parameter-efficient fine-tuning with LoRA
- Multi-label classification using ModernBERT
- Optimized for Codeforces tag prediction
- Supports both CodeBERT and ModernBERT
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import warnings

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    precision_recall_fscore_support
)
from tqdm import tqdm

# PEFT for LoRA
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# Accelerate for multi-GPU
from accelerate import Accelerator

from utils.data_loader import load_all_splits, load_dataset_split
from utils.logger import ExperimentLogger

warnings.filterwarnings('ignore')

# Target tags
TARGET_TAGS = [
    'math', 'graphs', 'strings', 'number theory',
    'trees', 'geometry', 'games', 'probabilities'
]

# Model configurations
MODEL_CONFIGS = {
    "modernbert": {
        "model_name": "answerdotai/ModernBERT-base",
        "lora_targets": ["Wqkv"],  # ModernBERT uses fused QKV attention
        "max_length": 800,  # ModernBERT supports very long sequences!
        "hidden_size": 768
    },
    "codebert": {
        "model_name": "microsoft/codebert-base",
        "lora_targets": ["query", "value"],
        "max_length": 512,
        "hidden_size": 768
    }
}


# Dataset for Multi-Label Classification
class CodeProblemsDataset(Dataset):
    """Dataset for code problems with multi-label tags"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        target_tags: List[str],
        max_length: int = 512,
        use_code: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_code = use_code
        
        # Multi-label binarizer
        self.mlb = MultiLabelBinarizer(classes=target_tags)
        self.mlb.fit([target_tags])
        self.labels = self.mlb.transform(df["tags"])
        
        print(f"   Dataset size: {len(df)}")
        print(f"   Num labels: {len(target_tags)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build text
        text = row["description"].strip() if pd.notna(row["description"]) else ""
        if self.use_code and "code" in self.df.columns:
            code = row["code"].strip() if pd.notna(row["code"]) else ""
            if code:
                text = f"{text}\n\n{code}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors=None  # Returns lists, not tensors
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": self.labels[idx].astype(np.float32)
        }

# Custom Model with LoRA for Multi-Label Classification
class ModernBERTLoRAForMultiLabel(nn.Module):
    """ModernBERT/CodeBERT with LoRA adapters for multi-label classification"""
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_labels: int = len(TARGET_TAGS),
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load base model
        print(f"   Loading base model: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from config
        self.hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   Output labels: {num_labels}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for multi-label classification
        
        Returns:
            dict with loss and logits
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool: use [CLS] token (first token)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        
        # Classify
        logits = self.classifier(pooled)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Binary cross-entropy for multi-label
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits
        }

# Data Collator for Dynamic Padding
class DataCollatorWithPaddingMultiLabel:
    """Collator that pads sequences dynamically"""
    
    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, features):
        # Extract components
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad input_ids and attention_mask
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Stack labels
        batch["labels"] = torch.tensor(np.stack(labels), dtype=torch.float32)
        
        return batch

# Training Configuration

class TrainingConfig:
    """Training hyperparameters"""
    
    def __init__(self, model_type: str = "modernbert"):
        # Select model configuration
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")
        
        model_cfg = MODEL_CONFIGS[model_type]
        
        self.model_type = model_type
        self.model_name = model_cfg["model_name"]
        self.max_length = model_cfg["max_length"]
        self.hidden_size = model_cfg["hidden_size"]
        
        # Training params
        self.num_epochs = 10
        self.batch_size = 16
        self.learning_rate = 5e-4
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.logging_steps = 50
        self.use_code = False
        
        # LoRA config
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = model_cfg["lora_targets"]
        
        # Scheduler
        self.scheduler_type = "cosine"  # or "linear"
        
    def to_dict(self):
        return vars(self)


# Main Training Function
def main(
    data_root: str,
    output_dir: str = "models/lora_codebert",
    config: Optional[TrainingConfig] = None
):
    """
    Main training loop with LoRA fine-tuning
    
    Args:
        data_root: Path to data directory
        output_dir: Where to save the model
        config: Training configuration
    """
    
    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator()
    
    # Default config
    if config is None:
        config = TrainingConfig(model_type="modernbert")
    
    accelerator.print("=" * 80)
    accelerator.print(f"LORA FINE-TUNING FOR {config.model_type.upper()}")
    accelerator.print("=" * 80)
    accelerator.print(f"   Model: {config.model_name}")
    accelerator.print(f"   Max Length: {config.max_length}")
    accelerator.print(f"   LoRA Targets: {config.lora_target_modules}")
    
    # 1. Load Data
    accelerator.print("\n ===> Loading data...")
    train_df, val_df, _ = load_all_splits(data_root)
    
    accelerator.print(f"   Train: {len(train_df)} samples")
    accelerator.print(f"   Val:   {len(val_df)} samples")
    
    # 2. Prepare Tokenizer and Datasets
    accelerator.print(f"\n===> Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Create datasets
    accelerator.print("\n===> Creating datasets...")
    train_dataset = CodeProblemsDataset(
        train_df, tokenizer, TARGET_TAGS, config.max_length, config.use_code
    )
    val_dataset = CodeProblemsDataset(
        val_df, tokenizer, TARGET_TAGS, config.max_length, config.use_code
    )
    
    # Data collator
    data_collator = DataCollatorWithPaddingMultiLabel(tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=data_collator
    )
    
    accelerator.print(f"   Train batches: {len(train_loader)}")
    accelerator.print(f"   Val batches:   {len(val_loader)}")
    
    # 3. Load Model and Apply LoRA
    accelerator.print(f"\n===> Loading model: {config.model_name}")
    model = ModernBERTLoRAForMultiLabel(
        model_name=config.model_name,
        num_labels=len(TARGET_TAGS),
        dropout=0.1
    )
    
    # LoRA configuration
    accelerator.print(f"\n===> Applying LoRA with targets: {config.lora_target_modules}")
    
    # For ModernBERT, we need to target the correct attention modules
    # ModernBERT uses fused attention: attn.Wqkv
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION  # We have custom head
    )
    
    # Apply LoRA to encoder only
    model.encoder = get_peft_model(model.encoder, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"   Trainable parameters: {trainable_params:,} / {total_params:,}")
    accelerator.print(f"   Percentage: {100 * trainable_params / total_params:.2f}%")
    
    # 4. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    #Scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    if config.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    accelerator.print(f"\n===> Optimizer: AdamW (lr={config.learning_rate})")
    accelerator.print(f"   Scheduler: {config.scheduler_type}")
    accelerator.print(f"   Warmup steps: {num_warmup_steps}")
    
    # 5. Prepare with Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # 6. Training Loop
    accelerator.print("\n" + "=" * 80)
    accelerator.print("===> STARTING TRAINING")
    accelerator.print("=" * 80)
    
    best_val_f1 = 0.0
    global_step = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_macro": [],
        "val_f1_micro": []
    }
    
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        accelerator.print(f"\n{'='*80}")
        accelerator.print(f"EPOCH {epoch + 1}/{config.num_epochs}")
        accelerator.print(f"{'='*80}")
        
        # Training
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            running_loss += loss.item()
            global_step += 1
            
            if global_step % config.logging_steps == 0:
                avg_loss = running_loss / config.logging_steps
                accelerator.print(
                    f"Step [{step+1}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss = 0.0
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Validation
        accelerator.print("\nValidating...")
        model.eval()
        
        all_preds = []
        all_labels = []
        eval_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc="Validation",
                disable=not accelerator.is_local_main_process
            ):
                outputs = model(**batch)
                eval_loss += outputs["loss"].item()
                
                # Get predictions
                logits = outputs["logits"]
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                
                # Gather from all processes
                all_preds_batch = accelerator.gather(preds).cpu().numpy()
                all_labels_batch = accelerator.gather(batch["labels"]).cpu().numpy()
                
                all_preds.append(all_preds_batch)
                all_labels.append(all_labels_batch)
                num_batches += 1
        
        # Concatenate
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Compute metrics
        avg_eval_loss = eval_loss / num_batches
        hamming = hamming_loss(all_labels, all_preds)
        
        _, _, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        _, _, f1_micro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="micro", zero_division=0
        )
        
        # Store history
        history["train_loss"].append(running_loss / len(train_loader))
        history["val_loss"].append(avg_eval_loss)
        history["val_f1_macro"].append(f1_macro)
        history["val_f1_micro"].append(f1_micro)
        
        # Print results
        accelerator.print(f"\n=== Epoch {epoch + 1} Results: ===")
        accelerator.print(f"   Val Loss:       {avg_eval_loss:.4f}")
        accelerator.print(f"   Val F1 (macro): {f1_macro:.4f}")
        accelerator.print(f"   Val F1 (micro): {f1_micro:.4f}")
        accelerator.print(f"   Hamming Loss:   {hamming:.4f}")
        
        # Save Checkpoints
        
        # Save best model
        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            
            if accelerator.is_main_process:
                best_dir = Path(output_dir) / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                
                # Unwrap model for saving
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Save encoder with LoRA
                unwrapped_model.encoder.save_pretrained(best_dir / "encoder")
                
                # Save classifier head
                torch.save(
                    unwrapped_model.classifier.state_dict(),
                    best_dir / "classifier.pt"
                )
                
                # Save tokenizer
                tokenizer.save_pretrained(best_dir)
                
                # Save config
                with open(best_dir / "config.json", "w") as f:
                    json.dump(config.to_dict(), f, indent=2)
                
                accelerator.print(f"   ===> Best model saved! F1: {best_val_f1:.4f}")
        
        # Save periodic checkpoint
        if accelerator.is_main_process and (epoch + 1) % 2 == 0:
            checkpoint_dir = Path(output_dir) / f"checkpoint_epoch_{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.encoder.save_pretrained(checkpoint_dir / "encoder")
            torch.save(
                unwrapped_model.classifier.state_dict(),
                checkpoint_dir / "classifier.pt"
            )
            tokenizer.save_pretrained(checkpoint_dir)
            
            accelerator.print(f"    Checkpoint saved: {checkpoint_dir}")
    
    # 7. Training Complete
    elapsed = time.time() - start_time
    accelerator.print("\n" + "=" * 80)
    accelerator.print(f"=== TRAINING COMPLETED in {elapsed/60:.2f} minutes ===")
    accelerator.print(f"=== Best Val F1 (macro): {best_val_f1:.4f} ===")
    accelerator.print("=" * 80)
    
    # Save final model
    if accelerator.is_main_process:
        final_dir = Path(output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.encoder.save_pretrained(final_dir / "encoder")
        torch.save(unwrapped_model.classifier.state_dict(), final_dir / "classifier.pt")
        tokenizer.save_pretrained(final_dir)
        
        # Save history
        with open(final_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        accelerator.print(f"\n===> Final model saved to {final_dir}")


# Inference Class
class LoRAPredictor:
    """Predictor using LoRA fine-tuned model"""
    
    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        Load LoRA fine-tuned model
        
        Args:
            model_dir: Directory containing encoder/, classifier.pt, tokenizer
            device: Device to use (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_dir)
        
        # Load config
        with open(model_path / "config.json", "r") as f:
            config_dict = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model with LoRA
        base_model = AutoModel.from_pretrained(config_dict["model_name"])
        self.encoder = PeftModel.from_pretrained(
            base_model,
            model_path / "encoder"
        )
        
        # Load classifier head
        self.classifier = nn.Linear(config_dict.get("hidden_size", 768), len(TARGET_TAGS))
        self.classifier.load_state_dict(
            torch.load(model_path / "classifier.pt", map_location=self.device)
        )
        
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        
        self.encoder.eval()
        self.classifier.eval()
        
        print(f"===> Model loaded from {model_dir}")
    
    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        threshold: float = 0.5,
        batch_size: int = 16
    ):
        """Predict tags for texts"""
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Encode
            outputs = self.encoder(**encodings)
            pooled = outputs.last_hidden_state[:, 0, :]
            
            # Classify
            logits = self.classifier(pooled)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
        
        all_probs = np.vstack(all_probs)
        
        # Apply threshold
        predictions = []
        for prob_vec in all_probs:
            pred_tags = [
                TARGET_TAGS[i]
                for i, prob in enumerate(prob_vec)
                if prob >= threshold
            ]
            predictions.append(pred_tags)
        
        return predictions, all_probs

# Helper to build text features (description +/- code)
def _build_texts_from_df(df: pd.DataFrame, use_code: bool) -> List[str]:
    """
    Build the string list that will be passed to vectorizer.
    Uses 'description' column and optionally 'code' column.
    If df has a 'text' column already, prefer that for backward compatibility.
    """
    if "text" in df.columns:
        return df["text"].fillna("").tolist()

    descs = df["description"].fillna("").tolist() if "description" in df.columns else [""] * len(df)
    if use_code and "code" in df.columns:
        codes = df["code"].fillna("").tolist()
        return [d.strip() + "\n\n" + c.strip() for d, c in zip(descs, codes)]
    else:
        return [d.strip() for d in descs]
    
# CLI

@click.group()
def cli():
    """LoRA Fine-tuning CLI for CodeBERT"""
    pass


@cli.command()
@click.argument("data_root", type=click.Path(exists=True))
@click.option("--output-dir", default="models/lora_modernbert")
@click.option("--model-type", type=click.Choice(["modernbert", "codebert"]), default="modernbert")
@click.option("--epochs", default=10)
@click.option("--batch-size", default=16)
@click.option("--learning-rate", default=5e-4, type=float)
@click.option("--max-length", type=int, default=None, help="Override default max length")
@click.option("--lora-r", default=8, help="LoRA rank")
@click.option("--lora-alpha", default=32, help="LoRA alpha")
@click.option("--use-code", is_flag=True)
def train(data_root, output_dir, model_type, epochs, batch_size, learning_rate,
          max_length, lora_r, lora_alpha, use_code):
    """Train with LoRA"""
    config = TrainingConfig(model_type=model_type)
    config.num_epochs = epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.lora_r = lora_r
    config.lora_alpha = lora_alpha
    config.use_code = use_code
    
    # Override max_length if specified
    if max_length is not None:
        config.max_length = max_length
    
    main(data_root, output_dir, config)

@cli.command()
@click.argument("data_root", type=click.Path(exists=True))
@click.option("--model-dir", required=True)
@click.option("--split", default="test")
@click.option("--threshold", default=0.5, help="Threshold for tag prediction")
@click.option("--batch-size", default=None, type=int, help="Batch size for prediction (default: 16)")
@click.option('--notes', default='', help='Optional notes for this experiment')
def evaluate(data_root, model_dir, split, threshold, batch_size, notes):
    """Evaluate LoRA model. Parameters like use_code are loaded from the saved model."""
    click.echo(f"==== Evaluating on {split} set ====\n")
    
    # Load predictor
    predictor = LoRAPredictor(model_dir)
    
    # Load config to get use_code
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as cf:
            cfg = json.load(cf)
        use_code = bool(cfg.get("use_code", False))
        model_type = cfg.get("model_type", "unknown")
        max_length = cfg.get("max_length", 512)
    else:
        use_code = False
        model_type = "unknown"
        max_length = 512
    
    # Use default batch_size if not specified
    if batch_size is None:
        batch_size = 16
    
    click.echo(f"Model configuration: model={model_type}, use_code={use_code}, max_length={max_length}")
    click.echo(f"Prediction settings: batch_size={batch_size}, threshold={threshold}\n")
    
    # Load data
    df = load_dataset_split(os.path.join(data_root, split))
    texts = _build_texts_from_df(df, use_code)
    
    click.echo(f"Evaluating {len(df)} examples...\n")
    
    # Predict
    predictions, probs = predictor.predict(texts, threshold=threshold, batch_size=batch_size)
    
    # Evaluate
    mlb = MultiLabelBinarizer(classes=TARGET_TAGS)
    mlb.fit([TARGET_TAGS])
    y_true = mlb.transform(df["tags"])
    y_pred = (probs >= threshold).astype(int)
    
    # Global metrics
    hamming = hamming_loss(y_true, y_pred)
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Per-tag metrics
    precision_per_tag, recall_per_tag, f1_per_tag, support_per_tag = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    predicted_count = y_pred.sum(axis=0)
    
    # Print per-tag table
    print("\n==== Target Tags Performance ====")
    print(f"{'Tag':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}{'Predicted':>10}")
    print("="*70)
    for idx, tag in enumerate(TARGET_TAGS):
        print(f"{tag:<20}{precision_per_tag[idx]:>10.3f}{recall_per_tag[idx]:>10.3f}{f1_per_tag[idx]:>10.3f}{int(support_per_tag[idx]):>10d}{int(predicted_count[idx]):>10d}")
    
    # Print global metrics
    print(f"\n{'='*70}")
    print("Global Metrics:")
    print(f"  Hamming Loss:           {hamming:.4f}")
    print(f"  Precision (samples):    {precision_samples:.4f}")
    print(f"  Recall (samples):       {recall_samples:.4f}")
    print(f"  F1 (samples):           {f1_samples:.4f}")
    print(f"  Precision (macro):      {precision_macro:.4f}")
    print(f"  Recall (macro):         {recall_macro:.4f}")
    print(f"  F1 (macro):             {f1_macro:.4f}")
    print("="*70)
    
    # Log results if notes provided
    logger = ExperimentLogger("results.md")
        
    metrics = {
        "per_tag": [{"tag": t, "precision": float(p), "recall": float(r), "f1": float(f),
                    "support": int(s), "predicted": int(pc)}
                    for t, p, r, f, s, pc in zip(TARGET_TAGS, precision_per_tag, recall_per_tag, 
                                                    f1_per_tag, support_per_tag, predicted_count)],
        "hamming_loss": float(hamming),
        "precision_samples": float(precision_samples),
        "recall_samples": float(recall_samples),
        "f1_samples": float(f1_samples),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro)
    }
    
    logger.log_result(
        model_name="LoRAFineTunedModel",
        embedding=model_type,
        use_code=use_code,
        use_stats_features=False,
        classifier="LoRA+Classifier",
        dataset=split,
        metrics=metrics,
        notes=notes
    )
    
    click.echo("\n==== Evaluation complete! ====")

@cli.command()
@click.argument("data_root", type=click.Path(exists=True))
@click.option("--model-dir", required=True)
@click.option("--split", default="test", type=click.Choice(["train", "val", "test"]))
@click.option("--output", default="predictions_lora.json")
@click.option("--threshold", default=0.5, help="Threshold for tag prediction")
@click.option("--batch-size", default=None, type=int, help="Batch size for prediction (default: 16)")
def predict(data_root, model_dir, split, output, threshold, batch_size):
    """Predict tags for problems in a dataset split. Parameters are loaded from the saved model."""
    click.echo(f"==== Predicting Tags on {split} set using LoRA model ====\n")
    
    # Load predictor
    predictor = LoRAPredictor(model_dir)
    
    # Load config
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as cf:
            cfg = json.load(cf)
        use_code = bool(cfg.get("use_code", False))
        model_type = cfg.get("model_type", "unknown")
        max_length = cfg.get("max_length", 512)
    else:
        use_code = False
        model_type = "unknown"
        max_length = 512
    
    # Use default batch_size if not specified
    if batch_size is None:
        batch_size = 16
    
    click.echo(f"Model configuration: model={model_type}, use_code={use_code}, max_length={max_length}")
    click.echo(f"Prediction settings: batch_size={batch_size}, threshold={threshold}\n")
    
    # Load data
    df = load_dataset_split(os.path.join(data_root, split))
    texts = _build_texts_from_df(df, use_code)
    
    click.echo(f"Making predictions on {len(df)} examples...")
    start = time.time()
    predictions, probs = predictor.predict(texts, threshold=threshold, batch_size=batch_size)
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
@click.option("--model-dir", required=True)
@click.option("--threshold", default=0.5, help="Threshold for tag prediction")
def predict_one(text, model_dir, threshold):
    """Predict tags for a single problem description. Parameters are loaded from the saved model."""
    # Load predictor
    predictor = LoRAPredictor(model_dir)
    
    # Load config
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as cf:
            cfg = json.load(cf)
        use_code = bool(cfg.get("use_code", False))
        model_type = cfg.get("model_type", "unknown")
        max_length = cfg.get("max_length", 512)
    else:
        use_code = False
        model_type = "unknown"
        max_length = 512
    
    click.echo(f"==== Predicting single sample ====")
    click.echo(f"Model configuration: model={model_type}, use_code={use_code}, max_length={max_length}")
    click.echo(f"Prediction settings: threshold={threshold}\n")
    
    start = time.time()
    predictions, probs = predictor.predict([text], threshold=threshold, batch_size=1)
    elapsed = time.time() - start
    
    click.echo(f"Predicted tags: {predictions[0]}")
    click.echo(f"Prediction time: {elapsed*1000:.2f} ms")
    
    # Show probabilities for all tags
    click.echo("\nTag probabilities:")
    for tag, prob in zip(TARGET_TAGS, probs[0]):
        click.echo(f"  {tag:<20}: {prob:.4f}")

if __name__ == "__main__":
    cli()