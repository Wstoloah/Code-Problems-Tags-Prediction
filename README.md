# Codeforces Tag Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains experiments and models for **multi-label tag prediction** on Codeforces-style programming problems. The project implements three different approaches:

- **Baseline classical ML pipeline** using TF-IDF/Count vectorization + XGBoost/Logistic Regression (in `src/models/baseline_model.py`)
- **Embedding-based approach** using frozen ModernBERT/CodeBERT embeddings + sklearn classifiers (in `src/models/bert_based_model.py`)
- **LoRA-based parameter-efficient fine-tuning** pipeline for ModernBERT/CodeBERT (in `src/models/finetuned_bert_model.py`)

**Goal:** Predict target tags (math, graphs, strings, number theory, trees, geometry, games, probabilities) from problem descriptions and optionally code snippets.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
  - [1. Baseline Model (TF-IDF + XGBoost)](#1-baseline-model-tf-idf--xgboost)
  - [2. Embedding-Based Model (ModernBERT/CodeBERT)](#2-embedding-based-model-modernbertcodebert)
  - [3. LoRA Fine-Tuning](#3-lora-fine-tuning)
- [Model Comparison](#model-comparison)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Testing](#testing)
- [Contributing](#contributing)

## Repository Structure

```
.
├── data/                                  # Dataset splits
│   ├── train/                            # Training samples
│   ├── val/                              # Validation samples
│   ├── test/                             # Test samples
│   └── code_classification_dataset/      # Original dataset
├── src/                                   # Source code
│   ├── models/                           # Model implementations
│   │   ├── baseline_model.py            # TF-IDF/Count Vectorizer + classical ML
│   │   ├── bert_based_model.py          # Frozen embeddings + classifiers
│   │   └── finetuned_bert_model.py      # LoRA fine-tuning
│   ├── tuning/                           # Hyperparameter tuning scripts
│   │   ├── tuning_xgboost_baseline_model.py
│   │   └── tuning_bert_xgboost_model.py
│   ├── utils/                            # Utility functions
│   │   ├── data_loader.py               # Data loading utilities
│   │   ├── logger.py                    # Experiment logging
│   │   └── split_datasets.py            # Dataset splitting
│   └── metrics/                          # Evaluation metrics
├── models/                                # Saved trained models
│   ├── baseline/                         # Baseline model checkpoints
│   ├── bert_based/                       # BERT-based model checkpoints
│   └── finetuned/                        # Fine-tuned model checkpoints
├── notebooks/                             # Jupyter notebooks
│   ├── EDA.ipynb                         # Exploratory Data Analysis
│   └── figures/                          # Generated figures
├── scripts/                               # Utility scripts
│   └── models_comparaison.py             # Model comparison script
├── cache/                                 # Cached embeddings
├── outputs/                               # Experiment outputs
│   ├── logs/                             # Training logs
│   ├── metrics/                          # Evaluation metrics
│   ├── plots/                            # Visualization plots
│   └── predictions/                      # Model predictions
├── tests/                                 # Unit tests
├── requirements.txt                       # Python dependencies
├── setup.py                              # Package setup
└── README.md                             # This file
```

## Requirements

- **Python:** 3.8 or higher
- **Operating System:** Windows, Linux, or macOS
- **Hardware:** 
  - CPU: Any modern processor
  - GPU: Recommended for LoRA fine-tuning (CUDA-compatible)
  - RAM: Minimum 8GB, 16GB+ recommended for large models

### Key Dependencies

- `transformers` - Hugging Face transformers library
- `sentence-transformers` - Sentence embeddings
- `torch` - PyTorch deep learning framework
- `accelerate` - Distributed training support
- `peft` - Parameter-efficient fine-tuning
- `scikit-learn` - Classical ML algorithms
- `xgboost` - Gradient boosting
- `click` - CLI framework
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization

See `requirements.txt` for the complete list.

## Installation

### 1. Clone the Repository

```powershell
https://github.com/Wstoloah/Code-Problems-Tags-Prediction.git
cd Code-Problems-Tags-Prediction
```

### 2. Create Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. (Optional) Configure Accelerate for GPU Training

If you plan to use LoRA fine-tuning with GPU:

```powershell
accelerate config
```

Follow the prompts to set up your hardware configuration.

## Data Format

Each dataset split is expected to be in CSV or JSON format with the following columns:

- **`description`** (required) - Problem statement text
- **`tags`** (required) - List of tags for training/evaluation
- **`code`** (optional) - Code snippet to include as additional context

### Target Tags

The model predicts the following 8 tags:
- `math` - Mathematical problems
- `graphs` - Graph theory problems
- `strings` - String manipulation problems
- `number theory` - Number theory problems
- `trees` - Tree-based problems
- `geometry` - Geometric problems
- `games` - Game theory problems
- `probabilities` - Probability and statistics problems

### Example Data Structure

```json
{
  "description": "Find the number of paths in a tree that sum to K",
  "tags": ["trees", "math"],
  "code": "def solve(n, k): ..."
}
```

Sample files are available in `data/`.

## Usage

All commands should be run from the repository root directory.

### 1. Baseline Model (TF-IDF/Count Vectorize + XGBoost)

The baseline model uses traditional feature extraction (TF-IDF or Count vectorization) combined with classical ML classifiers.

#### Train with Hyperparameter Tuning

Train an XGBoost baseline with automated hyperparameter search:

```powershell
python src/tuning/tuning_xgboost_baseline_model.py train data/ --model-out models/baseline/xgb_tuned_model.pkl --use-code --vectorizer count --n-iter 50
```

**Parameters:**
- `--use-code` - Include code snippets in training
- `--vectorizer {tfidf,count}` - Choice of text vectorization method
- `--n-iter` - Number of hyperparameter search iterations

#### Train Baseline Model Directly

Train with default parameters (Logistic Regression):

```powershell
python src/models/baseline_model.py train data/ --model-out models/baseline/logreg.pkl --use-code --vectorizer tfidf
```

#### Evaluate Baseline Model

```powershell
python src/models/baseline_model.py evaluate data/ --split test --model-path models/baseline/xgb_tuned_model.pkl
```

### 2. Embedding-Based Model (ModernBERT/CodeBERT)

This approach uses pre-trained transformer models to generate embeddings, then trains a classical classifier on top.

#### Train Embedding Model

Using ModernBERT with Logistic Regression:

```powershell
python src/models/bert_based_model.py train data/ --model modernbert --classifier logistic --model-path models/bert_based/modernbert_logistic.pkl --use-code --use-cache
```

Using CodeBERT with XGBoost:

```powershell
python src/models/bert_based_model.py train data/ --model codebert --classifier xgboost --model-path models/bert_based/codebert_xgb.pkl --use-code --use-cache
```

**Parameters:**
- `--model {modernbert,codebert}` - Choice of embedding model
- `--classifier {logistic,xgboost,randomforest}` - Classifier type
- `--use-cache` - Cache embeddings to disk for faster experimentation
- `--use-code` - Include code snippets

#### Evaluate Embedding Model

```powershell
python src/models/bert_based_model.py evaluate data/ --split test --model-path models/bert_based/modernbert_logistic.pkl --use-cache
```

#### Predict Single Sample

```powershell
python src/models/bert_based_model.py predict-one "Find number of paths in a tree that sum to K" --model-path models/bert_based/modernbert_logistic.pkl
```

### 3. LoRA Fine-Tuning

Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) for the best performance.

#### Train Fine-Tuned Model

```powershell
python src/models/finetuned_bert_model.py train data/ --output-dir models/finetuned/lora_modernbert --model-type modernbert --epochs 10 --batch-size 16 --use-code
```

**Parameters:**
- `--model-type {modernbert,codebert}` - Base model to fine-tune
- `--epochs` - Number of training epochs
- `--batch-size` - Training batch size
- `--learning-rate` - Learning rate (default: 2e-4)
- `--use-code` - Include code snippets

#### Evaluate Fine-Tuned Model

```powershell
python src/models/finetuned_bert_model.py evaluate data/ --model-dir models/finetuned/lora_modernbert/best_model --split val
```

#### Predict with Fine-Tuned Model

```powershell
python src/models/finetuned_bert_model.py predict-one "Graph traversal problem with cycles" --model-dir models/finetuned/lora_modernbert/best_model
```

## Model Comparison

Compare performance across different models:

```powershell
python scripts/models_comparaison.py --models baseline bert_based finetuned --output outputs/comparison.csv
```

This will generate comparative metrics and visualizations across all trained models.

## Exploratory Data Analysis

Explore the dataset using the provided Jupyter notebook:

```powershell
jupyter notebook notebooks/EDA.ipynb
```

The notebook includes:
- Dataset statistics and distribution
- Tag co-occurrence analysis
- Text length distributions
- Visualization of tag frequencies
- Data quality checks

## Outputs and Artifacts

The project generates various outputs during training and evaluation:

### Trained Models

Models are saved in the `models/` directory:
- **`models/baseline/`** - TF-IDF + classical ML models (`.pkl` files)
- **`models/bert_based/`** - Embedding-based models (`.pkl` files)
- **`models/finetuned/`** - LoRA fine-tuned models (model checkpoints)

### Cached Embeddings

When using `--use-cache`, embeddings are stored in `cache/` to speed up experimentation:
- **`cache/modernbert/`** - ModernBERT embeddings (`.npy` files)
- **`cache/codebert/`** - CodeBERT embeddings (`.npy` files)

### Experiment Outputs

Results and logs are saved in `outputs/`:
- **`outputs/logs/`** - Training logs and console output
- **`outputs/metrics/`** - Evaluation metrics (accuracy, F1, precision, recall)
- **`outputs/plots/`** - Visualization plots (confusion matrices, ROC curves)
- **`outputs/predictions/`** - Model predictions on test data

### Experiment Logging

The project uses `ExperimentLogger` (in `src/utils/logger.py`) to automatically track:
- Model hyperparameters
- Training progress
- Evaluation metrics
- Timestamp and experiment metadata

## Troubleshooting & Tips

### Common Issues

#### Out of Memory (GPU)

If you encounter GPU memory errors:

```powershell
# Reduce batch size
python src/models/finetuned_bert_model.py train data/ --output-dir models/finetuned/lora --batch-size 8

# Disable code inclusion
python src/models/finetuned_bert_model.py train data/ --output-dir models/finetuned/lora --batch-size 16
```

#### Slow Embedding Computation

Enable caching to avoid recomputing embeddings:

```powershell
python src/models/bert_based_model.py train data/ --model modernbert --use-cache
```

Cached embeddings are reused across runs, significantly speeding up experimentation.

#### Accelerate Configuration Issues

If `accelerate` complains about configuration:

```powershell
# Run configuration wizard
accelerate config

# Choose appropriate options for your hardware:
# - Single GPU for training on one GPU
# - CPU only for training without GPU
# - Multi-GPU for distributed training
```

#### Module Import Errors

If you encounter import errors, ensure you're running from the repository root:

```powershell
# Check current directory
pwd

# Should output: C:\Users\<username>\Illuin-technical-challenge
```

### Performance Tips

1. **Use caching** (`--use-cache`) when iterating on model architecture
2. **Start with smaller batch sizes** and increase gradually
3. **Use validation set** to monitor overfitting during training
4. **Enable code inclusion** (`--use-code`) for better performance on code-heavy problems
5. **Try different classifiers** (logistic, xgboost, random forest) for embedding-based models

## Contributing

Contributions are welcome! Here are some ideas for extending the project:

### Potential Improvements

1. **Additional Target Tags**
   - Add more fine-grained tags
   - Support dynamic tag selection based on dataset

2. **Model Enhancements**
   - Experiment with other transformer models (RoBERTa, DeBERTa)
   - Try ensemble methods combining multiple models
   - Threshold Calibration for different tags

3. **Evaluation & Analysis**
   - Add cross-validation support
   - Multi-dataset comparison scripts
   - Error analysis notebooks

4. **Performance Optimization**
   - Lightweight model distillation
   - Quantization for faster inference
   - Batch prediction API

5. **Documentation**
   - Add more usage examples
   - Create tutorial notebooks
   - Document model architecture decisions

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write/update tests as needed
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Acknowledgments

- Codeforces for the problem dataset
- Hugging Face for transformer models and libraries
- The open-source community for the excellent tools and frameworks

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note:** This project was developed as part of a technical challenge. For more details about the methodology and results, refer to the analysis in `notebooks/EDA.ipynb` and the results directory.

