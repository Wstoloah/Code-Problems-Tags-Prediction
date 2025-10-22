import os
from datetime import datetime

class ExperimentLogger:
    """Handles markdown-based experiment tracking."""

    def __init__(self, log_path="results.md"):
        self.log_path = log_path
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("# ==== Model Comparison Report ====\n\n")
                f.write(f"_Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
                f.write("| Date | Model | Embedding | Classifier | Dataset | Precision | Recall | F1 | Hamming | Notes |\n")
                f.write("|------|--------|------------|-------------|----------|-----------|--------|----|----------|--------|\n")

    def log_result(self, model_name, embedding, classifier, dataset, metrics, notes=""):
        line = (
            f"| {datetime.now().strftime('%Y-%m-%d %H:%M')} "
            f"| {model_name} | {embedding} | {classifier} "
            f"| {dataset} | {metrics.get('precision_macro', 0):.4f} "
            f"| {metrics.get('recall_macro', 0):.4f} "
            f"| {metrics.get('f1_macro', 0):.4f} "
            f"| {metrics.get('hamming_loss', 0):.4f} "
            f"| {notes} |\n"
        )
        with open(self.log_path, "a") as f:
            f.write(line)
        print(f"===> Results logged to {self.log_path}")
