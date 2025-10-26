import os
from datetime import datetime
import pandas as pd


class ExperimentLogger:
    """Handles markdown-based experiment tracking with separate tables for global and per-tag metrics.

    Behavior / contract:
    - Creates `results.md` with two sections when missing: "Global Metrics" and "Per-tag Metrics".
    - Appends a single row to the Global Metrics table for each call to `log_result`.
    - Appends one row per tag to the Per-tag Metrics table for each call to `log_result`.
    - If a section is missing from an existing file, it will be appended at the end.
    """

    GLOBAL_SECTION = "## Global Metrics"
    PER_TAG_SECTION = "## Per-tag Metrics"

    GLOBAL_HEADER = (
        "| Date | Model | ID | Embedding | Use code | Use stats features | Classifier | Dataset | Precision | Recall | F1 | Hamming | Notes |\n"
    )
    GLOBAL_DIVIDER = (
        "|------|--------|------------|------------|----------|--------------------|-------------|----------|-----------|--------|----------|--------|\n"
    )

    PER_TAG_HEADER = (
        "| Model | ID | Notes | Tag | Precision | Recall | F1 | Support | Predicted |\n"
    )
    PER_TAG_DIVIDER = (
        "|--------|------------|-------|-----|-----------|--------|----|----------|-----------|\n"
    )

    def __init__(self, log_path="results.md"):
        self.log_path = log_path
        self._init_file()
        self._ensure_sections()

    def _init_file(self):
        """Create markdown file with sections if missing."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("# Model Comparison Report\n\n")
                f.write(f"_Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

                # Global metrics section
                f.write(f"{self.GLOBAL_SECTION}\n\n")
                f.write(self.GLOBAL_HEADER)
                f.write(self.GLOBAL_DIVIDER)
                f.write("\n")

                # Per-tag metrics section
                f.write(f"{self.PER_TAG_SECTION}\n\n")
                f.write(self.PER_TAG_HEADER)
                f.write(self.PER_TAG_DIVIDER)
                f.write("\n")

    def _ensure_sections(self):
        """Ensure both markdown sections exist, add if missing."""
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        content = "".join(lines)
        changed = False

        if self.GLOBAL_SECTION not in content:
            lines.append(f"\n{self.GLOBAL_SECTION}\n\n")
            lines.append(self.GLOBAL_HEADER)
            lines.append(self.GLOBAL_DIVIDER)
            lines.append("\n")
            changed = True

        if self.PER_TAG_SECTION not in content:
            lines.append(f"\n{self.PER_TAG_SECTION}\n\n")
            lines.append(self.PER_TAG_HEADER)
            lines.append(self.PER_TAG_DIVIDER)
            lines.append("\n")
            changed = True

        if changed:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

    def _insert_row_in_table(self, table_header_start: str, row: str):
        """Insert a new row right after the existing table entries."""
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header_idx = None
        for i, line in enumerate(lines):
            if line.strip() == table_header_start.strip():
                header_idx = i
                break

        if header_idx is None:
            # fallback if header not found
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n" + row)
            return

        # Find end of table (the last line starting with '|')
        j = header_idx
        while j + 1 < len(lines) and lines[j + 1].strip().startswith("|"):
            j += 1

        # Insert new row at the end of the table
        lines.insert(j + 1, row)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _next_model_id(self, model_name):
        """Determine the next numeric ID for a given model by scanning existing logs."""
        if not os.path.exists(self.log_path):
            return 1

        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        max_id = 0
        for line in lines:
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) < 3:
                continue

            model = parts[1]
            id_value = parts[2]

            # Expect ID format: model-### (e.g., bert-003)
            if model == model_name and id_value.startswith(f"{model_name}-"):
                try:
                    num = int(id_value.split("-")[-1])
                    max_id = max(max_id, num)
                except ValueError:
                    continue

        return max_id + 1

    def log_result(
        self,
        model_name,
        embedding,
        use_code,
        use_stats_features,
        classifier,
        dataset,
        metrics,
        notes="",
    ):
        """Log one global metrics row and zero-or-more per-tag rows."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Use numeric ID per model
        model_id_num = self._next_model_id(model_name)
        log_id = f"{model_name}-{model_id_num:03d}"

        # --- Global metrics row ---
        global_line = (
            f"| {timestamp} | {model_name} | {log_id} | {embedding} | {use_code} | {use_stats_features} | "
            f"{classifier} | {dataset} | "
            f"{metrics.get('precision_macro', 0):.4f} | {metrics.get('recall_macro', 0):.4f} | "
            f"{metrics.get('f1_macro', 0):.4f} | {metrics.get('hamming_loss', 0):.4f} | {notes} |\n"
        )
        self._insert_row_in_table(self.GLOBAL_HEADER, global_line)

        # --- Per-tag metrics rows ---
        for tag_metrics in metrics.get("per_tag", []):
            tag_line = (
                f"| {model_name} | {log_id} | {notes} | {tag_metrics.get('tag', '')} | "
                f"{tag_metrics.get('precision', 0):.4f} | {tag_metrics.get('recall', 0):.4f} | "
                f"{tag_metrics.get('f1', 0):.4f} | {tag_metrics.get('support', 0)} | "
                f"{tag_metrics.get('predicted', 0)} |\n"
            )
            self._insert_row_in_table(self.PER_TAG_HEADER, tag_line)

        self._generate_csv_files()

    def _generate_csv_files(self):
        """Generate CSV files for global and per-tag metrics."""
        global_data = []
        per_tag_data = []

        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # --- Extract global metrics ---
        section = None
        for line in lines:
            if line.strip() == self.GLOBAL_SECTION:
                section = "global"
                continue
            elif line.strip() == self.PER_TAG_SECTION:
                section = "per_tag"
                continue

            if not line.startswith("|") or "------" in line:
                continue

            parts = [x.strip() for x in line.strip().split("|")[1:-1]]

            if section == "global" and len(parts) == 13:
                global_data.append(parts)
            elif section == "per_tag" and len(parts) == 9:
                per_tag_data.append(parts)

        # --- Save to CSV ---
        if global_data:
            global_df = pd.DataFrame(
                global_data,
                columns=[
                    "Date",
                    "Model",
                    "ID",
                    "Embedding",
                    "Use code",
                    "Use stats features",
                    "Classifier",
                    "Dataset",
                    "Precision",
                    "Recall",
                    "F1",
                    "Hamming",
                    "Notes",
                ],
            )
            global_df.to_csv("global_metrics.csv", index=False)

        if per_tag_data:
            per_tag_df = pd.DataFrame(
                per_tag_data,
                columns=[
                    "Model",
                    "ID",
                    "Notes",
                    "Tag",
                    "Precision",
                    "Recall",
                    "F1",
                    "Support",
                    "Predicted",
                ],
            )
            per_tag_df.to_csv("per_tag_metrics.csv", index=False)
