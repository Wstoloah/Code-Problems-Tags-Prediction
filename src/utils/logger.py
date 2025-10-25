import os
from datetime import datetime


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
        "| Date | Model | Embedding | Use code | Use stats features | Classifier | Dataset | Precision | Recall | F1 | Hamming | Notes |\n"
    )
    GLOBAL_DIVIDER = (
        "|------|--------|------------|----------|--------------------|-------------|----------|-----------|--------|----|----------|--------|\n"
    )

    PER_TAG_HEADER = (
        "| Model | Notes | Tag | Precision | Recall | F1 | Support | Predicted |\n"
    )
    PER_TAG_DIVIDER = (
        "|--------|-------|-----|-----------|--------|----|----------|-----------|\n"
    )

    def __init__(self, log_path="results.md"):
        self.log_path = log_path
        self._init_file()
        self._ensure_sections()

    def _init_file(self):
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
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        changed = False

        def ensure_section_above(header_row, section_title):
            nonlocal lines, changed
            for idx, line in enumerate(lines):
                if line.strip() == header_row.strip():
                    # Check if section title exists a few lines above
                    lookback = max(0, idx - 4)
                    if not any(lines[i].strip() == section_title for i in range(lookback, idx)):
                        lines.insert(idx, section_title + "\n\n")
                        changed = True
                    return True
            return False

        global_found = ensure_section_above(self.GLOBAL_HEADER, self.GLOBAL_SECTION)
        per_tag_found = ensure_section_above(self.PER_TAG_HEADER, self.PER_TAG_SECTION)

        if not global_found and not per_tag_found:
            lines.append(f"\n{self.GLOBAL_SECTION}\n\n")
            lines.append(self.GLOBAL_HEADER)
            lines.append(self.GLOBAL_DIVIDER)
            lines.append("\n")
            lines.append(f"{self.PER_TAG_SECTION}\n\n")
            lines.append(self.PER_TAG_HEADER)
            lines.append(self.PER_TAG_DIVIDER)
            lines.append("\n")
            changed = True

        if changed:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

    def _insert_row_in_table(self, table_header_start: str, row: str):
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith(table_header_start.strip()):
                header_idx = i
                break

        if header_idx is None:
            for i, line in enumerate(lines):
                if line.strip() in (self.GLOBAL_HEADER.strip(), self.PER_TAG_HEADER.strip()):
                    header_idx = i
                    break

        if header_idx is None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n" + row)
            return

        # Find end of table
        j = header_idx
        while j + 1 < len(lines) and lines[j + 1].lstrip().startswith("|"):
            j += 1

        lines.insert(j + 1, row)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

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

        # FIXED: number/order of columns must match header!
        global_line = (
            f"| {timestamp} | {model_name} | {embedding} | {use_code} | {use_stats_features} | "
            f"{classifier} | {dataset} | "
            f"{metrics.get('precision_macro', 0):.4f} | {metrics.get('recall_macro', 0):.4f} | "
            f"{metrics.get('f1_macro', 0):.4f} | {metrics.get('hamming_loss', 0):.4f} | {notes} |\n"
        )

        self._insert_row_in_table("| Date |", global_line)

        per_tag = metrics.get("per_tag", [])
        for tag_metrics in per_tag:
            tag_line = (
                f"| {model_name} | {notes} | {tag_metrics.get('tag', '')} | "
                f"{tag_metrics.get('precision', 0):.4f} | {tag_metrics.get('recall', 0):.4f} | "
                f"{tag_metrics.get('f1', 0):.4f} | {tag_metrics.get('support', 0)} | "
                f"{tag_metrics.get('predicted', 0)} |\n"
            )
            self._insert_row_in_table("| Model |", tag_line)

        print(f"===> Results logged to {self.log_path}")

