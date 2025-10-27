import os
import warnings
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, TypedDict
from threading import Lock
import pandas as pd


class TagMetrics(TypedDict):
    """Type definition for per-tag metrics."""
    tag: str
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int


class ExperimentMetrics(TypedDict, total=False):
    """Type definition for experiment metrics."""
    precision_macro: float
    recall_macro: float
    f1_macro: float
    hamming_loss: float
    per_tag: List[TagMetrics]


class ExperimentLogger:
    """Handles markdown-based experiment tracking with separate tables for global and per-tag metrics.

    Behavior / contract:
    - Creates `results.md` with two sections when missing: "Global Metrics" and "Per-tag Metrics".
    - Appends a single row to the Global Metrics table for each call to `log_result`.
    - Appends one row per tag to the Per-tag Metrics table for each call to `log_result`.
    - If a section is missing from an existing file, it will be appended at the end.
    - Automatically validates metrics before logging
    - Maintains proper data types in CSV exports
    - Handles file operations safely with proper error handling
    - Thread-safe ID generation
    """

    GLOBAL_SECTION = "## Global Metrics"
    PER_TAG_SECTION = "## Per-tag Metrics"

    GLOBAL_HEADER = (
        "| Date | Model | ID | Embedding | Use code | Use stats features | Classifier | Dataset | Precision | Recall | F1 | Hamming | Notes |\n"
    )
    GLOBAL_DIVIDER = (
        "|------|-------|-----|-----------|----------|--------------------|-----------|---------|-----------| -------|----|---------| ------|\n"
    )

    PER_TAG_HEADER = (
        "| Model | ID | Notes | Tag | Precision | Recall | F1 | Support | Predicted |\n"
    )
    PER_TAG_DIVIDER = (
        "|-------|-----|-------|-----|-----------|--------|----|---------|-----------|\n"
    )

    def __init__(self, log_path: Union[str, Path] = "results.md"):
        """Initialize the experiment logger.
        
        Args:
            log_path: Path to the markdown log file
        """
        self.log_path = Path(log_path)
        self.csv_dir = self.log_path.parent / "outputs" / "metrics"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._id_lock = Lock()  # Thread safety for ID generation
        
        try:
            self._init_file()
            self._ensure_sections()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize experiment logger: {e}") from e

    def _init_file(self):
        """Create markdown file with sections if missing."""
        try:
            if not self.log_path.exists():
                # Ensure parent directory exists
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                
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
        except OSError as e:
            raise IOError(f"Failed to create log file {self.log_path}: {e}") from e

    def _ensure_sections(self):
        """Ensure both markdown sections exist, add if missing."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split('\n')
            changed = False
            
            # Check and add missing sections
            if self.GLOBAL_SECTION not in content:
                warnings.warn(f"Global metrics section missing in {self.log_path}, adding it")
                lines.extend([
                    "",
                    self.GLOBAL_SECTION,
                    "",
                    self.GLOBAL_HEADER.rstrip('\n'),
                    self.GLOBAL_DIVIDER.rstrip('\n'),
                    ""
                ])
                changed = True

            if self.PER_TAG_SECTION not in content:
                warnings.warn(f"Per-tag metrics section missing in {self.log_path}, adding it")
                lines.extend([
                    "",
                    self.PER_TAG_SECTION,
                    "",
                    self.PER_TAG_HEADER.rstrip('\n'),
                    self.PER_TAG_DIVIDER.rstrip('\n'),
                    ""
                ])
                changed = True

            if changed:
                self._atomic_write(self.log_path, '\n'.join(lines) + '\n')
                    
        except OSError as e:
            raise IOError(f"Failed to ensure sections in {self.log_path}: {e}") from e

    def _atomic_write(self, filepath: Path, content: str) -> None:
        """Write content to file atomically using temp file.
        
        Args:
            filepath: Target file path
            content: Content to write
        """
        # Create temp file in same directory to ensure same filesystem
        fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f".{filepath.name}.",
            suffix=".tmp"
        )
        
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Atomic rename
            shutil.move(temp_path, filepath)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _insert_row_in_table(self, table_header_start: str, row: str) -> None:
        """Insert a new row right after the existing table entries.
        
        Args:
            table_header_start: The header line that identifies the target table
            row: The formatted markdown table row to insert
            
        Raises:
            IOError: If file operations fail
            ValueError: If table header not found or row format invalid
        """
        if not row.strip().startswith("|") or not row.strip().endswith("|"):
            raise ValueError("Invalid row format - must start and end with |")
            
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find table header
            header_idx = None
            for i, line in enumerate(lines):
                if line.strip() == table_header_start.strip():
                    header_idx = i
                    break

            if header_idx is None:
                warnings.warn(f"Table header '{table_header_start[:20]}...' not found, appending row")
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write("\n" + row)
                return

            # Find table end (last row starting with |)
            j = header_idx
            while j + 1 < len(lines) and lines[j + 1].strip().startswith("|"):
                j += 1

            # Insert row
            lines.insert(j + 1, row)
            
            # Atomic write
            self._atomic_write(self.log_path, ''.join(lines))
                
        except OSError as e:
            raise IOError(f"Failed to insert row in table: {e}") from e

    def _next_model_id(self, model_name: str) -> int:
        """Determine the next numeric ID for a given model by scanning existing logs.
        
        Thread-safe implementation using a lock.
        
        Args:
            model_name: Name of the model to get next ID for
            
        Returns:
            int: Next available numeric ID for the model
            
        Note:
            IDs are formatted as model-### (e.g., bert-003)
        """
        with self._id_lock:
            if not self.log_path.exists():
                return 1

            try:
                max_id = 0
                
                # Only check markdown file (single source of truth)
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.startswith("|"):
                            continue
                        parts = [p.strip() for p in line.split("|")[1:-1]]
                        if len(parts) < 3:
                            continue

                        # Check if this is a row for our model
                        if parts[1] == model_name and parts[2].startswith(f"{model_name}-"):
                            try:
                                num = int(parts[2].split("-")[-1])
                                max_id = max(max_id, num)
                            except (ValueError, IndexError):
                                continue

                return max_id + 1
                
            except OSError as e:
                warnings.warn(f"Error reading model IDs, starting from 1: {e}")
                return 1

    def _validate_string_param(self, value: str, param_name: str) -> None:
        """Validate that a string parameter is non-empty.
        
        Args:
            value: Value to validate
            param_name: Name of parameter for error message
            
        Raises:
            ValueError: If value is empty or not a string
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{param_name} must be a non-empty string")

    def _validate_metrics(self, metrics: ExperimentMetrics) -> None:
        """Validate metrics format and values.
        
        Args:
            metrics: Dictionary containing global and per-tag metrics
            
        Raises:
            ValueError: If metrics are invalid
        """
        # Required global metrics
        required_global = {'precision_macro', 'recall_macro', 'f1_macro', 'hamming_loss'}
        missing = required_global - set(metrics.keys())
        if missing:
            raise ValueError(f"Missing required global metrics: {missing}")
            
        # Validate global metrics are in [0,1]
        for metric in required_global:
            value = metrics.get(metric, 0)
            if not (isinstance(value, (int, float)) and 0 <= value <= 1):
                raise ValueError(f"Invalid {metric} value: {value}, must be float in [0,1]")
                
        # Validate per-tag metrics
        if "per_tag" in metrics:
            if not isinstance(metrics["per_tag"], list):
                raise ValueError("per_tag must be a list")
                
            for tag_metrics in metrics["per_tag"]:
                if not isinstance(tag_metrics, dict):
                    raise ValueError(f"Invalid tag metrics format: {tag_metrics}")
                if "tag" not in tag_metrics:
                    raise ValueError("Missing 'tag' in tag metrics")
                    
                # Validate metric values
                for metric in ['precision', 'recall', 'f1']:
                    value = tag_metrics.get(metric, 0)
                    if not (isinstance(value, (int, float)) and 0 <= value <= 1):
                        raise ValueError(
                            f"Invalid {metric} for tag {tag_metrics['tag']}: {value}, "
                            f"must be in [0,1]"
                        )
                
                # Validate support and predicted are non-negative integers
                for metric in ['support', 'predicted']:
                    value = tag_metrics.get(metric, 0)
                    if not (isinstance(value, int) and value >= 0):
                        raise ValueError(
                            f"Invalid {metric} for tag {tag_metrics['tag']}: {value}, "
                            f"must be non-negative integer"
                        )
                        
    def log_result(
        self,
        model_name: str,
        embedding: str,
        use_code: bool,
        use_stats_features: bool,
        classifier: str,
        dataset: str,
        metrics: ExperimentMetrics,
        notes: str = "",
    ) -> str:
        """Log experiment results with validation.
        
        Args:
            model_name: Name of the model
            embedding: Embedding method used
            use_code: Whether code features were used
            use_stats_features: Whether statistical features were used
            classifier: Name of the classifier
            dataset: Dataset identifier
            metrics: Dictionary containing global and per-tag metrics
            notes: Optional notes about the experiment
            
        Returns:
            str: The experiment ID (e.g., 'bert-001')
            
        Raises:
            ValueError: If parameters or metrics are invalid
            IOError: If logging operations fail
        """
        # Validate string parameters
        self._validate_string_param(model_name, "model_name")
        self._validate_string_param(embedding, "embedding")
        self._validate_string_param(classifier, "classifier")
        self._validate_string_param(dataset, "dataset")
        
        # Validate boolean parameters
        if not isinstance(use_code, bool):
            raise ValueError("use_code must be a boolean")
        if not isinstance(use_stats_features, bool):
            raise ValueError("use_stats_features must be a boolean")
        
        # Validate metrics
        self._validate_metrics(metrics)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        model_id_num = self._next_model_id(model_name)
        log_id = f"{model_name}-{model_id_num:03d}"

        try:
            # --- Global metrics row ---
            global_line = (
                f"| {timestamp} | {model_name} | {log_id} | {embedding} | {use_code} | "
                f"{use_stats_features} | {classifier} | {dataset} | "
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

            return log_id
            
        except Exception as e:
            raise IOError(f"Failed to log results for {log_id}: {e}") from e
        finally:
            # Update CSV files after successful logging
            try:
                self._generate_csv_files()
            except Exception as e:
                warnings.warn(f"Failed to update CSV files: {e}")

    def _generate_csv_files(self) -> None:
        """Generate CSV files for global and per-tag metrics with proper data types."""
        try:
            global_data = []
            per_tag_data = []

            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # --- Extract metrics from tables ---
            section = None
            for line in lines:
                line_stripped = line.strip()
                
                if line_stripped == self.GLOBAL_SECTION:
                    section = "global"
                    continue
                elif line_stripped == self.PER_TAG_SECTION:
                    section = "per_tag"
                    continue

                if not line.startswith("|") or "------" in line or not line_stripped:
                    continue

                parts = [x.strip() for x in line.strip().split("|")[1:-1]]

                if section == "global" and len(parts) == 13:
                    # Skip header row
                    if parts[0] != "Date":
                        global_data.append(parts)
                elif section == "per_tag" and len(parts) == 9:
                    # Skip header row
                    if parts[0] != "Model":
                        per_tag_data.append(parts)

            # --- Save global metrics CSV with proper types ---
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
                
                # Convert data types
                global_df["Date"] = pd.to_datetime(global_df["Date"], errors='coerce')
                global_df["Use code"] = global_df["Use code"].map(
                    {"True": True, "False": False}
                ).fillna(False)
                global_df["Use stats features"] = global_df["Use stats features"].map(
                    {"True": True, "False": False}
                ).fillna(False)
                
                # Convert numeric columns
                numeric_cols = ["Precision", "Recall", "F1", "Hamming"]
                for col in numeric_cols:
                    global_df[col] = pd.to_numeric(global_df[col], errors="coerce")
                
                csv_path = self.csv_dir / "global_metrics.csv"
                global_df.to_csv(csv_path, index=False)

            # --- Save per-tag metrics CSV with proper types ---
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
                
                # Convert numeric columns
                numeric_cols = ["Precision", "Recall", "F1", "Support", "Predicted"]
                for col in numeric_cols:
                    per_tag_df[col] = pd.to_numeric(per_tag_df[col], errors="coerce")
                
                csv_path = self.csv_dir / "per_tag_metrics.csv"
                per_tag_df.to_csv(csv_path, index=False)
                
        except Exception as e:
            raise IOError(f"Failed to generate CSV files: {e}") from e
            
    def get_model_metrics(self, model_id: Optional[str] = None) -> pd.DataFrame:
        """Get metrics for a specific model ID or all models.
        
        Args:
            model_id: Optional model ID to filter by (e.g., 'bert-001')
            
        Returns:
            pd.DataFrame: DataFrame containing the metrics
        """
        try:
            csv_path = self.csv_dir / "global_metrics.csv"
            if not csv_path.exists():
                return pd.DataFrame()
                
            df = pd.read_csv(csv_path)
            if model_id:
                return df[df["ID"] == model_id]
            return df
            
        except Exception as e:
            warnings.warn(f"Error reading metrics: {e}")
            return pd.DataFrame()