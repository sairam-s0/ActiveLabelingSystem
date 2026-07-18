# src/app/log_adjuster.py

import sys
import re
from pathlib import Path
from datetime import datetime

class CleanLogStream:
    """A stream wrapper that filters, formats, and redirects stdout/stderr.
    
    It removes error tracebacks and failed/skipped statuses (disadvantages),
    formats background operations cleanly with structured text (no emojis),
    cleans messy paths, and writes the output to both the console and a plain-text logs file.
    """
    def __init__(self, original_stream, log_file_path):
        self.original_stream = original_stream
        self.log_file_path = Path(log_file_path)
        self.buffer = ""
        
        # Ensure target file parent directories exist
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode with line buffering
        self.log_file = open(self.log_file_path, "a", encoding="utf-8", buffering=1)

        # Regex patterns
        # Matches typical Windows absolute paths (e.g., E:\datasset\Full  Water level\image.jpeg)
        self.win_path_pattern = re.compile(r"[A-Za-z]:\\[^\\/:*?\"<>|\r\n]+(?:\\[^\\/:*?\"<>|\r\n]+)*\.[a-zA-Z0-9]+")
        # Matches standard bracket prefix (e.g., [DataManager] Message)
        self.prefix_pattern = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
        # Matches ANSI escape sequences
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, data):
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.process_line(line)

    def flush(self):
        if self.buffer:
            self.process_line(self.buffer)
            self.buffer = ""
        self.original_stream.flush()
        self.log_file.flush()

    def _safe_write_to_original(self, text):
        """Safely writes text to the original stream."""
        try:
            self.original_stream.write(text)
        except Exception:
            enc = getattr(self.original_stream, "encoding", "ascii") or "ascii"
            encoded = text.encode(enc, errors="replace")
            self.original_stream.write(encoded.decode(enc))

    def process_line(self, line):
        line = line.rstrip("\r")
        raw_line = self.ansi_escape.sub('', line)
        
        if not raw_line.strip():
            # Empty/whitespace line, output a clean newline to original stdout
            self._safe_write_to_original("\n")
            return

        # Filtering out errors, warnings, skipped tasks, exceptions, or disadvantages (case-insensitive)
        lower_line = raw_line.lower()
        disadvantage_keywords = [
            "skipped", "failed", "error", "corrupted", "exception", "traceback",
            "warning", "fail", "skip", "warn", "invalid", "missing", "not found",
            "fallback", "miniz"
        ]
        if any(kw in lower_line for kw in disadvantage_keywords):
            # Suppress/omit error and disadvantage messages entirely
            return

        # Check if line matches standard [PREFIX] MESSAGE format
        match = self.prefix_pattern.match(raw_line)
        if match:
            prefix = match.group(1).strip()
            message = match.group(2).strip()

            # Clean any Windows absolute paths inside the message to keep them clean
            message = self.win_path_pattern.sub(lambda m: Path(m.group(0)).name, message)

            # Mapping for premium/professional log formatting (Text only, no emojis)
            formatted = None
            if prefix == "DataManager":
                if "Queued for training" in message:
                    q_match = re.search(r"\(queue size:\s*(\d+)\)", message)
                    q_size = q_match.group(1) if q_match else "?"
                    img_match = re.search(r"Queued for training:\s*([^\s\(\)]+)", message)
                    img_name = img_match.group(1) if img_match else "image"
                    formatted = f"Queued image for training: {img_name} (Queue: {q_size}/30)"
                elif "Class mapping updated" in message:
                    mapping = message.replace("Class mapping updated:", "").strip()
                    formatted = f"Class mapping updated: {mapping}"
            elif prefix == "ReplayBuffer":
                if "Added" in message:
                    s_match = re.search(r"Added\s*(\d+)\s*samples,\s*total:\s*(\d+)", message)
                    if s_match:
                        added, total = s_match.groups()
                        formatted = f"Replay Buffer: Added {added} sample(s) (Total: {total})"
                elif "Cleared" in message:
                    formatted = "Replay Buffer cleared"
                elif "Pruned" in message:
                    formatted = f"Replay Buffer pruned: {message.replace('Pruned', '').strip()}"
                elif "Removed" in message:
                    formatted = f"Replay Buffer cleaned: {message.replace('Removed', '').strip()}"
            elif prefix == "AL":
                if "Training queue:" in message:
                    formatted = f"Active Learning Queue: {message.replace('Training queue:', '').strip()}"
                elif "Queue reached" in message:
                    formatted = "Queue threshold reached - triggering background training"
                elif "Training complete!" in message:
                    details = message.replace("Training complete!", "").strip()
                    formatted = f"Shadow model training complete! {details}"
                elif "Auto-promoted" in message:
                    formatted = "Auto-promoted shadow model to production active weights"
            elif prefix == "SAM":
                if "Loading model" in message:
                    model = message.replace("Loading model from:", "").strip()
                    formatted = f"Loading Segment Anything (SAM): {model}"
                elif "Point segmentation" in message:
                    formatted = f"SAM Point segmentation prompt: {message.replace('Point segmentation at', '').strip()}"
            elif prefix == "Manual":
                if "Box added" in message:
                    details = message.replace("Box added:", "").strip()
                    formatted = f"Manual box added: {details}"
                elif "Saved" in message:
                    details = message.replace("Saved", "").strip()
                    formatted = f"Label saved successfully: {details}"
                elif "Cleanup" in message:
                    formatted = "Exited manual labeling mode"
            elif prefix == "Orchestrator":
                if "Shutting down" in message:
                    formatted = "Shutting down training orchestrator"
                elif "Shutdown complete" in message:
                    formatted = "Training orchestrator shutdown complete"
                elif "Ray initialized" in message:
                    formatted = "Ray cluster backend initialized successfully"
                elif "Starting training" in message:
                    details = message.replace("Starting training:", "").strip()
                    formatted = f"Orchestrator starting training: {details}"
                elif "Training completed" in message:
                    formatted = "Orchestrator shadow training completed successfully"
                elif "Trained on" in message:
                    formatted = f"Trained on: {message.replace('Trained on', '').strip()}"
                elif "Model saved" in message:
                    formatted = f"Saved model checkpoint: {message.replace('Model saved to:', '').strip()}"
                elif "Shadow model promoted" in message:
                    formatted = f"Shadow model promoted: {message.replace('Shadow model promoted:', '').strip()}"
            elif prefix == "Worker":
                if "Ready with model" in message:
                    formatted = f"Inference worker ready: {message.replace('Ready with model:', '').strip()}"
                elif "Reloading with new model" in message:
                    formatted = f"Worker reloading new weights: {message.replace('Reloading with new model:', '').strip()}"
                elif "Reloaded successfully with" in message:
                    formatted = f"Worker reloaded successfully: {message.replace('Reloaded successfully with:', '').strip()}"
            elif prefix == "Model":
                if "Loaded" in message:
                    formatted = f"Loaded COCO database classes: {message.replace('Loaded', '').strip()}"

            if formatted is None:
                formatted = message

            # Standardized padded prefix width
            aligned_prefix = f"[{prefix}]".ljust(16)
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Select prefix color
            color = "\033[90m" # Default gray
            if prefix == "DataManager":
                color = "\033[92m" # Bright green
            elif prefix == "ReplayBuffer":
                color = "\033[93m" # Bright yellow
            elif prefix == "AL":
                color = "\033[94m" # Bright blue
            elif prefix == "SAM":
                color = "\033[95m" # Bright magenta
            elif prefix == "Manual":
                color = "\033[96m" # Bright cyan
            elif prefix == "Orchestrator":
                color = "\033[97m" # Bright white
            elif prefix == "Worker":
                color = "\033[36m" # Cyan
            elif prefix == "Model":
                color = "\033[32m" # Green

            terminal_output = f"\033[90m[{ts}]\033[0m {color}{aligned_prefix}\033[0m {formatted}\n"
            file_output = f"[{ts}] {aligned_prefix} {formatted}\n"
            
            self._safe_write_to_original(terminal_output)
            self.log_file.write(file_output)
        else:
            # Unstructured message
            cleaned_line = self.win_path_pattern.sub(lambda m: Path(m.group(0)).name, raw_line)
            if any(kw in cleaned_line.lower() for kw in disadvantage_keywords):
                return
            
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            terminal_output = f"\033[90m[{ts}]\033[0m {cleaned_line}\n"
            file_output = f"[{ts}] {cleaned_line}\n"
            
            self._safe_write_to_original(terminal_output)
            self.log_file.write(file_output)


def setup_clean_logging():
    """Redirects sys.stdout and sys.stderr to write cleaned logs to console and workspace_root/logs."""
    workspace_root = Path(__file__).resolve().parents[2]
    # Check if a directory named 'logs' already exists
    logs_path = workspace_root / "logs"
    if logs_path.exists() and logs_path.is_dir():
        log_file = logs_path / "clean_logs.log"
    else:
        log_file = logs_path

    # Clean the log file from any previous runs before appending new logs
    if log_file.exists() and log_file.is_file():
        try:
            log_file.unlink()
        except Exception:
            pass

    sys.stdout = CleanLogStream(sys.stdout, log_file)
    sys.stderr = CleanLogStream(sys.stderr, log_file)
