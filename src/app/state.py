# app/state.py
"""
Global application state management - FIXED
Centralized storage for all runtime data
"""

import threading
from pathlib import Path

# Image state
current_image = None
current_image_path = None
current_index = 0
_current_img_size = (0, 0)

# File management
image_files = []

# Detection state
current_detections = []
scale_factor = None
last_image_entropy = 0.0

# Class configuration
selected_classes = []
custom_classes = []
coco_classes = []

# Label storage
labels = {}
auto_accepted_log = []
detection_errors = []
label_format = None
labels_output_dir = None
labels_output_path = None

# Training state
training_status = {}
class_samples = {}

# Worker state
worker_ready = False
executor = None

# Configuration
threshold = 70
qa_rate = 0.01
max_dim = 1600
weights = 'yolov8m' \
'.pt'

device = 'cpu'
has_gpu = False

# Autosave
autosave_file = None
autosave_lock = threading.Lock()

# UI flags
show_manual_instructions = True

# Training config
min_training_samples = 30


def reset_state():
    """Reset all state variables to defaults."""
    global current_image, current_image_path, current_index, _current_img_size
    global image_files, current_detections, scale_factor, last_image_entropy
    global selected_classes, custom_classes, labels, auto_accepted_log, detection_errors
    global label_format, labels_output_dir, labels_output_path
    global class_samples
    
    current_image = None
    current_image_path = None
    current_index = 0
    _current_img_size = (0, 0)
    
    image_files = []
    current_detections = []
    scale_factor = None
    last_image_entropy = 0.0
    
    selected_classes = []
    custom_classes = []
    
    labels = {}
    auto_accepted_log = []
    detection_errors = []
    label_format = None
    labels_output_dir = None
    labels_output_path = None
    
    class_samples = {}


def get_progress_stats():
    """Calculate labeling progress statistics."""
    total = len(image_files) if image_files else 0
    labeled = len(labels) if labels else 0
    return {
        'total': total,
        'labeled': labeled,
        'remaining': total - labeled,
        'percent': (labeled / total * 100) if total > 0 else 0
    }
