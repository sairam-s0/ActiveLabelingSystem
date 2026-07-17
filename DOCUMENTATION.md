# Active Labeling System - Codebase & Architecture Guide

This document provides a single, consolidated reference for the codebase structure, GUI components, styling systems, manual labeling mechanisms, and keyboard shortcut integrations.

---

## 1. Codebase Structure & Architecture

The application is structured into three primary directories:
- **`src/app/`**: Core GUI state management, dialog implementations, utils, and main window views.
- **`src/features/`**: Specific application feature managers, including manual labeling overlays, floating toolboxes, toolbars, and shortcut keyboard binders.
- **`src/core/`**: Active learning logic, model inference engines, sample selection strategies, replay buffers, and dataset version control mechanisms.

```text
src/
  ├── main.py                # Application entry point, global stylesheet initialization
  ├── app/
  │   ├── state.py           # Global state values (e.g., active weights, image indices, cache)
  │   ├── theme.py           # Unified slate-dark theme configuration palette
  │   ├── window.py          # MainWindow, TopControlBar, LeftSidePanel, BottomActionBar
  │   ├── dialogs.py         # Format/Class selectors and stats/version lists QDialogs
  │   └── utils.py           # PIL-to-PyQt transformations, bounding box helpers
  └── features/
      ├── manual.py          # ManualManager, DrawingOverlay canvas, and ManualToolbox dialog
      ├── shortcut_manager.py # Keyboard shortcut dispatcher
      └── toolbar_manager.py  # Floating context toolbar manager (for adding/deleting shapes)
```

---

## 2. Slate-Dark UI & Font Styling System

To prevent OS dark/light mode conflicts (e.g., white-on-white text rendering), the application uses an explicit, uniform slate-dark palette defined in `src/app/theme.py`:

- **Background (`COLORS['bg']` / `#0f172a`)**: Slate-900 for the main application workspace.
- **Cards/Panels (`COLORS['panel']` / `#1e293b`)**: Slate-800 for context panels, side drawers, and dialog dialogs.
- **Text (`COLORS['text']` / `#f8fafc`)**: Slate-50 high-contrast white for all user-facing copy.
- **Borders (`COLORS['border']` / `#334155`)**: Slate-700 for layout separation borders.

A comprehensive QSS stylesheet is loaded on startup in `src/main.py`. This stylesheet binds layout dimensions, border-radius constraints, color rules, hover styles, and drop-down item styling directly to Qt components.

---

## 3. Manual Labeling & Freehand Region Mechanics

Manual labeling mode (`src/features/manual.py`) allows interactive labeling of images via three distinct sub-modes:

### 1. Bounding Box Mode (`box`)
Allows users to draw bounding boxes on the labeling canvas by clicking and dragging. Deterministic color maps are computed based on the class hash value.

### 2. Freehand Region Mode (`polygon`)
Allows custom region segmentations. Users click on the canvas to add vertices, and the polygon is rendered on-screen via a path overlay. Double-clicking or clicking close to the starting point closes the polygon.
- **Point Undo Action**: Users can press `Backspace`, `Ctrl+Z`, or click the floating **"Undo Point"** button inside the manual toolbox to remove the last added vertex.
- **Region Completion**: Click the floating **"Finish Region"** button (or right-click) to complete the current polygon and save it as a polygon shape.

### 3. SAM Adaptation (`sam`)
Utilizes Segment Anything (SAM) points/detections to predict and refine regional boundaries on demand.

---

## 4. Keyboard Shortcuts Binders

Keyboard shortcuts (`src/features/shortcut_manager.py`) streamline the speed of operations:

| Key Binding | Action | Scope |
| :--- | :--- | :--- |
| **`Space` / `Enter`** | Save current detections and advance to the next image. | General Labeling & Manual Mode |
| **`Esc`** | Exit manual labeling mode completely (with confirm prompt). | Manual Mode |
| **`Ctrl + Z`** | Undo last point in the active polygon, or undo the last drawn box. | Manual Mode |
| **`Backspace`** | Undo last point in the active polygon. | Manual Mode (Freehand Region) |
| **`Delete`** | Delete the last completed box/shape. | Manual Mode |
| **`1 - 9`** | Switch active manual labeling class directly. | Manual Mode |
| **`A` / `R` / `N` / `M`** | Fast-trigger Accept, Reject, Skip, or Manual mode. | General Review View |
