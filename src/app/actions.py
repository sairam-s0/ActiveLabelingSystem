# src/app/actions.py

import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from app import state


def accept(app):
    if state.current_detections:
        # save with
        if hasattr(app, 'data_manager'):
            img_path = str(state.current_image_path)
            img_w, img_h = state._current_img_size
            
            app.data_manager.save_labels(
                image_path=img_path,
                detections=state.current_detections,
                entropy=state.last_image_entropy,
                img_width=img_w,
                img_height=img_h
            )
            
            # check if
            app.orchestrator.check_training_trigger()
        else:
            # fallback to
            save_label(app, state.current_detections, auto=False)
    
    state.current_index += 1
    app.process_next()


def reject(app):
    try:
        app.save_autosave()
    except Exception:
        pass
    state.current_index += 1
    app.process_next()


def skip(app):
    try:
        app.save_autosave()
    except Exception:
        pass
    state.current_index += 1
    app.process_next()


def save_label(app, detections, auto=False):
    img_path = str(state.image_files[state.current_index])
    img_w, img_h = state._current_img_size
    
    clean_dets = []
    for d in (detections or []):
        clean_dets.append({
            'bbox': [int(round(x)) for x in d['bbox']],
            'confidence': round(d.get('confidence', 0.0), 2),
            'class': d.get('class', state.selected_classes[0] if state.selected_classes else 'unknown')
        })
    
    state.labels[img_path] = {
        'detections': clean_dets,
        'auto': bool(auto),
        'timestamp': datetime.now().isoformat(),
        'image_width': img_w,
        'image_height': img_h
    }
    
    try:
        app.update_stats()
        app.save_autosave()
    except Exception:
        pass


def show_log(app):
    if not state.auto_accepted_log:
        QMessageBox.information(app, "Log", "No auto-accepted images yet")
        return
    
    msg = "\n".join([Path(p).name for p in state.auto_accepted_log[-20:]])
    QMessageBox.information(app, "Auto-accepted (last 20)", msg)


def export_json(app):
    if not state.labels:
        QMessageBox.warning(app, "No Data", "No labels to export")
        return
    
    file, _ = QFileDialog.getSaveFileName(app, "Export JSON", "", "JSON Files (*.json)")
    if file:
        with open(file, 'w') as f:
            json.dump(state.labels, f, indent=2)
        QMessageBox.information(app, "Success", f"Exported {len(state.labels)} labels")


def export_coco(app):
    if not state.labels:
        QMessageBox.warning(app, "No Data", "No labels to export")
        return
    
    file, _ = QFileDialog.getSaveFileName(app, "Export COCO", "", "JSON Files (*.json)")
    if not file:
        return
    
    coco = {'images': [], 'annotations': [], 'categories': []}
    class_to_id = {}
    cat_id = 1
    
    # build categories
    for img_path, ld in state.labels.items():
        for det in ld.get('detections', []):
            cls_name = det.get('class', 'unknown')
            if cls_name not in class_to_id:
                class_to_id[cls_name] = cat_id
                coco['categories'].append({
                    'id': cat_id,
                    'name': cls_name,
                    'supercategory': 'object'
                })
                cat_id += 1
    
    # build images
    img_id = 1
    ann_id = 1
    for img_path, ld in state.labels.items():
        coco['images'].append({
            'id': img_id,
            'file_name': Path(img_path).name,
            'width': ld.get('image_width', 0),
            'height': ld.get('image_height', 0)
        })
        
        for det in ld.get('detections', []):
            bbox = det['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            cls_name = det.get('class', 'unknown')
            cat_id = class_to_id.get(cls_name, 1)
            
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [bbox[0], bbox[1], w, h],
                'area': w * h,
                'iscrowd': 0
            })
            ann_id += 1
        img_id += 1
    
    with open(file, 'w') as f:
        json.dump(coco, f, indent=2)
    
    QMessageBox.information(
        app,
        "Success",
        f"COCO exported with {len(coco['categories'])} classes"
    )


def promote_shadow_model(app):
    if not hasattr(app, 'orchestrator'):
        QMessageBox.warning(app, "Error", "Training system not initialized")
        return
    
    # confirm promotion
    reply = QMessageBox.question(
        app,
        "Promote Shadow Model",
        "Promote shadow model to active?\n\n"
        "This will make it the default for inference.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    
    if reply != QMessageBox.StandardButton.Yes:
        return
    
    # attempt promotion
    result = app.orchestrator.promote_shadow_model(validate=True)
    
    if result['success']:
        QMessageBox.information(
            app,
            "Success",
            f"Shadow model promoted!\n\n"
            f"Version: {result['version']}\n"
            f"Path: {result['path']}\n\n"
            f"Restart the app to use the new model."
        )
    elif result.get('requires_confirmation'):
        # validation failed
        reply = QMessageBox.question(
            app,
            "Validation Warning",
            f"Model validation detected issues:\n\n{result['error']}\n\n"
            f"Promote anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            result = app.orchestrator.promote_shadow_model(validate=False)
            if result['success']:
                QMessageBox.information(app, "Success", "Model promoted!")
    else:
        QMessageBox.critical(
            app,
            "Error",
            f"Promotion failed:\n\n{result['error']}"
        )