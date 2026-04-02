# src/core/model_manager.py
import shutil
import os
import tempfile
from pathlib import Path
from datetime import datetime

# lazy import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ModelManager:
    def __init__(self, models_dir="models", base_model_name="yolov8m.pt"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.base_model_name = base_model_name
        self.active_symlink = self.models_dir / "active_model.pt"
        self._cached_model_path = None  # cache for
        
        # ensure versions
        (self.models_dir / "versions").mkdir(exist_ok=True)

    def _is_temp_directory(self, path: Path) -> bool:
        temp_prefixes = [tempfile.gettempdir()]
        str_path = str(path.resolve())
        return any(str_path.startswith(prefix) for prefix in temp_prefixes)

    def _find_project_root(self, start_path: Path = None) -> Path:
        if start_path is None:
            start_path = Path.cwd()
        
        current = start_path.resolve()
        fs_root = Path(current.root)
        
        while current != fs_root:
            # check for
            if (current / ".git").exists():
                return current
            if (current / "src").exists():
                return current
            if (current / "requirements.txt").exists():
                return current
            if current.name == "dynamic":
                return current
            current = current.parent
        
        return fs_root  # fallback to

    def find_model_file(self, model_name: str) -> str:
        if self._cached_model_path:
            return self._cached_model_path

        # quick search
        search_paths = [
            self.models_dir / model_name,          # 1 models
            Path.cwd() / model_name,               # 2 current
            Path.cwd().parent / model_name,        # 3 parent
        ]
        
        for path in search_paths:
            if path.exists():
                self._cached_model_path = str(path.resolve())
                return self._cached_model_path

        # skip deep
        if self._is_temp_directory(Path.cwd()):
            return None

        # deep recursive
        try:
            project_root = self._find_project_root()
            # search in
            for found in project_root.rglob(f"**/{model_name}"):
                if found.is_file():
                    self._cached_model_path = str(found.resolve())
                    return self._cached_model_path
        except Exception as e:
            print(f"[ModelManager] Deep search failed: {e}")

        return None

    def download_model(self, model_name: str) -> str:
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO not available for auto-download")
        
        print(f"[ModelManager] Downloading {model_name}...")
        try:
            # this downloads
            model = YOLO(model_name)
            downloaded_path = Path.cwd() / model_name
            
            if not downloaded_path.exists():
                raise FileNotFoundError(f"Downloaded model not found at {downloaded_path}")
            
            # move to
            target_path = self.models_dir / model_name
            shutil.move(str(downloaded_path), str(target_path))
            
            print(f"[ModelManager] Downloaded {model_name} to {target_path}")
            return str(target_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_name}: {e}")

    def get_active_model(self) -> str:
        if not self.active_symlink.exists():
            # try to
            found_path = self.find_model_file(self.base_model_name)
            
            if found_path is None:
                # auto download
                if YOLO_AVAILABLE:
                    found_path = self.download_model(self.base_model_name)
                else:
                    raise FileNotFoundError(
                        f"Base model '{self.base_model_name}' not found. "
                        f"Searched in models/, current dir, and project directories. "
                        f"Please place it in the 'models/' directory or install ultralytics for auto-download."
                    )
            
            # ensure base
            self.base_model = Path(found_path)
            
            # create initial
            try:
                os.symlink(self.base_model.resolve(), self.active_symlink)
            except OSError:
                # windows fallback
                shutil.copy2(self.base_model, self.active_symlink)
                
        return str(self.active_symlink)

    def resolve_active_path(self) -> str:
        # ensure active
        self.get_active_model()  # side effect

        active_path = self.active_symlink

        # check if
        if active_path.is_symlink():
            try:
                resolved = active_path.resolve()
                if resolved.exists():
                    return str(resolved)
                else:
                    # broken symlink
                    return str(active_path)
            except Exception:
                # fallback on
                return str(active_path)
        else:
            # not a
            return str(active_path)

    def promote_shadow(self, shadow_path: str) -> dict:
        shadow_file = Path(shadow_path)
        if not shadow_file.exists():
            return {'success': False, 'error': 'Shadow file not found'}

        # 1 version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"v_{timestamp}.pt"
        target_path = self.models_dir / "versions" / version_name
        
        shutil.copy2(shadow_file, target_path)

        # 2 update
        try:
            # create temp
            tmp_link = self.models_dir / "tmp_active.pt"
            if tmp_link.exists():
                os.remove(tmp_link)
            
            os.symlink(target_path.resolve(), tmp_link)
            os.replace(tmp_link, self.active_symlink)
            
        except OSError:
            # fallback for
            shutil.copy2(target_path, self.active_symlink)

        return {
            'success': True,
            'version': version_name,
            'path': str(target_path),
            'timestamp': timestamp
        }

    def rollback(self, specific_version: str = None) -> dict:
        versions_dir = self.models_dir / "versions"
        if not versions_dir.exists():
            return {'success': False, 'error': 'No versions directory'}

        versions = sorted(versions_dir.glob("*.pt"), key=os.path.getmtime)
        
        if not versions:
            return {'success': False, 'error': 'No versions to rollback to'}

        target = None
        if specific_version:
            # find specific
            for v in versions:
                if v.name == specific_version:
                    target = v
                    break
        else:
            # rollback to
            if len(versions) >= 2:
                target = versions[-2]
            else:
                target = versions[0]  # fallback to

        if target is None:
            return {'success': False, 'error': 'Target version not found'}

        # update symlink
        try:
            tmp_link = self.models_dir / "tmp_active.pt"
            if tmp_link.exists():
                os.remove(tmp_link)
            os.symlink(target.resolve(), tmp_link)
            os.replace(tmp_link, self.active_symlink)
        except OSError:
            shutil.copy2(target, self.active_symlink)

        return {'success': True, 'target': target.name}

    def list_versions(self) -> list:
        versions_dir = self.models_dir / "versions"
        versions = versions_dir.glob("*.pt")
        return [
            {
                "name": v.name,
                "size_mb": round(v.stat().st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(v.stat().st_ctime).isoformat()
            }
            for v in sorted(versions, key=os.path.getmtime, reverse=True)
        ]