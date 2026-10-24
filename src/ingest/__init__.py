# The functions are imported when first used (see src/features/__init__.py)
import importlib

_MODULES = {
    'ingest_video_frames': 'src.ingest.video_ingest',
    'ingest_imu_data': 'src.ingest.imu_ingest',
    'ingest_ins_data': 'src.ingest.ins_ingest',
}

__all__ = list(_MODULES)


def __getattr__(name):
    if name in _MODULES:
        return getattr(importlib.import_module(_MODULES[name]), name)
    raise AttributeError(f"module 'src.ingest' has no attribute {name!r}")
