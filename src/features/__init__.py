# The functions are imported when first used, so that importing src.features (for the INS
# features) doesn't load PyTorch: PyTorch and XGBoost in one process crash on macOS.
import importlib

_MODULES = {
    'process_video_features': 'src.features.video_features',
    'process_imu_features': 'src.features.imu_features',
    'process_ins_features': 'src.features.ins_features',
}

__all__ = list(_MODULES)


def __getattr__(name):
    if name in _MODULES:
        return getattr(importlib.import_module(_MODULES[name]), name)
    raise AttributeError(f"module 'src.features' has no attribute {name!r}")
