"""Optional TensorRT backend. Imported by serve.py only when USE_TRT=1."""

import tensorrt


def build_engine(model_name):
    """Stub: a real service would build a TensorRT engine for the model here."""
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    raise NotImplementedError(f"TensorRT engine build for {model_name} is not implemented ({logger})")
