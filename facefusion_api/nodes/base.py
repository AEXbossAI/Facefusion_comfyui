"""
Base imports and utilities for all ComfyUI nodes.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Tuple, Optional, List, Dict, Any

import torch
import numpy as np
from comfy.comfy_types import IO
from comfy_api.input_impl.video_types import VideoFromComponents
from comfy_api.util import VideoComponents
from PIL import Image as _PIL_Image

def bytesio_to_image_tensor(buffer):
    import numpy as np, torch
    img = _PIL_Image.open(buffer).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_bytesio(tensor, mime_type="image/png"):
    import numpy as np, io as _io
    arr = (tensor.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    img = _PIL_Image.fromarray(arr)
    buf = _io.BytesIO()
    fmt = {"image/png": "PNG", "image/webp": "WEBP", "image/jpeg": "JPEG"}.get(mime_type, "PNG")
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

from httpx import Client as HttpClient, Headers
from httpx_retries import Retry, RetryTransport
from torch import Tensor

from ..types import FaceSwapperModel, InputTypes
from ..utils import tensor_to_cv2, cv2_to_tensor, get_average_embedding, implode_pixel_boost, explode_pixel_boost
from ..detection import detect_faces, select_faces
from ..swap_local import swap_faces_local
from ..models import get_local_swapper, get_face_occluder, get_face_parser, MODEL_CONFIGS
from .content_filter_utils import analyse_frame, blur_frame, CONTENT_FILTER_AVAILABLE

__all__ = [
    # Standard library
    'ThreadPoolExecutor',
    'partial',
    'BytesIO',
    'Tuple',
    'Optional',
    'List',
    'Dict',
    'Any',
    # PyTorch
    'torch',
    'Tensor',
    'np',
    # ComfyUI
    'IO',
    'VideoFromComponents',
    'VideoComponents',
    'bytesio_to_image_tensor',
    'tensor_to_bytesio',
    'HttpClient',
    'Headers',
    'Retry',
    'RetryTransport',
    # Our types
    'FaceSwapperModel',
    'InputTypes',
    # Our utilities
    'tensor_to_cv2',
    'cv2_to_tensor',
    'get_average_embedding',
    'implode_pixel_boost',
    'explode_pixel_boost',
    'detect_faces',
    'select_faces',
    'swap_faces_local',
    'get_local_swapper',
    'get_face_occluder',
    'get_face_parser',
    'MODEL_CONFIGS',
    'analyse_frame',
    'blur_frame',
    'CONTENT_FILTER_AVAILABLE',
]













