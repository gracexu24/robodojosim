from __future__ import annotations

import cv2
import numpy as np

_RGB_MARKER_PAYLOAD = b"XPL-RGB1"


def encode_xpolicylab_jpeg(image: np.ndarray, quality: int = 90) -> np.ndarray:
    """Encode RGB as the marked JPEG stream expected by XPolicyLab."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 uint8 RGB image, got {array.shape} {array.dtype}")
    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not success:
        raise ValueError("OpenCV failed to encode JPEG")
    jpeg = encoded.tobytes()
    marker = b"\xff\xfe" + (len(_RGB_MARKER_PAYLOAD) + 2).to_bytes(2, "big") + _RGB_MARKER_PAYLOAD
    offset = 2
    if len(jpeg) >= 6 and jpeg[2:4] == b"\xff\xe0":
        app0_end = 4 + int.from_bytes(jpeg[4:6], "big")
        if app0_end <= len(jpeg):
            offset = app0_end
    return np.frombuffer(jpeg[:offset] + marker + jpeg[offset:], dtype=np.uint8)


def decode_jpeg(buffer: np.ndarray) -> np.ndarray:
    """Decode one marked standard JPEG into RGB (used by portable tests/tools)."""

    image = cv2.imdecode(np.asarray(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV failed to decode JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
