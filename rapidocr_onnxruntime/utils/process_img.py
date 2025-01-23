# -*- encoding: utf-8 -*-
# @Author: MTF / rinor4ever
# @Contact: r4ajeti@gmail.com
from typing import Tuple

import numpy as np

def reduce_max_side(
    img: np.ndarray, max_side_len: int = 2000
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    ratio = 1.0
    
    if max(h, w) > max_side_len:
        ratio = max_side_len / max(h, w)
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
    else:
        resize_h = h
        resize_w = w

    # Round to nearest multiple of 32 using integer arithmetic
    resize_h = (resize_h + 16) // 32 * 32
    resize_w = (resize_w + 16) // 32 * 32

    if resize_w <= 0 or resize_h <= 0:
        raise ValueError("resize_w or resize_h is less than or equal to 0")

    try:
        from PIL import Image
        # Determine image mode
        if img.ndim == 3:
            mode = 'RGB' if img.shape[2] == 3 else 'RGBA'
        else:
            mode = 'L'
        
        pil_img = Image.fromarray(img.astype(np.uint8), mode=mode)
        pil_img = pil_img.resize((resize_w, resize_h), Image.BILINEAR)
        resized_img = np.array(pil_img)
    except ImportError:
        # Fallback to numpy-based nearest-neighbor interpolation
        scale_x = resize_w / w
        scale_y = resize_h / h
        x = np.linspace(0, w-1, resize_w)
        y = np.linspace(0, h-1, resize_h)
        xv, yv = np.meshgrid(x, y)
        resized_img = img[np.floor(yv).astype(int), np.floor(xv).astype(int)]

    ratio_h = h / resize_h
    ratio_w = w / resize_w
    return resized_img, ratio_h, ratio_w


def increase_min_side(
    img: np.ndarray, min_side_len: int = 30
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    ratio = 1.0
    
    if min(h, w) < min_side_len:
        if h < w:
            ratio = min_side_len / h
        else:
            ratio = min_side_len / w
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
    else:
        resize_h = h
        resize_w = w

    # Round to nearest multiple of 32 using integer arithmetic
    resize_h = (resize_h + 16) // 32 * 32
    resize_w = (resize_w + 16) // 32 * 32

    if resize_w <= 0 or resize_h <= 0:
        raise ValueError("resize_w or resize_h is less than or equal to 0")

    try:
        from PIL import Image
        # Determine image mode based on array shape
        if img.ndim == 3:
            mode = 'RGB' if img.shape[2] == 3 else 'RGBA'
        else:
            mode = 'L'
        
        pil_img = Image.fromarray(img.astype(np.uint8), mode=mode)
        pil_img = pil_img.resize((resize_w, resize_h), Image.BILINEAR)
        resized_img = np.array(pil_img)
    except ImportError:
        # Numpy fallback with nearest-neighbor interpolation
        x = np.linspace(0, w-1, resize_w)
        y = np.linspace(0, h-1, resize_h)
        xv, yv = np.meshgrid(x, y)
        resized_img = img[np.floor(yv).astype(int), np.floor(xv).astype(int)]

    ratio_h = h / resize_h
    ratio_w = w / resize_w
    return resized_img, ratio_h, ratio_w


def add_round_letterbox(
    img: np.ndarray,
    padding_tuple: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Adds letterbox padding to an image using numpy.
    Padding order: (top, bottom, left, right)
    """
    top, bottom, left, right = padding_tuple
    
    # Handle color vs grayscale images
    if img.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))
    
    # Create padded array with black borders (zeros)
    padded_img = np.pad(
        img,
        pad_width=pad_width,
        mode='constant',
        constant_values=0
    )
    
    return padded_img


class ResizeImgError(Exception):
    pass
