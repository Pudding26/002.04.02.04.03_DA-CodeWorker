import numpy as np
from PIL import Image
import logging


def _create_stack_and_opt_crop(images: list[np.ndarray]) -> tuple[np.ndarray, bool, str]:
    """
    Handles a list (or singleton) of images and returns a consistent image stack.

    Returns:
        - stacked ndarray of shape (N, H, W) or (N, H, W, 3)
        - bool flag: whether cropping/resizing was applied
        - string: "GS" for grayscale or "RGB" for color
    """
    # Handle special case: already a stacked or single image array
    if len(images) == 1 and isinstance(images[0], np.ndarray):
        return _handle_single_input_array(images[0])

    # Handle list of images
    return _handle_list_of_images(images)


# ────────────────────────────────────────────────────────────────
# 📦 Case 1: Handle single array input (e.g. already stacked)
# ────────────────────────────────────────────────────────────────

def _handle_single_input_array(img: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """
    Detect if the single image is already a stacked array.
    """
    if img.ndim == 2:
        # Single grayscale image
        return img[np.newaxis, ...], False, "GS"

    elif img.ndim == 3:
        if img.shape[-1] == 3:
            # Single RGB image
            return img[np.newaxis, ...], False, "RGB"
        else:
            # Stack of grayscale images
            return img, False, "GS"

    elif img.ndim == 4 and img.shape[-1] == 3:
        # Stack of RGB images
        return img, False, "RGB"

    raise ValueError(f"Unsupported image shape in single array: {img.shape}")


# ────────────────────────────────────────────────────────────────
# 🔍 Case 2: Handle list of individual images
# ────────────────────────────────────────────────────────────────

def _handle_list_of_images(images: list[np.ndarray]) -> tuple[np.ndarray, bool, str]:
    """
    Handle stacking from a list of individual grayscale or RGB images.
    Applies cropping if needed.
    """
    shapes = [img.shape for img in images]
    first_shape = shapes[0]

    if all(s == first_shape for s in shapes):
        # All shapes match — safe to stack
        if len(first_shape) == 2:
            return np.stack(images), False, "GS"
        elif len(first_shape) == 3 and first_shape[-1] == 3:
            return np.stack(images), False, "RGB"
        else:
            raise ValueError(f"Unsupported image shape: {first_shape}")

    # Shapes differ — fall back to cropping & resizing
    logging.debug(f"Shape mismatch in image stack: {shapes}. Applying center crop + resize.")
    return _center_crop_and_stack(images, first_shape)


# ────────────────────────────────────────────────────────────────
# ✂️ Helper 1: Center-crop and resize all to same shape
# ────────────────────────────────────────────────────────────────

def _center_crop_and_stack(images: list[np.ndarray], reference_shape: tuple) -> tuple[np.ndarray, bool, str]:
    """
    Crops and resizes all images to match the aspect ratio and shape of the first image.
    """
    h0, w0 = reference_shape[:2]
    target_aspect = w0 / h0

    # Center-crop all to same aspect ratio
    cropped = [center_crop_to_aspect(img, target_aspect) for img in images]

    # Resize all to same dimensions
    target_shape = cropped[0].shape[:2]  # (H, W)
    resized = [np.array(Image.fromarray(img).resize(target_shape[::-1])) for img in cropped]

    final_shape = resized[0].shape
    if not all(img.shape == final_shape for img in resized):
        raise ValueError(f"Shape mismatch after crop + resize: {[img.shape for img in resized]}")

    if len(final_shape) == 2:
        return np.stack(resized), True, "GS"
    elif len(final_shape) == 3 and final_shape[-1] == 3:
        return np.stack(resized), True, "RGB"
    else:
        raise ValueError(f"Unsupported final stacked shape: {final_shape}")


# ────────────────────────────────────────────────────────────────
# 🔧 Utility: Center-crop to a target aspect ratio
# ────────────────────────────────────────────────────────────────

def center_crop_to_aspect(img: np.ndarray, target_aspect: float) -> np.ndarray:
    """
    Crops a single image to the given aspect ratio (width / height), centered.
    """
    h, w = img.shape[:2]
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Image too wide → crop width
        new_w = int(h * target_aspect)
        start_w = (w - new_w) // 2
        return img[:, start_w:start_w + new_w]

    else:
        # Image too tall → crop height
        new_h = int(w / target_aspect)
        start_h = (h - new_h) // 2
        return img[start_h:start_h + new_h, :]
