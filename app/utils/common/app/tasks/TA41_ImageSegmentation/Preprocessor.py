import numpy as np
import cv2

class Preprocessor:
    def __init__(self, config: dict):
        self.cfg = config["preprocessing"]

    def apply_one(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2 or image.shape[2] == 1:
            gray_image = image if len(image.shape) == 2 else image[:, :, 0]
            new_gray = None
        else:
            gray_image = self._to_grayscale(image)
            new_gray = gray_image
        image = gray_image
        image = self._apply_contrast(image)
        image = self._apply_noise_filter(image)
        image = self._apply_normalization(image)
        return image, new_gray

    def _to_grayscale(self, image):
        mode = self.cfg.get("gray_channel", "Luminance")
        if mode == "Red":
            return image[:, :, 0]
        elif mode == "Green":
            return image[:, :, 1]
        elif mode == "Blue":
            return image[:, :, 2]
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _apply_contrast(self, gray):
        method = self.cfg["contrast"]["method"]
        if method == "CLAHE":
            p = self.cfg["contrast"]["clahe"]
            clahe = cv2.createCLAHE(p["clip_limit"], (p["tile_size"], p["tile_size"]))
            return clahe.apply(gray)
        elif method == "Gamma":
            gamma = self.cfg["contrast"]["gamma"]
            return np.power(gray / 255.0, gamma) * 255
        elif method == "Manual":
            p = self.cfg["contrast"]["manual"]
            return gray.astype(np.float32) * p["contrast"] + p["brightness"]
        elif method == "Histogram":
            return cv2.equalizeHist(gray)
        return gray

    def _apply_noise_filter(self, gray):
        method = self.cfg["noise"]["method"]
        if method == "Gaussian":
            k = self.cfg["noise"]["gaussian"]["ksize"]
            return cv2.GaussianBlur(gray, (k, k), 0)
        elif method == "Median":
            k = self.cfg["noise"]["median"]["ksize"]
            return cv2.medianBlur(gray, k)
        elif method == "Bilateral":
            p = self.cfg["noise"]["bilateral"]
            return cv2.bilateralFilter(gray, p["d"], p["sigma_color"], p["sigma_space"])
        return gray

    def _apply_normalization(self, gray):
        method = self.cfg["normalization"]["method"]

        if method == "MinMax":
            img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        elif method == "ZScore":
            mean, std = np.mean(gray), np.std(gray)
            img = ((gray - mean) / std) * 64 + 128
        elif method == "HistogramStretch":
            img = 255 * (gray - gray.min()) / (gray.max() - gray.min())
        else:
            img = gray

        return cv2.convertScaleAbs(img)


