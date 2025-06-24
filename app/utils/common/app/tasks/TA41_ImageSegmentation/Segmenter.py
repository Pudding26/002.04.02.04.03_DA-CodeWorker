import cv2
import numpy as np



class Segmenter:
    def __init__(self, config: dict):
        self.cfg = config["segmentation"]

    def apply_one(self, image: np.ndarray) -> np.ndarray:
        method = self.cfg["threshold"]["method"]
        if method == "Otsu":
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "Adaptive":
            p = self.cfg["threshold"]["adaptive"]
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, p["block_size"], p["C"])
        elif method == "Manual":
            val = self.cfg["threshold"]["manual"]["value"]
            _, binary = cv2.threshold(image, val, 255, cv2.THRESH_BINARY)
        else:
            binary = image.copy()

        edge_method = self.cfg["edge"]["method"]
        if edge_method == "Canny":
            p = self.cfg["edge"]["canny"]
            binary = cv2.Canny(image, p["low"], p["high"])
        elif edge_method == "Sobel":
            p = self.cfg["edge"]["sobel"]
            sobel = cv2.Sobel(image, cv2.CV_64F, p["dx"], p["dy"], ksize=p["ksize"])
            binary = np.uint8(np.clip(np.abs(sobel), 0, 255))

        return binary

