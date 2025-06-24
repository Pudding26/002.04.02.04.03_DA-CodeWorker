import numpy as np

from app.tasks.TA41_ImageSegmentation.Preprocessor import Preprocessor
from app.tasks.TA41_ImageSegmentation.Segmenter import Segmenter
from app.tasks.TA41_ImageSegmentation.FeatureExtractor import FeatureExtractor

class SegmentationWrapper:
    def __init__(self, config: dict, gpu_mode = False):
        self.preprocessor = Preprocessor(config)
        self.segmenter = Segmenter(config)
        self.extractor = FeatureExtractor()
        self.gpu_mode = gpu_mode

    def run_single(self, image: np.ndarray) -> dict:
        filtered, new_gray = self.preprocessor.apply_one(image)
        mask = self.segmenter.apply_one(filtered)
        if self.gpu_mode == False: # Allows to combine the extraction either with or without GPU
            features = self.extractor.apply_one(mask)
            return {
                "filtered_image": filtered, 
                "new_gray": new_gray, 
                "segmentation_mask": mask, 
                "features": features
            }
        
        else:
            return {
                "filtered_image": filtered, 
                "new_gray": new_gray, 
                "segmentation_mask": mask, 
                "features": None
            }
        