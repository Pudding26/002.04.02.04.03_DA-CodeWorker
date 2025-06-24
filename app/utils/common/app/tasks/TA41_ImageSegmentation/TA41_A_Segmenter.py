from app.tasks.TA41_ImageSegmentation.SegmentationWrapper import SegmentationWrapper

class TA41_A_Segmenter:
    def __init__(self, config: dict, image_stack: list, image_stack_id: str, gpu_mode = False):
        self.config = config
        self.image_stack = image_stack
        self.image_stack_id = image_stack_id
        self.gpu_mode = gpu_mode
        self.wrapper = SegmentationWrapper(config, gpu_mode)

    def run_stack(self) -> dict:
        filtered_stack = []
        mask_stack = []
        features_list = []
        new_gray = []

        for idx, image in enumerate(self.image_stack):
            result = self.wrapper.run_single(image)
            filtered_stack.append(result["filtered_image"])
            mask_stack.append(result["segmentation_mask"])
            features_list.append(result["features"])
            new_gray.append(result["new_gray"])

        return {
            "filtered_image_stack": filtered_stack,
            "mask_stack": mask_stack,
            "features": features_list,
            "new_gray_stack": new_gray,
        }







        
