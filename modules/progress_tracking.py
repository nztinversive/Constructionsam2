import numpy as np

def calculate_progress(previous_segmentation: np.ndarray, current_segmentation: np.ndarray) -> float:
    if previous_segmentation.shape != current_segmentation.shape:
        raise ValueError("Segmentation masks must have the same dimensions")
    
    previous_area = np.sum(previous_segmentation > 0)
    current_area = np.sum(current_segmentation > 0)
    
    if previous_area == 0:
        return 100.0 if current_area > 0 else 0.0
    
    progress = (current_area - previous_area) / previous_area * 100
    return max(0.0, min(100.0, progress))