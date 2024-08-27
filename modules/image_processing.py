import numpy as np
import cv2

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image for further processing.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Preprocessed image.
    """
    # Resize the image to a fixed size (e.g., 800x600)
    resized_image = cv2.resize(image, (800, 600))
    
    # Ensure the image has 3 channels (RGB)
    if len(resized_image.shape) == 2:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    elif resized_image.shape[2] == 4:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGBA2RGB)
    
    return resized_image

def align_images(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """
    Align the new image with the previous image using feature matching.
    
    Args:
        previous_image (np.ndarray): Previous timepoint image.
        new_image (np.ndarray): New timepoint image.
    
    Returns:
        np.ndarray: Aligned new image.
    """
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to warp image
    height, width = image1.shape
    aligned_image = cv2.warpPerspective(image2, h, (width, height))

    return aligned_image