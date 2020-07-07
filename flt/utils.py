import cv2
import numpy as np


def split_views(image, l=1/3, r=2/3):

    w = image.shape[1]
    l, r = int(w * l), int(w * r)
    return image[:,:l,... ],image[:,l:r,... ],image[:,r:,...]


def draw_boxes(image, boxes):

    res = image.copy()
    for box in boxes:
        cv2.rectangle(res, box, (0,0,255), 2) 

    return res


def mask_boxes(image, boxes, invert=False):
    """Masks rectangular area in the image

    Args:
        image (np.ndarray): Input image
        boxes (List): List of (x1, y1, x2, y2)
        invert (bool, optional): Invert the masking. Defaults to True.
    """

    mask = np.zeros(image.shape[:2],dtype=np.uint8)
    for box in boxes:
        cv2.rectangle(mask, box, (255, 255, 255), -1)

    if not invert:
        mask = np.invert(mask)
        
    return cv2.bitwise_and(image, image, mask=mask)
        
