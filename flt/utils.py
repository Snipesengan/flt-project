import cv2
import numpy as np


def split_views(image, left_segment=1/3, right_segment=2/3):
    """Split image into 3 section vertically.

    Args:
        image (np.ndarray): image to be split
        left_semgent (float, optional): first split, normalised.
                                        Defaults to 1/3.
        right_segment (float, optional): second split normalised.
                                        Defaults to 2/3.

        Returns:
            Tuple[np.ndarrray]: Views for left, center and right.
        """
    w = image.shape[1]
    l, r = int(w * left_segment), int(w * right_segment)
    return image[:, :l, ...], image[:, l:r, ...], image[:, r:, ...]


def draw_boxes(image, boxes):
    res = image.copy()
    for box in boxes:
        cv2.rectangle(res, box, (0, 0, 255), 2)
    return res


def mask_boxes(image, boxes, invert=False):
    """Masks rectangular area in the image.

    Args:
        image (np.ndarray): Input image.
        boxes (List): List of (x1, y1, x2, y2).
        invert (bool, optional): Invert the masking. Defaults to True.


        Returns:
            np.ndarray: Masked image.
    """

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        cv2.rectangle(mask, box, (255, 255, 255), -1)

    if not invert:
        mask = np.invert(mask)

    return cv2.bitwise_and(image, image, mask=mask)
