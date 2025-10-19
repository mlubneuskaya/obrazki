import cv2


def apply_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 0)
