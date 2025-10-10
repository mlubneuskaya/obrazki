import warnings

import numpy as np


def find_otsu_threshold_slow_but_intuitive(img):
    return np.nanargmin(
        [
            (img <= t).mean() * img.var(where=img <= t)
            + (img > t).mean() * img.var(where=img > t)
            for t in range(img.max() + 1)
        ]
    )

def find_otsu_threshold(img):
    histogram = np.bincount(img.flatten(), minlength=img.max() + 1)
    p = histogram / histogram.sum()
    q = np.cumsum(p)
    values = np.arange(img.max() + 1)
    mu = np.cumsum(p * values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma_b2 = (mu[-1] * q - mu) ** 2 / q * (1.0 - q)
        return int(np.nanargmax(sigma_b2))

def binarize(img, light=0, dark=255):
    threshold = find_otsu_threshold(img)
    img[img < threshold] = light
    img[img >= threshold] = dark
    return img

def local_otsu(img, window_size, light=0, dark=255):
    img_width, img_height = img.shape
    horizontal_padding, vertical_padding = img_width % window_size, img_height % window_size
    img = np.pad(img, [(0, horizontal_padding), (0, vertical_padding)], mode='reflect')
    for i in range(0, img_width + horizontal_padding, window_size):
        for j in range(0, img_height + vertical_padding, window_size):
            img[i : i + window_size, j: j + window_size] = binarize(img[i : i + window_size, j: j + window_size], light=light, dark=dark)
    return img[:img_width, :img_height]
