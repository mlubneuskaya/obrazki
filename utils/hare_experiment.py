import cv2
import numpy as np
from scipy.stats import norm


def bimodal_cdf(m1, m2, mu1, mu2, w1=0.5, n=10**5):  # TODO add random choice instead of weights
    w2 = 1 - w1
    hist, bin_edges = np.histogram(
            np.clip(
                np.concatenate(
                    [norm.rvs(loc=m1, scale=mu1, size=int(n*w1)), norm.rvs(loc=m2, scale=mu2, size=int(n*w2))]
                ),
                0,
                255,
            ),
            bins=256,
            density=True
        )
    bin_widths = np.diff(bin_edges)
    cdf = np.cumsum(hist * bin_widths)
    return cdf / cdf[-1], hist


def generate_matching_lookup_table(source_cdf, reference_cdf):
    inv_reference_cdf = np.interp(source_cdf, reference_cdf, np.arange(0, 256))
    return np.rint(inv_reference_cdf)


def run_hare_experiment(image, reference_distribution_parameters):
    source_hist, bins = np.histogram(image, bins=256, density=True)
    source_cdf = np.cumsum(source_hist * np.diff(bins))
    source_cdf /= source_cdf[-1]
    reference_cdf, reference_hist = bimodal_cdf(**reference_distribution_parameters)
    lookup_table = generate_matching_lookup_table(source_cdf, reference_cdf)
    return {
        'source_hist': source_hist,
        'reference_hist': reference_hist,
        'converted': cv2.LUT(image, lookup_table).astype(np.uint8),
    }