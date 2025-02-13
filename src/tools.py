# inspired from https://www.kaggle.com/code/burcuamirgan/deeplabv3-deepglobe-lc-classification

import numpy as np
from . import dataset

# Impervious surfaces
# Agricultural
# Forest
# Soil
# Water
# Wetlands
# Snow and ice

class_rgb_dfc25 = {
    "background": [255, 255, 255],
    "intact": [70, 181, 121],
    "damaged": [228, 189, 139],
    "destroyed": [182, 70, 69],
}

class_grey_dfc25 = {
    "background": 0,
    "intact": 1,
    "damaged": 2,
    "destroyed": 3,
}


def convert8bit(A, mu, sig, mask):
    """_summary_

    Args:
        A (_type_): _description_
        mu (_type_): _description_
        sig (_type_): _description_
        mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    A = (A - mu) / (6 * sig) * 255 + 128
    A[A > 255] = 255
    A[A < 0] = 0
    A = A.astype(np.uint8)
    sz = A.shape
    A = A.flatten()
    A[mask] = 0
    return A.reshape(sz).astype(np.uint8)


def make_mask(a, grey_codes=class_grey_dfc25, rgb_codes=class_rgb_dfc25):
    """
    a: semantic map (H x W x n-classes)
    """
    out = np.zeros(shape=a.shape[:2], dtype="uint8")
    for k, v in rgb_codes.items():
        mask = np.all(np.equal(a, v), axis=-1)
        out[mask] = grey_codes[k]
    return out


def make_rgb(a, grey_codes=class_grey_dfc25, rgb_codes=class_rgb_dfc25):
    """
    a: labels (H x W)
    rgd_codes: dict of class-rgd code
    grey_codes: dict of label code
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in grey_codes.items():
        out[a == v, 0] = rgb_codes[k][0]
        out[a == v, 1] = rgb_codes[k][1]
        out[a == v, 2] = rgb_codes[k][2]
    return out


def mean_var(data_gen):
    """mean and variance computation for a generator of numpy arrays

    Mean and variance are computed in a divide and conquer fashion individally for each array.
    The results are then properly aggregated.

    Parameters
    ----------

    data_gen: generator
        data_gen is supposed to generate numpy arrays

    """

    try:
        head = next(iter(data_gen))
    except StopIteration:
        raise ValueError("You supplied an empty generator!")
    return _mean_var(*_comp(head), data_gen)


def _comp(els):
    """individual computation for each array"""
    n_el = els.size
    sum_el = els.sum()  # basically mean
    sum2_el = ((els - sum_el / n_el) ** 2).sum()  # basically variance
    return (sum_el, sum2_el, n_el)


def _mean_var(sum_a, sum2_a, n_a, data_list):
    """divide and conquer mean and variance computation"""

    def _combine_samples(sum_a, sum2_a, n_a, sum_b, sum2_b, n_b):
        """implements formulae 1.5 in [3]"""
        sum_c = sum_a + sum_b
        sum1_c = sum2_a + sum2_b
        sum1_c += ((sum_a * (n_b / n_a) - sum_b) ** 2) * (n_a / n_b) / (n_a + n_b)

        return (sum_c, sum1_c, n_a + n_b)

    for el_b in data_list:
        # iteration and aggreation
        sum_a, sum2_a, n_a = _combine_samples(sum_a, sum2_a, n_a, *_comp(el_b))
    return (sum_a / n_a, sum2_a / n_a)
