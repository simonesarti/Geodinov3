import numpy as np

"""
Clouds:
0: TODO
1: TODO
2: TODO
3: TODO
"""


NUM_CLOUDS_CLASSES = 4


def get_clouds_label(clouds_array: np.ndarray, valid: bool) -> tuple[np.ndarray, np.ndarray]:

    if valid:
        num_elem = clouds_array.size
        clouds = np.array([
            np.count_nonzero(clouds_array == 0) / num_elem,
            np.count_nonzero(clouds_array == 1) / num_elem,
            np.count_nonzero(clouds_array == 2) / num_elem,
            np.count_nonzero(clouds_array == 3) / num_elem,
        ], dtype=np.float32)    # SHAPE: [4]
        clouds_weight = np.array([1.0], dtype=np.float32)   # SHAPE: [1]
    else:
        clouds = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)     # SHAPE: [4]
        clouds_weight = np.array([0.0], dtype=np.float32)   # SHAPE: [1]

    return clouds, clouds_weight
