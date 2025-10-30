import numpy as np

"""
Water
0: Land
1: Water
"""


NUM_WATER_CLASSES = 2


def get_water_label(water_array: np.ndarray, valid: bool = True) -> tuple[np.ndarray, np.ndarray]:

    if valid:     # bbox_fully contained & not corrupted
        water_value = water_array.sum() / water_array.size
        water = np.array([water_value], dtype=np.float32)    # SHAPE: [1]
        water_weight = np.array([1.0], dtype=np.float32)    # SHAPE: [1]
        return water, water_weight

    water = np.array([0.0], dtype=np.float32)   # SHAPE: [1]
    water_weight = np.array([0.0], dtype=np.float32)    # SHAPE: [1]
    return water, water_weight
