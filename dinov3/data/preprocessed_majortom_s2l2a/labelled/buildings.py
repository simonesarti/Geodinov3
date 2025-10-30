import numpy as np

"""
Structures
0 - 100 average m2 per pixel, uint8 band
"""


def get_buildings_label(buildings_array: np.ndarray, valid: bool = True) -> tuple[np.ndarray, np.ndarray]:

    if valid:   # bbox_fully contained & not corrupted
        buildings_value = (buildings_array.sum() / buildings_array.size) / 100.0
        buildings = np.array([buildings_value], dtype=np.float32)   # SHAPE: [1]
        buildings_weight = np.array([1.0], dtype=np.float32)   # SHAPE: [1]
        return buildings, buildings_weight

    buildings = np.array([0.0], dtype=np.float32)   # SHAPE: [1]
    buildings_weight = np.array([0.0], dtype=np.float32)    # SHAPE: [1]
    return buildings, buildings_weight
