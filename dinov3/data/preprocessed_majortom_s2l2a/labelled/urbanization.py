import numpy as np

"""
Degurba
0:  Water / Nodata
11: Very Low Density Rural
12: Low Density Rural
13: Rural Cluster
21: Suburban Or Peri-Urban
22: Semi-Dense Urban Cluster
23: Dense Urban Cluster
30: Urban Centre
"""

NUM_DEGURBA_CLASSES = 7     # exclude nodata
URBANIZATION_CLASSES_MAPPING = {
    0: 0,       # Water / Nodata
    11: 1,      # Very Low Density Rural
    12: 2,      # Low Density Rural
    13: 3,      # Rural Cluster
    21: 4,      # Suburban Or Peri-Urban
    22: 5,      # Semi-Dense Urban Cluster
    23: 6,      # Dense Urban Cluster
    30: 7,      # Urban Centre
}


def get_urbanization_label(urbanization_array: np.ndarray, valid: bool = True) -> tuple[np.ndarray, np.ndarray]:

    if valid:     # bbox_fully contained & not corrupted
        urbanization_array = urbanization_array.flatten()
        urbanization_non_zero = np.count_nonzero(urbanization_array)

        if urbanization_non_zero > 0:
            map_urbanization = np.vectorize(lambda x: URBANIZATION_CLASSES_MAPPING.get(x))
            mapped = map_urbanization(urbanization_array)
            urbanization_values = np.bincount(mapped, minlength=NUM_DEGURBA_CLASSES+1)[1:] / urbanization_non_zero
            percentage_of_non_zero_values = urbanization_non_zero / urbanization_array.size
            urbanization = np.array(urbanization_values, dtype=np.float32)    # SHAPE: [7]
            urbanization_weight = np.array([percentage_of_non_zero_values], dtype=np.float32)  # SHAPE: [1]
            return urbanization, urbanization_weight

    urbanization = [0.0] * NUM_DEGURBA_CLASSES
    urbanization = np.array(urbanization, dtype=np.float32)     # SHAPE: [7]
    urbanization_weight = np.array([0.0], dtype=np.float32)    # SHAPE: [1]
    return urbanization, urbanization_weight
