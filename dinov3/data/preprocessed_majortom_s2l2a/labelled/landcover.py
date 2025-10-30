import numpy as np

"""
Land Cover
0:  Nodata
10: Tree Cover
20: Shrubland
30: Grassland
40: Cropland
50: Built-Up
60: Bare / Sparse Vegetation
70: Snow / Ice
80: Permanent Water Bodies
90: Herbaceous Wetland
95: Mangroves
100: Moss / Lichen
"""

NUM_LANDCOVER_CLASSES = 11  # exclude nodata
LANDCOVER_CLASSES_MAPPING = {
    0: 0,       # nodata
    10: 1,      # trees
    20: 2,      # shrubs
    30: 3,      # grass
    40: 4,      # crops
    50: 5,      # built
    60: 6,      # bare
    70: 7,      # snow
    80: 8,      # water
    90: 9,      # wetland
    95: 10,     # mangrove
    100: 11,    # moss/lichen
}


def get_landcover_label(landcover_array: np.ndarray, valid: bool = True) -> tuple[np.ndarray, np.ndarray]:

    if valid:     # bbox_fully contained & not corrupted
        landcover_array = landcover_array.flatten()
        landcover_non_zero = np.count_nonzero(landcover_array)

        if landcover_non_zero > 0:
            map_landcover = np.vectorize(lambda x: LANDCOVER_CLASSES_MAPPING.get(x))
            mapped = map_landcover(landcover_array)
            landcover_values = np.bincount(mapped, minlength=NUM_LANDCOVER_CLASSES+1)[1:] / landcover_non_zero
            percentage_of_non_zero_values = landcover_non_zero / landcover_array.size
            landcover = np.array(landcover_values, dtype=np.float32)    # SHAPE: [11]
            landcover_weight = np.array([percentage_of_non_zero_values], dtype=np.float32)  # SHAPE: [1]
            return landcover, landcover_weight

    landcover = [0.0] * NUM_LANDCOVER_CLASSES   # all zeros = nodata class
    landcover = np.array(landcover, dtype=np.float32)     # SHAPE: [11]
    landcover_weight = np.array([0.0], dtype=np.float32)    # SHAPE: [1]
    return landcover, landcover_weight
