import numpy as np

"""
Terrain (Terrain22 https://gisstar.gsi.go.jp/terrain2021/)
0:  Water / Nodata
1:  Mountain Summit
2:  Cliff Slope
3:  Lower/Hilly Mountain
4:  Steep Hills/Dissected Cliff Slope
5:  Large Highland Slope (Steep)
6:  Large Highland Slope (Moderate)
7:  Mountain Valley Slope
8:  Moderate Hills
9:  Terrace/Fan/Plateau (High, Dissected, Sinks < 50%)
10: Terrace/Fan/Plateau (High, Dissected, Sinks >= 50%)
11: Terrace/Fan/Plateau (High, Surface, Sinks < 50%)
12: Terrace/Fan/Plateau (High, Surface, Sinks >= 50%)
13: Valley Slope (Sinks < 50%)
14: Valley Slope (Sinks >= 50%)
15: Terrace/Fan/Plateau (Low, Dissected, Sinks < 50%)
16: Terrace/Fan/Plateau (Low, Dissected, Sinks >= 50%)
17: Terrace/Fan/Plateau (Low, Surface, Sinks < 50%)
18: Terrace/Fan/Plateau (Low, Surface, Sinks >= 50%)
19: High Plain (Sinks < 50%)
20: High Plain (Sinks >= 50%)
21: Low Plain (Sinks < 50%)
22: Low Plain (Sinks >= 50)
"""

NUM_TERRAIN_CLASSES = 22    # exclude nodata


def get_terrain_label(terrain_array: np.ndarray, valid: bool = True) -> tuple[np.ndarray, np.ndarray]:

    if valid:     # bbox_fully contained & not corrupted
        terrain_array = terrain_array.flatten()
        terrain_non_zero = np.count_nonzero(terrain_array)

        if terrain_non_zero > 0:
            terrain_values = np.bincount(terrain_array, minlength=NUM_TERRAIN_CLASSES+1)[1:] / terrain_non_zero
            percentage_of_non_zero_values = terrain_non_zero / terrain_array.size
            terrain = np.array(terrain_values, dtype=np.float32)    # SHAPE: [22]
            terrain_weight = np.array([percentage_of_non_zero_values], dtype=np.float32)  # SHAPE: [1]
            return terrain, terrain_weight

    terrain = [0.0] * NUM_TERRAIN_CLASSES
    terrain = np.array(terrain, dtype=np.float32)  # SHAPE: [22]
    terrain_weight = np.array([0.0], dtype=np.float32)    # SHAPE: [1]
    return terrain, terrain_weight
