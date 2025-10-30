import numpy as np

"""
Climate
0:  Water / Nodata
1:  Af   Tropical, Rainforest
2:  Am   Tropical, Monsoon
3:  Aw   Tropical, Savannah
4:  Bwh  Arid, Desert, Hot
5:  Bwk  Arid, Desert, Cold
6:  Bsh  Arid, Steppe, Hot
7:  Bsk  Arid, Steppe, Cold
8:  Csa  Temperate, Dry Summer, Hot Summer
9:  Csb  Temperate, Dry Summer, Warm Summer
10: Csc  Temperate, Dry Summer, Cold Summer
11: Cwa  Temperate, Dry Winter, Hot Summer
12: Cwb  Temperate, Dry Winter, Warm Summer
13: Cwc  Temperate, Dry Winter, Cold Summer
14: Cfa  Temperate, No Dry Season, Hot Summer
15: Cfb  Temperate, No Dry Season, Warm Summer
16: Cfc  Temperate, No Dry Season, Cold Summer
17: Dsa  Cold, Dry Summer, Hot Summer
18: Dsb  Cold, Dry Summer, Warm Summer
19: Dsc  Cold, Dry Summer, Cold Summer
20: Dsd  Cold, Dry Summer, Very Cold Winter
21: Dwa  Cold, Dry Winter, Hot Summer
22: Dwb  Cold, Dry Winter, Warm Summer
23: Dwc  Cold, Dry Winter, Cold Summer
24: Dwd  Cold, Dry Winter, Very Cold Winter
25: Dfa  Cold, No Dry Season, Hot Summer
26: Dfb  Cold, No Dry Season, Warm Summer
27: Dfc  Cold, No Dry Season, Cold Summer
28: Dfd  Cold, No Dry Season, Very Cold Winter
29: Et   Polar, Tundra
30: Ef   Polar, Frost
"""


NUM_CLIMATE_CLASSES = 30    # exclude nodata


def get_climate_label(climate_array: np.ndarray, valid: bool = True) -> tuple[np.ndarray, np.ndarray]:

    if valid:  # bbox_fully contained & not corrupted
        climate_array = climate_array.flatten()
        climate_non_zero = np.count_nonzero(climate_array)

        if climate_non_zero > 0:
            climate_values = np.bincount(climate_array, minlength=NUM_CLIMATE_CLASSES+1)[1:] / climate_non_zero
            percentage_of_non_zero_values = climate_non_zero / climate_array.size
            climate = np.array(climate_values, dtype=np.float32)    # SHAPE: [30]
            climate_weight = np.array([percentage_of_non_zero_values], dtype=np.float32)  # SHAPE: [1]
            return climate, climate_weight

    climate = [0.0] * NUM_CLIMATE_CLASSES
    climate = np.array(climate, dtype=np.float32)  # SHAPE: [30]
    climate_weight = np.array([0.0], dtype=np.float32)    # SHAPE: [1]
    return climate, climate_weight
