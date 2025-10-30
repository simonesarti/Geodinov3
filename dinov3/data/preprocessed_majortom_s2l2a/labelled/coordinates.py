import numpy as np

from preprocessed_majortom_s2l2a.geo_utils import encode_latitude, encode_longitude


def get_coords_label(lat: float, lng: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.concatenate([encode_latitude(lat), encode_longitude(lng)], dtype=np.float32)    # SHAPE: [4]
    coords_weight = np.array([1.0], dtype=np.float32)   # SHAPE: [1]
    return coords, coords_weight
