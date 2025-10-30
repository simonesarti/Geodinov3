import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.windows import from_bounds

DIM_10M = 1068


def get_reference_random_offset(patch_size: int) -> tuple[int, int]:
    ub = DIM_10M - patch_size + 1
    x_offset = np.random.randint(0, ub)
    y_offset = np.random.randint(0, ub)
    return x_offset, y_offset


def is_bbox_fully_contained(base_bbox: BoundingBox, contained_bbox: BoundingBox) -> bool:
    """
    Checks if a window is completely within the raster dataset.

    Parameters:
        base_bbox (BoundingBox): Bounding box containing the other one
        contained_bbox (BoundingBox): Bounding box contained within the other one

    Returns:
        bool: True if the second bounding box is fully contained within the first one.
    """

    return (
        contained_bbox.left >= base_bbox.left and
        contained_bbox.right <= base_bbox.right and
        contained_bbox.bottom >= base_bbox.bottom and
        contained_bbox.top <= base_bbox.top
    )


def raster_extract_filled_bands_from_bbox(raster: rasterio.DatasetReader, ref_bbox: BoundingBox, fill_value: int,) -> np.ndarray:
    # Create a window that matches the geographical bounds of the reference raster window
    window = from_bounds(ref_bbox.left, ref_bbox.bottom, ref_bbox.right, ref_bbox.top, raster.transform)
    # extract the window and fill nodata values with 0
    bands_array = raster.read(window=window, masked=True).filled(fill_value)
    return bands_array


def encode_latitude(lat):
    """ Latitude goes from -90 to 90 """

    lat_adj = lat + 90.0
    lat_max = 180

    encoded_sin = (np.sin(2 * np.pi * (lat_adj / lat_max)) + 1) / 2.0
    encoded_cos = (np.cos(2 * np.pi * (lat_adj / lat_max)) + 1) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


def encode_longitude(lng):
    """ Longitude goes from -180 to 180 """

    lng_adj = lng + 180.0
    lng_max = 360

    encoded_sin = (np.sin(2 * np.pi * (lng_adj / lng_max)) + 1) / 2.0
    encoded_cos = (np.cos(2 * np.pi * (lng_adj / lng_max)) + 1) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)
