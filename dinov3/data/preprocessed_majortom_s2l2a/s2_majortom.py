from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.coords import BoundingBox
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from time import time
import os
import logging
import sys

os.environ["GDAL_CACHEMAX"] = str(4 * 1024)    # 4GB
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

from torch.utils.data import Dataset

from .satellite.sentinel2 import (
    get_s2_num_bands,
    get_valid_s2_band_names,
    get_sentienl2_bands_mask,
    compose_sentinel2_with_clouds_bands_path,
    normalize_sentinel2,
    S2_BANDS_NAMES,
)

from .geo_utils import is_bbox_fully_contained, raster_extract_filled_bands_from_bbox, get_reference_random_offset
from .labelled.buildings import get_buildings_label
from .labelled.clouds import get_clouds_label
from .labelled.coordinates import get_coords_label
from .labelled.climate import get_climate_label
from .labelled.landcover import get_landcover_label
from .labelled.terrain import get_terrain_label
from .labelled.urbanization import get_urbanization_label
from .labelled.water import get_water_label
from .labelled.path import compose_labels_path


# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                                # Log to console
    ],)
logger = logging.getLogger("majortoms2")
# ================================================================


class MajorTOM(Dataset):
    def __init__(
        self,
        dataset_path: str = "/archive/group/major-tom/MajorTOM/",
        patch_size: int = 128,
        data_transform=None,
        s2_bands: tuple[str, ...] = ("B02", "B03", "B04"),
        use_buildings: bool = False,
        use_climate: bool = False,
        use_clouds: bool = False,
        use_coords: bool = False,
        use_landcover: bool = False,
        use_terrain: bool = False,
        use_urbanization: bool = False,
        use_water: bool = False,
        get_placeholder_unused_band: bool = False,
    ):

        super(MajorTOM, self).__init__()

        logger.info("Dataset setup start")
        self.dataset_path = Path(dataset_path)
        self.s2_level = "L2A"
        self.s2_pickle_path = self.dataset_path / "S2L2A_paths.pkl"

        if self.s2_pickle_path.is_file():
            logger.info("Unpickling S2 dataset ...")
            self.s2_dataset = pd.read_pickle(self.s2_pickle_path)
            logger.info(f"Dataset unpickled")
        else:
            raise FileNotFoundError(f"S2 pickle file {str(self.s2_pickle_path)} not found")

        self.patch_size = patch_size
        self.data_transform = data_transform

        self.get_placeholder_unused_band = get_placeholder_unused_band

        # -------------- SENTINEL 2 -------------------
        logger.info("SETTING UP SENTINEL-2 L2A")

        self.s2_bands = ()  # set default value
        self.s2_bands_mask = (False, ) * get_s2_num_bands()     # set default value

        s2_bands = tuple([s2_band.upper() for s2_band in s2_bands])
        valid_band_names = get_valid_s2_band_names(s2_bands)
        if not valid_band_names:    # tuple is empty
            raise AttributeError(f"Error: Provided completely invalid set of S2 band names {s2_bands}")
        else:
            self.s2_bands = valid_band_names
            self.s2_bands_mask = get_sentienl2_bands_mask(valid_band_names)
            logger.info(f"using S2 bands {self.s2_bands}")
            logger.info(f"S2 bands: {S2_BANDS_NAMES}")
            logger.info(f"using S2 bands mask {self.s2_bands_mask}")

        # -------------- STATIC LABELS -------------------
        logger.info("SETTING UP LABELS")

        self.use_buildings = use_buildings
        self.use_climate = use_climate
        self.use_clouds = use_clouds
        self.use_coords = use_coords
        self.use_landcover = use_landcover
        self.use_terrain = use_terrain
        self.use_urbanization = use_urbanization
        self.use_water = use_water

        logger.info(f"Using buildings: {use_buildings}")
        logger.info(f"Using climate: {use_climate}")
        logger.info(f"Using clouds: {use_clouds}")
        logger.info(f"Using coords: {use_coords}")
        logger.info(f"Using landcover: {use_landcover}")
        logger.info(f"Using terrain: {use_terrain}")
        logger.info(f"Using urbanization: {use_urbanization}")
        logger.info(f"Using water: {use_water}")

        self.use_static_labels = use_buildings or use_climate or use_landcover or use_terrain or use_urbanization or use_water
        self.use_labels = self.use_static_labels or use_clouds or use_coords

        # Store paths instead of opening files immediately
        self.static_dataset_path = compose_labels_path(self.dataset_path.parent)

        # initialized on first __getitem__
        self.static_dataset = None

        # Add a worker ID tracker
        self._worker_id = None

        logger.info("DATASET INITIALIZED")

    def _init_static_datasets(self):
        """
        Initialize static dataset. This method is called
        to ensure each worker process has its own file handles.
        """

        # Get current worker info
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info is not None else -1  # Use -1 for main process

        # Only initialize if not already done for this worker
        if self.static_dataset is None and self.use_labels:
            self.static_dataset = rasterio.open(self.static_dataset_path)
            self._worker_id = current_worker_id
            logger.info(f"Opened static dataset in worker {current_worker_id}")

    def _close_static_datasets(self):
        """
        Close all static datasets safely.
        """

        if self.static_dataset is not None and not self.static_dataset.closed:
            try:
                self.static_dataset.close()
            except Exception:
                pass  # Ignore errors during cleanup

        self.static_dataset = None
        logger.info(f"Closed static dataset in worker {self._worker_id}")

    def __del__(self):
        """
        Cleanup method to properly close static datasets when the object is destroyed.
        """
        self._close_static_datasets()

    def __len__(self):
        return len(self.s2_dataset)

    def __getitem__(self, idx):

        t0 = time()
        self._init_static_datasets()

        # get information about the path to the S2 image
        df_row = self.s2_dataset.iloc[idx]

        # define a window of size (patch_size, patch_size) using random offset
        # the dimension of the image is know a priori (1068x1068 at 10m resolution)
        x_off, y_off = get_reference_random_offset(self.patch_size)
        window = Window(col_off=x_off, row_off=y_off, width=self.patch_size, height=self.patch_size)

        # open the raster and generate bounding boxes based on the window
        s2_raster_path = compose_sentinel2_with_clouds_bands_path(self.dataset_path, self.s2_level, df_row)
        logger.debug(f"reference raster {s2_raster_path}")
        s2_raster = rasterio.open(s2_raster_path)
        s2_patch_bbox_original_crs = BoundingBox(*window_bounds(window, s2_raster.transform))
        s2_patch_bbox_wgs84 = BoundingBox(*transform_bounds(
            src_crs=s2_raster.crs.to_string(),      # in Majortom, every patch has a different CRS,
            dst_crs="EPSG:4326",                    # data static CRS is "EPSG:4326"
            left=s2_patch_bbox_original_crs.left,
            bottom=s2_patch_bbox_original_crs.bottom,
            right=s2_patch_bbox_original_crs.right,
            top=s2_patch_bbox_original_crs.top,
        ))
        logger.debug(f"reference bound: {s2_patch_bbox_wgs84}")
        logger.debug(f"BOTTOM = {s2_patch_bbox_wgs84.bottom}")
        logger.debug(f"LEFT = {s2_patch_bbox_wgs84.left}")
        logger.debug(f"TOP = {s2_patch_bbox_wgs84.top}")
        logger.debug(f"RIGHT = {s2_patch_bbox_wgs84.right}")

        t1=time()

        # ============ DATA =============
        
        fill_value = 0

        bands_array = raster_extract_filled_bands_from_bbox(
            raster=s2_raster,
            ref_bbox=s2_patch_bbox_original_crs,
            fill_value=fill_value,
        )

        s2_raster.close()

        t4=time()
        assert bands_array.shape[1] == self.patch_size, f"Got bands_array of shape {bands_array.shape}, but target patch size is {self.patch_size}"
        assert bands_array.shape[2] == self.patch_size, f"Got bands_array of shape {bands_array.shape}, but target patch size is {self.patch_size}"

        clouds_array = bands_array[-1]
        bands_array = bands_array[:-1]  # only sentinel channels

        bands_mask_array = np.array(self.s2_bands_mask, dtype=bool)
        if self.get_placeholder_unused_band:
            bands_array[~bands_mask_array] = fill_value
            assert bands_array.shape[0] == get_s2_num_bands()
        else:
            bands_array = bands_array[bands_mask_array]
            assert bands_array.shape[0] == sum(self.s2_bands_mask)

        bands_array = normalize_sentinel2(bands_array)

        t5=time()
        x = torch.from_numpy(bands_array).to(dtype=torch.float32)
        if self.data_transform is not None:
            x = self.data_transform(x)

        t2=time()
        # ============ LABELS =============

        if not self.use_labels:
            return x, None
        
        labels = {}

        if self.use_clouds:
            labels["clouds"], labels["clouds_weight"] = get_clouds_label(clouds_array, valid=True)

        if self.use_coords:
            patch_center_lat = (s2_patch_bbox_wgs84.bottom + s2_patch_bbox_wgs84.top) / 2
            patch_center_lng = (s2_patch_bbox_wgs84.left + s2_patch_bbox_wgs84.right) / 2
            labels["coords"], labels["coords_weight"] = get_coords_label(patch_center_lat, patch_center_lng)
            logger.debug(f"lat, lon: {patch_center_lat}, {patch_center_lng}")

        if self.use_static_labels:

            bbox_in_static_bounds = is_bbox_fully_contained(
                base_bbox=self.static_dataset.bounds,
                contained_bbox=s2_patch_bbox_wgs84,
            )

            combined_array = raster_extract_filled_bands_from_bbox(
                raster=self.static_dataset,
                ref_bbox=s2_patch_bbox_wgs84,
                fill_value=0,
            )

            # ORDER OF BANDS AFTER PREPROCESSING:
            # 0: BUILD
            # 1: CLIMATE
            # 2: DEGURBA
            # 3: LANDCOVER
            # 4: TERRAIN
            # 5: WATER

            if self.use_buildings:
                labels["buildings"], labels["buildings_weight"] = get_buildings_label(
                    buildings_array=combined_array[0],
                    valid=bbox_in_static_bounds,
                )

            if self.use_climate:
                labels["climate"], labels["climate_weight"] = get_climate_label(
                    climate_array=combined_array[1],
                    valid=bbox_in_static_bounds,
                )

            if self.use_urbanization:
                labels["urbanization"], labels["urbanization_weight"] = get_urbanization_label(
                    urbanization_array=combined_array[2],
                    valid=bbox_in_static_bounds,
                )

            if self.use_landcover:
                labels["landcover"], labels["landcover_weight"] = get_landcover_label(
                    landcover_array=combined_array[3],
                    valid=bbox_in_static_bounds,
                )

            if self.use_terrain:
                labels["terrain"], labels["terrain_weight"] = get_terrain_label(
                    terrain_array=combined_array[4],
                    valid=bbox_in_static_bounds,
                )

            if self.use_water:
                labels["water"], labels["water_weight"] = get_water_label(
                    water_array=combined_array[5],
                    valid=bbox_in_static_bounds,
                )

        labels = {
            key: torch.from_numpy(value).to(dtype=torch.float32)
            for key, value in labels.items()
        }

        t3=time()
        logger.debug(f"{(t1-t0):.6f}")
        logger.debug(f"{(t2 - t1):.6f} : {(t4 - t1):.6f} -> {(t2 - t5):.6f} -> {(t5-t4):.6f}")
        logger.debug(f"{(t3-t2):.6f}")
        logger.debug(f"tot: {(t3 - t0):.6f}")
        logger.debug(f"---------- {self._worker_id}")

        return x, labels
