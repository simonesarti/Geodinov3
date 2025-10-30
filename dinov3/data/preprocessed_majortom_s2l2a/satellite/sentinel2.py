from pathlib import Path

import numpy as np
import pandas as pd

S2_LEVELS = ("L2A", "L1C")

# TODO: WANRNING: MISSING B01, B09, B10 IN DOWNLOADED DATASET
# S2_BANDS_NAMES = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
S2_BANDS_NAMES = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


def get_s2_num_bands() -> int:
    return len(S2_BANDS_NAMES)


def get_s2_num_bands_with_clouds() -> int:
    return len(S2_BANDS_NAMES) + 1


def check_valid_s2_level(level: str) -> bool:
    return level in S2_LEVELS


def get_valid_s2_band_names(band_names: tuple[str]) -> tuple[str]:
    band_names = tuple(set(band_names))
    valid_band_names = [s2_band_name for s2_band_name in S2_BANDS_NAMES if s2_band_name in band_names]
    return tuple(valid_band_names)


def get_sentienl2_bands_mask(band_names: tuple[str]) -> tuple[bool]:
    mask = [s2_band_name in band_names for s2_band_name in S2_BANDS_NAMES]
    return tuple(mask)


def normalize_sentinel2(data: np.ndarray) -> np.ndarray:
    return data / 10_000


def compose_sentinel2_data_path(dataset_path: Path, level: str, df_row: pd.Series) -> Path:
    if level == "L2A":
        data_path = dataset_path / "Core-S2L2A" / "L2A"
        data_path = data_path / df_row['row'] / f"{df_row['row']}_{df_row['column']}"
        data_path = data_path / df_row['L2A_name']
    elif level == "L1C":
        data_path = dataset_path / "Core-S2L1C" / "L1C"
        data_path = data_path / df_row['row'] / f"{df_row['row']}_{df_row['column']}"
        data_path = data_path / df_row['L1C_name']
    else:
        raise NotImplementedError(f"Level {level} not implemented for Sentinel2, choose between ['L2A', 'L1C']")

    return data_path


def compose_sentinel2_with_clouds_bands_path(dataset_path: Path, level: str, df_row: pd.Series) -> Path:

    data_path = compose_sentinel2_data_path(dataset_path, level, df_row)
    return data_path / "combined.tif"


def compose_sentinel2_thumbnail_path(dataset_path: Path, level: str, df_row: pd.Series) -> Path:

    data_path = compose_sentinel2_data_path(dataset_path, level, df_row)
    return data_path / "thumbnail.png"
