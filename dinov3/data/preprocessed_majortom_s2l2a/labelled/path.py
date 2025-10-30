from pathlib import Path


def compose_labels_path(dataset_path: Path) -> Path:
    return dataset_path / "useful_files" / "data_static" / "combined.tif"
