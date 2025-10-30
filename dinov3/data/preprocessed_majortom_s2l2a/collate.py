import torch
from time import time

def collate_array(batch):
    """
    Custom collate function for MajorTOM when samples are yielded as arrays.

    Args:
        batch (list of tuples): Each element is a tuple (image_data, labels), where:
            - image_data is a (bands, H, W) tensor.
            - labels is a dict containing label tensors.

    Returns:
        tuple:
            - collated_images: Stacked image tensors.
            - collated_labels: Dictionary of stacked label tensors.
    """

    collated_images = torch.stack([sample[0] for sample in batch], dim=0)

    collated_labels = None
    if batch[0][1] is not None:     # batch[0][1] is the labels [1] for the first sample [0]
        collated_labels = {}
        for label_key in batch[0][1].keys():
            collated_labels[label_key] = torch.stack([sample[1][label_key] for sample in batch], dim=0)

    return collated_images, collated_labels


def collate_multicrop_array(batch):
    """
    Custom collate function for MajorTOM when data transforms produce
    multiple views/images per sample.

    Args:
        batch (list of tuples): Each element is a tuple (image_list, labels), where:
            - image_list is a list of (bands, H, W) tensors (one per view).
            - labels is a dict containing label tensors.

    Returns:
        tuple:
            - collated_images: List of stacked image tensors, one per view.
              Each element has shape (batch_size, bands, H, W).
            - collated_labels: Dictionary of stacked label tensors.
    """

    # Extract list of images crops
    image_lists = [sample[0] for sample in batch]

    # Determine number of views, all samples have the same number of views
    num_views = len(image_lists[0])

    # Collate images per view
    collated_images = []
    for view_idx in range(num_views):
        # Stack all images for this view across the batch
        view_images = torch.stack([img_list[view_idx] for img_list in image_lists], dim=0)
        collated_images.append(view_images)

    collated_labels = None
    if batch[0][1] is not None:     # batch[0][1] is the labels [1] for the first sample [0]
        collated_labels = {}
        for label_key in batch[0][1].keys():
            collated_labels[label_key] = torch.stack([sample[1][label_key] for sample in batch], dim=0)

    return collated_images, collated_labels
