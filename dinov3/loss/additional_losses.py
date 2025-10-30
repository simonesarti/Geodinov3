import torch.nn.functional as F
from torch import nn


def reshape_labels_weights(pred, labels, weights):
    vb, d = pred.shape          # viewed batch size (views * batch_size), dim
    b, l = labels.shape         # (original batch size, label dim)
    _, w = weights.shape        # (original batch size, weight dim)
    v = vb // b                 # number of views per sample

    # repeat batch V times (AB -> AB AB AB)
    labels = labels.repeat(v, 1)
    weights = weights.repeat(v, 1)

    weights = weights.squeeze()  # one loss value per sample shape [BV, ], match with [W, ] instead of [W,1]

    # collation of crops works like this:
    # crop0 img0, crop0 img1, crop0 img2,   -> all crop0s (sequence of images)
    # crop1 img 0, crop 1 img1, crop1 img2, -> all crop1s (sequence of images)

    # Therefore i want labels and weights to be:
    # lbl img0, lbl img1, lbl img2,     -> sequence of image labels
    # lbl img0, lbl img1, lbl img2      -> sequence of image labels

    return labels, weights


class SampleWeightedMSELoss(nn.Module):
    def __init__(self):
        super(SampleWeightedMSELoss, self).__init__()

    def forward(self, pred, labels, weights):
        # Repeat labels and weights to match pred shape
        labels, weights = reshape_labels_weights(pred, labels, weights)
        # Sample-wise Mean Squared Error
        mse_per_element = F.mse_loss(pred, labels, reduction='none')    # [VB, F]
        mse_per_sample = mse_per_element.mean(dim=1)    # [VB, ] avoids scaling with number of features
        # Weight each sample
        weighted_mse = mse_per_sample * weights
        # compute weighted mean
        return weighted_mse.sum() / weights.sum()


class SampleWeightedBCELoss(nn.Module):
    def __init__(self):
        super(SampleWeightedBCELoss, self).__init__()

    def forward(self, pred, labels, weights):
        # Repeat labels and weights to match pred shape
        labels, weights = reshape_labels_weights(pred, labels, weights)
        # Sample-wise Binary cross-entropy
        bce_per_sample = F.binary_cross_entropy_with_logits(pred, labels, reduction='none')    # [VB, ]
        # Weight each sample
        weighted_bce = bce_per_sample * weights
        # compute weighted average
        return weighted_bce.sum() / weights.sum()


class SampleWeightedCCELoss(nn.Module):
    def __init__(self):
        super(SampleWeightedCCELoss, self).__init__()

    def forward(self, pred, labels, weights):
        # Repeat labels and weights to match pred shape
        labels, weights = reshape_labels_weights(pred, labels, weights)
        # Sample-wise Categorical cross-entropy
        cce_per_sample = F.cross_entropy(pred, labels, reduction='none')   # [VB, ]
        # Weight each sample
        weighted_cce = cce_per_sample * weights
        # compute weighted average
        return weighted_cce.sum() / weights.sum()


# ========================== SINGLE NETWORK LOSSES =================================


class BuildingsLoss(SampleWeightedMSELoss):         # MSE
    def __init__(self):
        super(BuildingsLoss, self).__init__()


class ClimateLoss(SampleWeightedCCELoss):           # CCE
    def __init__(self):
        super(ClimateLoss, self).__init__()


class CloudsLoss(SampleWeightedCCELoss):            # CCE
    def __init__(self):
        super(CloudsLoss, self).__init__()


class CoordsLoss(SampleWeightedMSELoss):            # MSE
    def __init__(self):
        super(CoordsLoss, self).__init__()


class LandcoverLoss(SampleWeightedCCELoss):         # CCE
    def __init__(self):
        super(LandcoverLoss, self).__init__()


class TerrainLoss(SampleWeightedCCELoss):           # CCE
    def __init__(self):
        super(TerrainLoss, self).__init__()


class UrbanizationLoss(SampleWeightedCCELoss):      # CCE
    def __init__(self):
        super(UrbanizationLoss, self).__init__()


class WaterLoss(SampleWeightedMSELoss):             # MSE
    def __init__(self):
        super(WaterLoss, self).__init__()
