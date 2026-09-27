import torch
from torch import nn


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, epsilon=1e-7):

        inputs=torch.sigmoid(inputs)

        inputs=inputs.view(-1)
        targets=targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2 * intersection + epsilon) / (inputs.sum() + targets.sum() + epsilon)
        bce=nn.functional.binary_cross_entropy(inputs, targets, reduction="mean")

        return dice_loss + bce