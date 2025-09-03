import torch
from torch import nn, Tensor
import numpy as np


import torch.nn.functional as F

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print(f"[DEBUG] RobustCrossEntropyLoss: input type: {input.dtype}, shape: {input.shape}")
        # print(f"[DEBUG] RobustCrossEntropyLoss: target type: {target.dtype}, shape: {target.shape}")
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        
        # Explicitly call F.cross_entropy with correct input and target shapes
        return F.cross_entropy(input, target.long(), weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=self.label_smoothing)


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
