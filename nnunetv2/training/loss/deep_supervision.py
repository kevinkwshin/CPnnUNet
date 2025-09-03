from torch import nn
import torch

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!
        losses =[]
        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        for i, inputs in enumerate(zip(*args)):
            if weights[i] != 0.0:
                loss_val = self.loss(*inputs)
                # print(f"[DEBUG] DeepSupervisionWrapper: weights[i]={weights[i]}, loss_val type={type(loss_val)}, shape={loss_val.shape if isinstance(loss_val, torch.Tensor) else 'N/A'}")
                losses.append(float(weights[i].item()) * loss_val)
        return sum(losses)
        # print(f"[DEBUG] DeepSupervisionWrapper: weights[i] type: {type(weights[i])}, inputs: {inputs}")
        # print(f"[DEBUG] DeepSupervisionWrapper inputs: {inputs}")
