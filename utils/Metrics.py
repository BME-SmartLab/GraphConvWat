# -*- coding: utf-8 -*-
import torch

class Metrics():
    def __init__(self, bias, scale, device):
        self.bias   = torch.tensor(bias).to(device)
        self.scale  = torch.tensor(scale).to(device)

    def _rescale(self, data):
        return torch.add(
                torch.multiply(
                    data,
                    self.scale
                    ),
                self.bias
                )

    def rel_err(self, y_pred, y_true, mask=None):
        if mask is None:
            y_pred  = y_pred[:, 0]
            y_true  = y_true[:, 0]
        else:
            y_pred  = torch.masked_select(y_pred[:, 0], mask)
            y_true  = torch.masked_select(y_true[:, 0], mask)
        y_pred  = self._rescale(y_pred)
        y_true  = self._rescale(y_true)
        err     = torch.subtract(y_true, y_pred)
        rel_err = torch.abs(torch.divide(err, y_true))
        return torch.mean(rel_err)
