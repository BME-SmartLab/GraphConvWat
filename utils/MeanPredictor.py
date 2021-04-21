# -*- coding: utf-8 -*-
import torch

class MeanPredictor():
    def __init__(self, device):
        pass

    def pred(self, y_true, mask=None):
        if mask is None:
            raise NotImplementedError
        else:
            y_pred  = torch.zeros_like(y_true).squeeze(dim=1)
            pred_val= torch.masked_select(y_true[:, 0], mask).mean()
            y_pred  += (pred_val*~mask)
            y_pred  += torch.multiply(y_true.squeeze(dim=1), mask)
            return y_pred
