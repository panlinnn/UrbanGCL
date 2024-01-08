import torch
import numpy as np


def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)   # gt(a,b)  a>b则返回true, a<b则返回false
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))


def mape_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def rmse_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((true-pred) ** 2))


def mae_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred-true))


def mape_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))


def rmse_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.sqrt(np.mean((true-pred) ** 2))


def test_metrics(pred, true, mask1=5, mask2=5, mask3=5):
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae = mae_np(pred, true, mask1)
        mape = mape_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae  = mae_torch(pred, true, mask1).item()
        mape = mape_torch(pred, true, mask2).item()
    else:
        raise TypeError
    return mae, mape


def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = mae_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss