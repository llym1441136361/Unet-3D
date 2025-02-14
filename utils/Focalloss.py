import torch
import torch.nn as nn
import numpy as np

class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma=2, reduction='mean', weight=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.crit = nn.CrossEntropyLoss(reduction='none', weight=self.alpha)
        # self.crit = nn.BCELoss(reduction='none')

    def forward(self, logits, label):
        ce_loss = self.crit(logits, label)
        # ce_lossn = ce_loss.cpu().detach().numpy()
        log_pt = ce_loss.neg()
        pt = torch.exp(log_pt)
        index = torch.argmin(pt).item()
        max_o = 1 - pt.view(-1)[index].item()
        weights = (1 - pt).pow(self.gamma)
        max_n = weights.view(-1)[index].item()
        weights = self.weight*weights*max_o/max_n
        fl = weights * ce_loss
        if self.reduction == 'mean':
            loss = fl.mean()
        if self.reduction == 'sum':
            loss = fl.sum()
        return loss
