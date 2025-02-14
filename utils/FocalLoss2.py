import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class MultiCEFocalLoss(torch.nn.Module):

    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.class_num).permute(0, 4, 1, 2, 3)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1, 1)].to('cuda')
        probs = (pt * class_mask).sum(1).view(-1, 1) 
        log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
