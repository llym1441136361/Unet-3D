import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net1, dataloader, device, batch_size, reshape_m):
    net1.eval()
    # net2.eval()
    num_val_batches = len(dataloader)
    dice_score1 = [0, 0, 0]
    total_dice_score1 = [0, 0, 0]
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        volumes1, volumes2, volumes3, volumes4, true_masks0, true_masks1, true_masks2 = batch
        images1 = torch.cat((volumes1, volumes2, volumes3, volumes4), dim=1)
        # images2 = volumes4
        images1 = images1.to(device=device, dtype=torch.float32, non_blocking=True)
        # images2 = images2.to(device=device, dtype=torch.float32, non_blocking=True)
        true_masks1 = true_masks0.to(device=device, dtype=torch.long, non_blocking=True)
        # new_label_masks0 = reshape_m(true_masks0)
        # new_label_masks0[new_label_masks0 == 4] = 3
        new_label_masks = reshape_m(true_masks1)
        new_label_masks1 = new_label_masks.copy()
        new_label_masks2 = new_label_masks.copy()
        new_label_masks3 = new_label_masks.copy()
        with torch.no_grad():
            # predict the mask
            mask_pred = net1(images1)
        mask_pred = mask_pred.argmax(dim=1)
        mask_pred = mask_pred.cpu().numpy().astype(np.float32)
        mask_p1 = mask_pred.copy()
        mask_p1[mask_p1 == 1] = 0
        mask_p1[mask_p1 == 2] = 0
        mask_p1[mask_p1 == 3] = 1
        new_label_masks1[new_label_masks1 == 1] = 0
        new_label_masks1[new_label_masks1 == 2] = 0
        new_label_masks1[new_label_masks1 == 4] = 1
        mask_p1 = torch.tensor(mask_p1).to(device=device, dtype=torch.float32, non_blocking=True)
        new_label_masks_t1 = torch.tensor(new_label_masks1).to(device=device, dtype=torch.float32,
                                                               non_blocking=True)
        mask_p2 = mask_pred.copy()
        mask_p2[mask_p2 == 1] = 1
        mask_p2[mask_p2 == 2] = 0
        mask_p2[mask_p2 == 3] = 1
        new_label_masks2[new_label_masks2 == 1] = 1
        new_label_masks2[new_label_masks2 == 2] = 0
        new_label_masks2[new_label_masks2 == 4] = 1
        mask_p2 = torch.tensor(mask_p2).to(device=device, dtype=torch.float32, non_blocking=True)
        new_label_masks_t2 = torch.tensor(new_label_masks2).to(device=device, dtype=torch.float32,
                                                               non_blocking=True)
        mask_p3 = mask_pred.copy()
        mask_p3[mask_p3 == 1] = 1
        mask_p3[mask_p3 == 2] = 1
        mask_p3[mask_p3 == 3] = 1
        new_label_masks3[new_label_masks3 == 1] = 1
        new_label_masks3[new_label_masks3 == 2] = 1
        new_label_masks3[new_label_masks3 == 4] = 1
        mask_p3 = torch.tensor(mask_p3).to(device=device, dtype=torch.float32, non_blocking=True)
        new_label_masks_t3 = torch.tensor(new_label_masks3).to(device=device, dtype=torch.float32,
                                                                   non_blocking=True)
        for label_id in range(batch_size):
            dice_score1[0] = dice_coeff(mask_p1[label_id], new_label_masks_t1[label_id], reduce_batch_first=False).cpu().numpy()
            dice_score1[1] = dice_coeff(mask_p2[label_id], new_label_masks_t2[label_id], reduce_batch_first=False).cpu().numpy()
            dice_score1[2] = dice_coeff(mask_p3[label_id], new_label_masks_t3[label_id], reduce_batch_first=False).cpu().numpy()
            total_dice_score1 = np.array(total_dice_score1) + np.array(dice_score1)
    net1.train()
    # net2.train()

    return total_dice_score1/(num_val_batches * batch_size) #, total_dice_score2/(num_val_batches * batch_size), total_dice_score_sum/(num_val_batches * batch_size)
