import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import nibabel
from torch.utils.data import DataLoader, random_split
from dataloader import Brats21Dataset
from modules2.Unet3D_res_at_da3 import UNet3D
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.dice_score import dice_loss

affine = np.array([[-1., -0., -0., 0.],
                   [-0., -1., -0., 239.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

save_dir = '/root/autodl-tmp/output/'

def reshape_m(true_masks):
    [n, _, h, w, d] = true_masks.shape
    new_label_masks = np.zeros([n, h, w, d])
    for label_id in range(n):
        label_mask = true_masks[label_id].cpu().numpy()
        [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
        label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
        new_label_masks[label_id] = label_mask
    return new_label_masks, n


def evaluate(net, dataloader, device, batch_size):
    net.eval()
    num_val_batches = len(dataloader)
    i = 0
    result = open("result.txt", "w")
    dice_score = [0, 0, 0]
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
        volumes1_view, _ = reshape_m(volumes1)
        volumes2_view, _ = reshape_m(volumes2)
        volumes3_view, _ = reshape_m(volumes3)
        volumes4_view, _ = reshape_m(volumes4)
        new_label_masks, n = reshape_m(true_masks1)
        new_label_masks1 = new_label_masks.copy()
        new_label_masks2 = new_label_masks.copy()
        new_label_masks3 = new_label_masks.copy()
        # new_label_masks2 = reshape_m(true_masks2)
        # new_label_masks_t0 = torch.tensor(new_label_masks0).to(device=device, dtype=torch.long, non_blocking=True)
        with torch.no_grad():
            # predict the mask
            mask_pred = net(images1)
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
        new_label_masks_t1 = torch.tensor(new_label_masks1).to(device=device, dtype=torch.float32, non_blocking=True)
        mask_p2 = mask_pred.copy()
        mask_p2[mask_p2 == 1] = 1
        mask_p2[mask_p2 == 2] = 0
        mask_p2[mask_p2 == 3] = 1
        new_label_masks2[new_label_masks2 == 1] = 1
        new_label_masks2[new_label_masks2 == 2] = 0
        new_label_masks2[new_label_masks2 == 4] = 1
        mask_p2 = torch.tensor(mask_p2).to(device=device, dtype=torch.float32, non_blocking=True)
        new_label_masks_t2 = torch.tensor(new_label_masks2).to(device=device, dtype=torch.float32, non_blocking=True)
        mask_p3 = mask_pred.copy()
        mask_p3[mask_p3 == 1] = 1
        mask_p3[mask_p3 == 2] = 1
        mask_p3[mask_p3 == 3] = 1
        new_label_masks3[new_label_masks3 == 1] = 1
        new_label_masks3[new_label_masks3 == 2] = 1
        new_label_masks3[new_label_masks3 == 4] = 1
        mask_p3 = torch.tensor(mask_p3).to(device=device, dtype=torch.float32, non_blocking=True)
        new_label_masks_t3 = torch.tensor(new_label_masks3).to(device=device, dtype=torch.float32, non_blocking=True)
        mask_pred[mask_pred == 3] = 4
        for label_id in range(n):
            # mask_pred1 = F.one_hot(mask_pr1.to(device=device, dtype=torch.long), num_classes=4).permute(0, 4, 1, 2, 3).float()
            dice_score[0] = dice_coeff(mask_p1[label_id], new_label_masks_t1[label_id], reduce_batch_first=False)
            # mask_pred1 = F.one_hot(mask_pr1.to(device=device, dtype=torch.long), num_classes=4).permute(0, 4, 1, 2, 3).float()
            dice_score[1] = dice_coeff(mask_p2[label_id], new_label_masks_t2[label_id], reduce_batch_first=False)
            # mask_pred1 = F.one_hot(mask_pr1.to(device=device, dtype=torch.long), num_classes=4).permute(0, 4, 1, 2, 3).float()
            dice_score[2] = dice_coeff(mask_p3[label_id], new_label_masks_t3[label_id], reduce_batch_first=False)
            vis_mask_p = nibabel.Nifti1Image(mask_pred[label_id], affine)
            nibabel.save(vis_mask_p, f"{save_dir}{i} mask{dice_score[0]:.3f} {dice_score[1]:.3f} {dice_score[2]:.3f}.nii.gz")
            vis_label_mask = nibabel.Nifti1Image(new_label_masks[label_id], affine)
            nibabel.save(vis_label_mask, f"{save_dir}{i} label_mask.nii.gz")
            vis_label_t1 = nibabel.Nifti1Image(volumes1_view[label_id], affine)
            nibabel.save(vis_label_t1, f"{save_dir}{i} view_t1.nii.gz")
            vis_label_t2 = nibabel.Nifti1Image(volumes2_view[label_id], affine)
            nibabel.save(vis_label_t2, f"{save_dir}{i} view_t2.nii.gz")
            vis_label_t1ce = nibabel.Nifti1Image(volumes3_view[label_id], affine)
            nibabel.save(vis_label_t1ce, f"{save_dir}{i} view_t1ce.nii.gz")
            vis_label_flair = nibabel.Nifti1Image(volumes4_view[label_id], affine)
            nibabel.save(vis_label_flair, f"{save_dir}{i} view_flair.nii.gz")
            result.write(f'{i} {dice_score[0]} {dice_score[1]} {dice_score[2]}\n')
            # print(f'dice single{dice_score}')
            i = i+1
    result.close()


if __name__ == '__main__':
    dataset = Brats21Dataset(
        '/root/Unet/datalistt.txt')  # r'D:\Brats21\datalist.txt'
    resume_path = '/root/autodl-tmp/trainmax/checkpoint1maxt12.5.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_set = dataset
    batch_size = 1
    checkpoint = torch.load(resume_path, map_location='cpu')
    net = UNet3D(4, 4)
    net.load_state_dict(checkpoint, strict=False)
    loader_args = dict(batch_size=batch_size, num_workers=15, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    net.to(device=device)
    evaluate(net, val_loader, device, batch_size)
