import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import nibabel
from torch.utils.data import DataLoader, random_split
from dataloader import Brats21Dataset
from modules2.Unet3D_res_at_da3 import UNet3D
import utils.surface_distance.metrics as surfdist
from utils.dice_score import dice_coeff
# from utils.Recall import recall
# from utils.Precision import precision


affine = np.array([[-1., -0., -0., 0.],
                   [-0., -1., -0., 239.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

save_dir =r'D:\Brats21\share\output\\'    #'/root/autodl-tmp/output/'


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
    # recall_score = [0, 0, 0]
    # precision_score = [0, 0, 0]
    surface_distances = [0, 0, 0]
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        volumes1, volumes2, volumes3, volumes4, true_masks0,_,_ = batch
        # images1 = torch.cat((volumes1, volumes2, volumes3, volumes4), dim=1)
        # images2 = volumes4
        images1 = volumes1.to(device=device, dtype=torch.float32, non_blocking=True)
        # images2 = images2.to(device=device, dtype=torch.float32, non_blocking=True)
        true_masks1 = true_masks0.to(device=device, dtype=torch.long, non_blocking=True)
        # new_label_masks0 = reshape_m(true_masks0)
        # new_label_masks0[new_label_masks0 == 4] = 3
        new_label_masks, n = reshape_m(true_masks1)
        new_label_masks1 = new_label_masks.copy()
        new_label_masks2 = new_label_masks.copy()
        new_label_masks3 = new_label_masks.copy()
        with torch.no_grad():
            # predict the mask
            mask_pred = net(images1)
        mask_pred = mask_pred.argmax(dim=1)
        mask_pred = mask_pred.cpu().numpy().astype(np.float32)

        mask_p1 = mask_pred.copy()
        mask_p1[mask_p1 == 1] = 0
        mask_p1[mask_p1 == 2] = 0
        mask_p1[mask_p1 == 3] = 1
        mask_b1 = mask_p1.astype(np.bool_)
        new_label_masks1[new_label_masks1 == 1] = 0
        new_label_masks1[new_label_masks1 == 2] = 0
        new_label_masks1[new_label_masks1 == 4] = 1
        new_label_masks1 = new_label_masks1.astype(np.bool_)
        mask_p1 = torch.tensor(mask_p1).to(device=device, dtype=torch.float32, non_blocking=True)
        new_label_masks_t1 = torch.tensor(new_label_masks1).to(device=device, dtype=torch.float32, non_blocking=True)
        mask_p2 = mask_pred.copy()
        mask_p2[mask_p2 == 1] = 1
        mask_p2[mask_p2 == 2] = 0
        mask_p2[mask_p2 == 3] = 1
        mask_b2 = mask_p2.astype(np.bool_)
        new_label_masks2[new_label_masks2 == 1] = 1
        new_label_masks2[new_label_masks2 == 2] = 0
        new_label_masks2[new_label_masks2 == 4] = 1
        new_label_masks2 = new_label_masks2.astype(np.bool_)
        mask_p2 = torch.tensor(mask_p2).to(device=device, dtype=torch.float32, non_blocking=True)
        new_label_masks_t2 = torch.tensor(new_label_masks2).to(device=device, dtype=torch.float32, non_blocking=True)
        mask_p3 = mask_pred.copy()
        mask_p3[mask_p3 == 1] = 1
        mask_p3[mask_p3 == 2] = 1
        mask_p3[mask_p3 == 3] = 1
        mask_b3 = mask_p3.astype(np.bool_)
        new_label_masks3[new_label_masks3 == 1] = 1
        new_label_masks3[new_label_masks3 == 2] = 1
        new_label_masks3[new_label_masks3 == 4] = 1
        new_label_masks3 = new_label_masks3.astype(np.bool_)
        mask_p3 = torch.tensor(mask_p3).to(device=device, dtype=torch.float32, non_blocking=True)
        new_label_masks_t3 = torch.tensor(new_label_masks3).to(device=device, dtype=torch.float32, non_blocking=True)
        # mask_pred[mask_pred == 3] = 4
        for label_id in range(n):
            dice_score[0] = dice_coeff(mask_p1[label_id], new_label_masks_t1[label_id], reduce_batch_first=False)
            dice_score[1] = dice_coeff(mask_p2[label_id], new_label_masks_t2[label_id], reduce_batch_first=False)
            dice_score[2] = dice_coeff(mask_p3[label_id], new_label_masks_t3[label_id], reduce_batch_first=False)
            surface_distances[0] = surfdist.compute_surface_distances(new_label_masks1[label_id], mask_b1[label_id], spacing_mm=(1.0, 1.0, 1.0))
            surface_distances[1] = surfdist.compute_surface_distances(new_label_masks2[label_id], mask_b2[label_id], spacing_mm=(1.0, 1.0, 1.0))
            surface_distances[2] = surfdist.compute_surface_distances(new_label_masks3[label_id], mask_b3[label_id], spacing_mm=(1.0, 1.0, 1.0))
            hd_dist_95_1  = surfdist.compute_robust_hausdorff(surface_distances[0], 95)
            hd_dist_95_2 = surfdist.compute_robust_hausdorff(surface_distances[1], 95)
            hd_dist_95_3 = surfdist.compute_robust_hausdorff(surface_distances[2], 95)
            result.write(f'{i} {dice_score[0]} {dice_score[1]} {dice_score[2]} {hd_dist_95_1} {hd_dist_95_2} {hd_dist_95_3}\n')
            print(f"dice: {dice_score[0]}{dice_score[1]}{dice_score[2]}")
            print(f"surface_distances: {hd_dist_95_1} {hd_dist_95_2} {hd_dist_95_3}")
            # vis_mask_p = nibabel.Nifti1Image(mask_pred[label_id], affine)
            # nibabel.save(vis_mask_p, f"{save_dir}{i} mask{dice_score[0]:.3f} {dice_score[1]:.3f} {dice_score[2]:.3f}.nii.gz")
            # vis_label_mask = nibabel.Nifti1Image(new_label_masks[label_id], affine)
            # nibabel.save(vis_label_mask, f"{save_dir}{i} label_mask.nii.gz")
            # vis_label_t1 = nibabel.Nifti1Image(volumes1_view[label_id], affine)
            # nibabel.save(vis_label_t1, f"{save_dir}{i} view_t1.nii.gz")
            # vis_label_t2 = nibabel.Nifti1Image(volumes2_view[label_id], affine)
            # nibabel.save(vis_label_t2, f"{save_dir}{i} view_t2.nii.gz")
            # vis_label_t1ce = nibabel.Nifti1Image(volumes3_view[label_id], affine)
            # nibabel.save(vis_label_t1ce, f"{save_dir}{i} view_t1ce.nii.gz")
            # vis_label_flair = nibabel.Nifti1Image(volumes4_view[label_id], affine)
            # nibabel.save(vis_label_flair, f"{save_dir}{i} view_flair.nii.gz")
            # recall_score[0] = recall(mask_p1[label_id], new_label_masks_t1[label_id], reduce_batch_first=False)
            # recall_score[1] = recall(mask_p2[label_id], new_label_masks_t2[label_id], reduce_batch_first=False)
            # recall_score[2] = recall(mask_p3[label_id], new_label_masks_t3[label_id], reduce_batch_first=False)
            # precision_score[0] = precision(mask_p1[label_id], new_label_masks_t1[label_id], reduce_batch_first=False)
            # precision_score[1] = precision(mask_p2[label_id], new_label_masks_t2[label_id], reduce_batch_first=False)
            # precision_score[2] = precision(mask_p3[label_id], new_label_masks_t3[label_id], reduce_batch_first=False)

            i = i + 1
    result.close()


if __name__ == '__main__':
    dataset = Brats21Dataset(
        '/root/Unet/datalistt.txt')  # r'D:\Brats21\datalist.txt'
    resume_path = '/root/autodl-tmp/trainmax/checkpoint1maxtmod1.pth'
    device = torch.device('cuda')
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_set = dataset
    batch_size = 1
    checkpoint = torch.load(resume_path, map_location='cpu')
    net = UNet3D(1, 4)
    net.load_state_dict(checkpoint, strict=False)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    net.to(device=device)
    evaluate(net, val_loader, device, batch_size)
