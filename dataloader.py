import os

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage


class Brats21Dataset(Dataset):

    def __init__(self, img_list):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.img_list)))
        # self.root_dir = root_dir
        # self.input_D = sets.input_D
        # self.input_H = sets.input_H
        # self.input_W = sets.input_W
        # self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [x, y, z] = data.shape
        new_data = np.reshape(data, (1, x, y, z))
        new_data = new_data.astype("float32")
        return new_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels
        ith_info = self.img_list[idx].split(" ")
        img1 = nibabel.load(ith_info[0])
        img2 = nibabel.load(ith_info[1])
        img3 = nibabel.load(ith_info[2])
        img4 = nibabel.load(ith_info[3])
        mask0 = nibabel.load(ith_info[4])
        mask1 = nibabel.load(ith_info[5])
        mask2 = nibabel.load(ith_info[6])
        assert mask1 is not None
        # img_array, mask_array = self.__training_data_process__(img1, mask)
        # data processing
        img1, img2, img3, img4, mask0, mask1, mask2 = self.__training_data_process__(img1, img2, img3, img4, mask0,
                                                                                     mask1, mask2)

        # tensor array
        img1_array = self.__nii2tensorarray__(img1)
        img2_array = self.__nii2tensorarray__(img2)
        img3_array = self.__nii2tensorarray__(img3)
        img4_array = self.__nii2tensorarray__(img4)
        mask_array0 = self.__nii2tensorarray__(mask0)
        mask_array1 = self.__nii2tensorarray__(mask1)
        mask_array2 = self.__nii2tensorarray__(mask2)

        assert img1_array.shape == mask_array1.shape, "img shape:{} is not equal to mask shape:{}".format(
            img1_array.shape, mask_array1.shape)
        return img1_array, img2_array, img3_array, img4_array, mask_array0, mask_array1, mask_array2

    def __drop_invalid_range__(self, img1, img2, img3, img4, mask0, mask1, mask2):
        """
        裁剪无效区域即背景部分
        """
        zero_value = img1[0, 0, 0]
        non_zeros_idx = np.where(img1 != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        return img1[min_z:max_z, min_h:max_h, min_w:max_w], img2[min_z:max_z, min_h:max_h, min_w:max_w], \
               img3[min_z:max_z, min_h:max_h, min_w:max_w], img4[min_z:max_z, min_h:max_h, min_w:max_w], \
               mask0[min_z:max_z, min_h:max_h, min_w:max_w], mask1[min_z:max_z, min_h:max_h, min_w:max_w], \
               mask2[min_z:max_z, min_h:max_h, min_w:max_w]

    def __random_center_crop__(self, data, label):
        from random import random
        """
        随机裁切，可以提高泛用性，我没用这部分，测试效果不太好
        """
        target_indexs = np.where(label > 0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth * 1.0 / 2) * random())
        Y_min = int((min_H - target_height * 1.0 / 2) * random())
        X_min = int((min_W - target_width * 1.0 / 2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __itensity_normalize_one_volume__(self, volume):
        """
        归一化，我用的是最大最小归一化
        """
        pixels = volume[volume > 0]
        min = pixels.min()
        max = pixels.max()
        out = (volume - min) / max
        out_random = np.zeros(volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        尺寸缩放，可以改数字修改尺寸
        """
        [height, width, depth] = data.shape
        scale = [128.0 / height, 128.0 / width, 112.0 / depth]
        data = ndimage.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data, label):
        """
        这也是随机裁切，另一种方法，也没有用
        """
        # random center crop
        data, label = self.__random_center_crop__(data, label)

        return data, label

    def __training_data_process__(self, img1, img2, img3, img4, mask0, mask1, mask2):
        """
        整个预处理部分的执行过程
        """
        # 转换数据格式
        img1 = img1.get_fdata()
        img2 = img2.get_fdata()
        img3 = img3.get_fdata()
        img4 = img4.get_fdata()
        mask0 = mask0.get_fdata()
        mask1 = mask1.get_fdata()
        mask2 = mask2.get_fdata()

        # 剔除无效数据
        img1, img2, img3, img4, mask0, mask1, mask2 = self.__drop_invalid_range__(img1, img2, img3, img4, mask0, mask1,
                                                                                  mask2)

        # 调整尺寸
        img1 = self.__resize_data__(img1)
        img2 = self.__resize_data__(img2)
        img3 = self.__resize_data__(img3)
        img4 = self.__resize_data__(img4)
        mask0 = self.__resize_data__(mask0)
        mask1 = self.__resize_data__(mask1)
        mask2 = self.__resize_data__(mask2)

        # 归一化
        img1 = self.__itensity_normalize_one_volume__(img1)
        img2 = self.__itensity_normalize_one_volume__(img2)
        img3 = self.__itensity_normalize_one_volume__(img3)
        img4 = self.__itensity_normalize_one_volume__(img4)

        return img1, img2, img3, img4, mask0, mask1, mask2

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data


if __name__ == '__main__':
    brain = Brats21Dataset(r'D:\Brats21\datalist.txt')
    i = brain.__getitem__(2)
    print(i[4])
