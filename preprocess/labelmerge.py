import os
import numpy as np
import nibabel


class Merge:
    def __init__(self, img_list):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        file_num = len(self.img_list)
        for idx in range(file_num):
            ith_info = self.img_list[idx].split(" ")
            mask = nibabel.load(ith_info[4])
            mask_affine = mask.affine
            # mask = self.remove_2(mask)
            # output_dir = ith_info[4].replace('seg', 'seg_change')
            mask = self.change2_4to1(mask)
            output_dir = ith_info[4].replace('seg', 'seg_merge')
            print(output_dir)
            mask = nibabel.Nifti1Image(np.array(mask).astype(np.float32), mask_affine)
            mask.to_filename(f'{output_dir}')

    def remove_2(self, data):
        data = data.get_fdata()
        data[data == 2] = 0
        data[data == 4] = 2
        return data

    def change2_4to1(self, data):
        data = data.get_fdata()
        data[data == 2] = 1
        data[data == 4] = 1
        return data



if __name__ == '__main__':
    brain = Merge('/content/drive/MyDrive/Tasker/datalist.txt')
