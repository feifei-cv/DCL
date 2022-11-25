import os
import shutil
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from PIL import Image




if __name__ == '__main__':

    root = '/disks/disk0/feifei/data/semantic_data/bdd100k'
    list_path = '/disks/disk0/feifei/paper/paper5/dataset/lists/images_trainval_night_correct_filenames.txt'
    target_img_path = '/disks/disk0/feifei/paper/paper5/dataset/bdd100night/rgb'
    target_label_path = '/disks/disk0/feifei/paper/paper5/dataset/bdd100night/gt'
    img_paths = [i_id.strip() for i_id in open(list_path)]

    with open("bdd100k.txt", "a") as file:
        for img_path in img_paths:
            _, _, split, name = img_path.split('/')
            img_ori_path = os.path.join(root, 'images', '10k', split, name)
            img_test= os.path.join('bdd100night', 'rgb', name)
            file.write(img_test+'\n')
            shutil.copy(img_ori_path, target_img_path)


    with open("label_bdd100k.txt", "a") as file:
        for img_path in img_paths:
            _, _, split, name = img_path.split('/')
            label_ori_path = os.path.join(root, 'labels', 'sem_seg', 'masks', split,
                                          name.replace('.jpg', '.png'))
            label_test = os.path.join('bdd100night', 'gt', name.replace('.jpg', '.png'))
            file.write(label_test + '\n')
            shutil.copy(label_ori_path, target_label_path)




