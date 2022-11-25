import os
import torch
from PIL import Image
import numpy as np

from dataset.transforms import *
import torchvision.transforms as standard_transforms



class ACDC(torch.utils.data.Dataset):

    orig_dims = (1080, 1920)
    def __init__(self, args, root, stage="train", dims= (1080, 1920)):

        super().__init__()
        self.root = root
        self.dims = dims

        acdcnight_transform_list = [
            joint_transforms2.RandomSizeAndCrop(args.input_size_target, False, pre_size=None,
                                                scale_min=0.9, scale_max=1.1, ignore_index=255),
            joint_transforms2.Resize(args.input_size),
            joint_transforms2.RandomHorizontallyFlip()]
        self.joint_transform = joint_transforms2.Compose(acdcnight_transform_list)

        train_input_transform = []
        train_input_transform += [standard_transforms.ToTensor(),
                                  standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        self.target_transform = extended_transforms.MaskToTensor()
        self.transform = standard_transforms.Compose(train_input_transform)

        self.stage = stage
        if self.stage == 'train':
            self.split = 'train'
        elif self.stage == 'val':
            self.split = 'val'
        elif self.stage == 'test':
            self.split = 'val'

        self.images_dir = os.path.join(self.root, 'rgb_anon')
        self.semantic_dir = os.path.join(self.root, 'gt')
        img_parent_dir = os.path.join(self.images_dir, "night", self.split)

        self.files = []
        for recording in os.listdir(img_parent_dir):
            img_dir = os.path.join(img_parent_dir, recording)
            for file_name in os.listdir(img_dir):

                img_night = os.path.join(img_dir, file_name)
                ref_img_dir = img_dir.replace(self.split, self.split + '_ref')
                ref_file_name = file_name.replace('rgb_anon', 'rgb_ref_anon')
                img_day = os.path.join(ref_img_dir, ref_file_name)
                self.files.append({
                    "img_night": img_night,
                    "img_day": img_day,
                    "name": ref_file_name
                })



    def __getitem__(self, index):

        datafiles = self.files[index]
        image_n = Image.open(datafiles["img_night"]).convert('RGB')
        image_d = Image.open(datafiles["img_day"]).convert('RGB')

        name = datafiles["name"]

        if self.joint_transform is not None:
            image_n, image_d = self.joint_transform(image_n, image_d)

        if self.transform is not None:
            image_n = self.transform(image_n)
            image_d = self.transform(image_d)

        size = image_n.shape
        return image_n, image_d, np.array(size), name

    def __len__(self) -> int:
        return len(self.files)


if __name__ == '__main__':

    from configs.train_config import get_arguments
    from torch.utils.data import DataLoader
    args = get_arguments()

    acdc_data = ACDC(args, root='/disks/disk0/feifei/data/semantic_data/ACDC')
    data_loder = DataLoader(dataset=acdc_data, batch_size=2,shuffle=True, num_workers=2, pin_memory=True)

    targetloader_iter = enumerate(data_loder)
    _, batch = targetloader_iter.__next__()
    images_n, images_d, _, _ = batch

    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.utils
    im_d = torchvision.utils.make_grid(images_n, normalize=True, padding=5)
    im_d_1 = np.transpose(im_d.cpu(), (1, 2, 0))
    plt.imshow(im_d_1)
    plt.show()

    im_n = torchvision.utils.make_grid(images_d, normalize=True, padding=5)
    im_n_1 = np.transpose(im_n.cpu(), (1, 2, 0))
    plt.imshow(im_n_1)
    plt.show()

    print('finish')
