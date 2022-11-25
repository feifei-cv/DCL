import os.path as osp
import numpy as np
import random
from torch.utils import data
from PIL import Image
from dataset.transforms import *
import torchvision.transforms as standard_transforms
from day2night import color_transfer


class cityscapesDataSet(data.Dataset):

    def __init__(self, args, root, list_path, max_iters=None, set='val'):
        self.root = root
        self.list_path = list_path

        train_input_transform = []
        train_input_transform += [standard_transforms.ToTensor(),
                                  standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        cityscape_transform_list = [
                                    joint_transforms.RandomSizeAndCrop(args.input_size, False, pre_size=None,
                                                                       scale_min=0.5, scale_max=1.0, ignore_index=255),
                                    joint_transforms.Resize(args.input_size),
                                    joint_transforms.RandomHorizontallyFlip()]
        self.joint_transform = joint_transforms.Compose(cityscape_transform_list)

        self.target_transform = extended_transforms.MaskToTensor()
        self.transform = standard_transforms.Compose(train_input_transform)

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name)).replace("leftImg8bit", "gtFine_labelIds")
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        data_set = 'zurich'
        if data_set == 'zurich': ### random zurich target night
            list_path_target = '/disks/disk0/feifei/paper/paper5/dataset/lists/zurich_dn_pair_train.csv'
            root_target = '/disks/disk0/feifei/data/semantic_data/Zurich/train/rgb_anon'
            self.files_night_target = []
            self.img_ids_target = [i_id.strip() for i_id in open(list_path_target)]
            for pair in self.img_ids_target:
                night, day = pair.split(",")
                img_night = osp.join(root_target, "%s" % (night)+"_rgb_anon.png")
                self.files_night_target.append(img_night)
        elif data_set == 'acdc': ### random ACDC target night
            list_path_target = '/disks/disk0/feifei/paper/paper5/dataset/lists/acdc_night_train.txt'
            root_target = '/disks/disk0/feifei/data/semantic_data/ACDC/rgb_anon'
            self.files_night_target = []
            self.img_ids_target = [i_id.strip() for i_id in open(list_path_target)]
            for pair in self.img_ids_target:
                img_night = osp.join(root_target, pair)
                self.files_night_target.append(img_night)

        #############
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')

        ##### Trans
        img_file = random.choice(self.files_night_target)
        img2 = Image.open(img_file).convert('RGB')
        ### BGR
        img1 = np.asarray(image)[:, :, ::-1]  ## day: BGR
        img2 = np.asarray(img2)[:, :, ::-1]  ## random night: BGR
        image_n1 = color_transfer(img1, img2)
        image_n = Image.fromarray(image_n1[:,:,::-1].astype('uint8')).convert('RGB')

        ###########
        import cv2
        # cv2.imwrite("imgs/source.jpg", img1)
        # cv2.imwrite("imgs/target.jpg", img2)
        # image_n_arr = np.asarray(image_n)[:, :, ::-1] ## trans: BGR
        # cv2.imshow("image_n_arr1", img1)
        # cv2.imshow("image_n_arr2", img2)
        # cv2.imshow("image_n_arr3", image_n_arr)
        # cv2.waitKey()
        # cv2.imwrite("imgs/night.jpg", image_n_arr)
        # ##########

        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        label = np.asarray(label, np.uint8)
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy.astype(np.uint8))
        if self.joint_transform is not None:
            image, image_n, label = self.joint_transform(image, image_n, label)

        ##############
        # im_test= np.asarray(image)[:, :, ::-1] ## BGR
        # im_test1 = np.asarray(image_n)[:, :, ::-1]
        # import cv2
        # cv2.imshow("im_test", im_test)
        # cv2.imshow("im_test1", im_test1)
        # cv2.waitKey()
        # cv2.imwrite("imgs/source_d.jpg", im_test)
        # cv2.imwrite("imgs/source_n.jpg", im_test1)
        # ###############

        if self.transform is not None:
            image = self.transform(image)
            image_n = self.transform(image_n)
        if self.target_transform is not None:
            label = self.target_transform(label)
        size = image.shape

        ######
        # import matplotlib.pyplot as plt
        # import torchvision
        ###########
        # im_d = torchvision.utils.make_grid(image.unsqueeze(dim=0), normalize=True, padding=5)
        # im_d_1 = np.transpose(im_d.cpu(), (1, 2, 0))
        # plt.imshow(im_d_1)
        # plt.show()
        # im_n = torchvision.utils.make_grid(image_n.unsqueeze(dim=0), normalize=True, padding=5)
        # im_n_1 = np.transpose(im_n.cpu(), (1, 2, 0))
        # plt.imshow(im_n_1)
        # plt.show()
        ###########
        # palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170,
        #            30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0,
        #            70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        # zero_pad = 256 * 3 - len(palette)
        # for i in range(zero_pad):
        #     palette.append(0)
        # labels = np.asarray(label.cpu())
        # labels = Image.fromarray(labels.astype(np.uint8)).convert('P')
        # labels.putpalette(palette)
        # labels.save('1.png')
        # labels = np.asarray(label.cpu())
        # labels = Image.fromarray(labels.astype(np.uint8)).convert('P')
        # labels.putpalette(palette)
        # labels.save('2.png')
        ########

        return image, label, image_n, label, np.array(size), name

if __name__ == '__main__':

    from configs.train_config import get_arguments
    args = get_arguments()
    trainloader = data.DataLoader(cityscapesDataSet(args, args.data_dir,
                                    list_path='/disks/disk0/feifei/paper/paper5/dataset/lists/cityscapes_train.txt',
                                                    max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                    set=args.set),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    _, batch = trainloader_iter.__next__()
    image, label, image_n, _, _, _ = batch

    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.utils

    im_d = torchvision.utils.make_grid(image, normalize=True, padding=5)
    im_d_1 = np.transpose(im_d.cpu(), (1, 2, 0))
    plt.imshow(im_d_1)
    plt.show()
    im_n = torchvision.utils.make_grid(image_n, normalize=True, padding=5)
    im_n_1 = np.transpose(im_n.cpu(), (1, 2, 0))
    plt.imshow(im_n_1)
    plt.show()

