import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
import torchvision.transforms as standard_transforms



class BDD100k(data.Dataset):


    def __init__(self, root, list_path, max_iters=None, set='val'):

        self.root = root
        self.list_path = list_path

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        val_input_transform = standard_transforms.Compose([
            # standard_transforms.Resize((540, 960)),
            standard_transforms.Resize((360, 768)),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        self.transform = val_input_transform

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set

        for name in self.img_ids:
            img_file = osp.join(self.root, "%s" % name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        return self.transform(image), name


if __name__ == '__main__':

    DATA_DIRECTORY = '/disks/disk0/feifei/paper/paper5/dataset'
    DATA_LIST_PATH = '/disks/disk0/feifei/paper/paper5/dataset/lists/bdd100k.txt'
    testloader = data.DataLoader(BDD100k(DATA_DIRECTORY, DATA_LIST_PATH))

    trainloader_iter = enumerate(testloader)
    _, batch = trainloader_iter.__next__()
    image, name = batch

    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.utils

    im_d = torchvision.utils.make_grid(image, normalize=True, padding=5)
    im_d_1 = np.transpose(im_d.cpu(), (1, 2, 0))
    plt.imshow(im_d_1)
    plt.show()



