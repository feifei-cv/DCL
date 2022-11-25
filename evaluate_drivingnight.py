import os
import numpy as np
import json
from os.path import join
import sys

from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
from utils import Logger

from network import *
from dataset.zurich_night_dataset import zurich_night_DataSet
from dataset.nighttimedriving import NighttimeDriving
from dataset.bdd100knight import BDD100k
from dataset.acdc_val import ACDC_val
from configs.test_config import get_arguments


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)+1e-7)

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def main(args, data_set):

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda")

    args.save = './result/' + data_set + '/DCL_PSPNet'
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'PSPNet':
        model = PSPNet(args, num_classes=args.num_classes)
    if args.model == 'DeepLab':
        model = Deeplab(args, num_classes=args.num_classes)
    if args.model == 'RefineNet':
        model = RefineNet(args, num_classes=args.num_classes, imagenet=False)
    saved_state_dict = torch.load(args.restore_from)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)
    model = model.to(device)
    model.eval()

    if data_set == 'zurich':
        testloader = data.DataLoader(zurich_night_DataSet(args.data_dir, args.data_list, set=args.set))
    elif data_set == 'nightdriving':
        DATA_DIRECTORY = '/disks/disk0/feifei/data/semantic_data/NighttimeDrivingTest/leftImg8bit'
        DATA_LIST_PATH = './dataset/lists/nightdriving_test.txt'
        testloader = data.DataLoader(NighttimeDriving(DATA_DIRECTORY, DATA_LIST_PATH, set=args.set))
    elif data_set == 'acdc':
        DATA_DIRECTORY = '/disks/disk0/feifei/data/semantic_data/ACDC/rgb_anon'
        DATA_LIST_PATH = '/disks/disk0/feifei/paper/paper5/dataset/lists/acdc_night_val.txt'
        testloader = data.DataLoader(ACDC_val(DATA_DIRECTORY, DATA_LIST_PATH, set=args.set))
    elif data_set == 'bdd100knight':
        DATA_DIRECTORY = '/disks/disk0/feifei/paper/paper5/dataset'
        DATA_LIST_PATH = './dataset/lists/bdd100k.txt'
        testloader = data.DataLoader(BDD100k(DATA_DIRECTORY, DATA_LIST_PATH, set=args.set))
    interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

    if data_set == 'bdd100knight':
        interp = nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=True)

    weights = torch.FloatTensor(
        [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
         0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
         0.00413907]).cuda()
    weights = torch.log(weights)
    weights = (torch.mean(weights) - weights) / torch.std(weights) * args.std + 1.0
    for index, batch in enumerate(testloader):
        image, name = batch
        image = image.to(device)
        with torch.no_grad():
            if args.model == 'RefineNet':
                output2, fe = model(image)
            else:
                _, output2, fe = model(image)
        weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
        weights_prob = weights_prob.transpose(1, 3)
        output2 = output2 * weights_prob
        output = interp(output2).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8) ## predict label
        output_col = colorize_mask(output)
        output = Image.fromarray(output)
        name = name[0].split('/')[-1]
        if data_set == 'bdd100knight':
            name = name.replace('jpg', 'png')
        output.save('%s/%s' % (args.save, name))

        args.save1 = args.save + str(1)
        if not os.path.exists(args.save1):
            os.makedirs(args.save1)
        output_col.save('%s/%s_color.png' % (args.save1, name.split('.')[0])) ## color mask


def compute_mIoU(data_set, gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)

    num_classes = int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=str)
    mapping = np.array(info['label2train'], dtype=int)
    hist = np.zeros((num_classes, num_classes))

    if data_set == 'zurich':
        image_path_list = join(devkit_dir, 'zurich_val.txt')
        label_path_list = join(devkit_dir, 'label_zurich.txt')
        gt_imgs = open(label_path_list, 'r').read().splitlines()
        gt_imgs = [join(gt_dir, x) for x in gt_imgs]
        pred_imgs = open(image_path_list, 'r').read().splitlines()
        pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    elif data_set == 'nightdriving':
        image_path_list = join(devkit_dir, 'nightdriving_test.txt')
        pred_imgs = open(image_path_list, 'r').read().splitlines()
        pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]
        gt_imgs = [join(gt_dir, x.split('/')[-1][:-15] + 'gtCoarse_labelIds.png') for x in pred_imgs]

    elif data_set == 'acdc':
        image_path_list = join(devkit_dir, 'acdc_night_val.txt')
        pred_imgs = open(image_path_list, 'r').read().splitlines()
        pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]
        gt_imgs = [join(gt_dir, x.split('/')[-1].split('_')[0], x.split('/')[-1].replace('rgb_anon',
                                                                                         'gt_labelIds')) for x in pred_imgs]

    elif data_set == 'bdd100knight':
        image_path_list = join(devkit_dir, 'bdd100k.txt')
        label_path_list = join(devkit_dir, 'label_bdd100k.txt')
        gt_imgs = open(label_path_list, 'r').read().splitlines()
        gt_imgs = [join(gt_dir, x) for x in gt_imgs]
        pred_imgs = open(image_path_list, 'r').read().splitlines()
        pred_imgs = [join(pred_dir, x.split('/')[-1].replace('.jpg','.png')) for x in pred_imgs]


    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        if data_set != 'bdd100knight':
            label = label_mapping(label, mapping)
        assert len(label.flatten()) == len(pred.flatten())
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs




if __name__ == '__main__':

    args = get_arguments()
    args.restore_from = '/disks/disk0/feifei/paper/paper5/snapshots/PSPNet--38.87-25000/DCL25000.pth'

    # ## Zurich val
    # main(args, data_set='zurich')
    # compute_mIoU('zurich', '/disks/disk0/feifei/data/semantic_data/Zurich/val/', './result/zurich/DCL_PSPNet', './dataset/lists')
    # #
    # # # nightdriving test
    # main(args, data_set='nightdriving')
    # compute_mIoU('nightdriving',
    #              '/disks/disk0/feifei/data/semantic_data/NighttimeDrivingTest/gtCoarse_daytime_trainvaltest/test/night',
    #                                                                 './result/nightdriving/DCL_PSPNet', './dataset/lists')
    #
    # ##bdd100k-night test
    # main(args, data_set='bdd100knight')
    # compute_mIoU('bdd100knight',
    #              '/disks/disk0/feifei/paper/paper5/dataset', './result/bdd100knight/DCL_PSPNet', './dataset/lists')

    ## ACDC val
    main(args, data_set='acdc')
    compute_mIoU('acdc',
                 '/disks/disk0/feifei/data/semantic_data/ACDC/gt/night/val', './result/acdc/DCL_PSPNet', './dataset/lists')