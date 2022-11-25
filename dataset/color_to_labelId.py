import json
import argparse
import numpy as np
from PIL import Image
from os.path import join
import pandas as pd
import os
import cv2

def get_label_info(csv_path):
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label

with open(join('./DANNet-main/dataset/lists', 'info.json'), 'r') as fp:
    info = json.load(fp)
label_name = np.array(info['label'], dtype=np.str)
mapping = np.array(info['label2train'], dtype=np.int)
id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18, 0: 19}
path = '/mydatasets/ACDC/gt/train/'
csv_path = './DANNet-main/dataset/lists/class_dict.csv'
label = get_label_info(csv_path)
print(label)
for filename in os.listdir(path):
    image = cv2.imread(path + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for id, key in enumerate(label):
        equality = np.equal(image, label[key])
        class_map = np.all(equality, axis=-1)
        image[class_map] = [k for k, v in id_to_trainid.items() if v == id]
    # for k, v in id_to_trainid.items():
    #     image[image == v] = k
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gt = filename.rsplit('_', 1)[0]
    cv2.imwrite('/mydatasets/ACDC/gt/train_trainid/'+gt+'_labelIds.png', image)