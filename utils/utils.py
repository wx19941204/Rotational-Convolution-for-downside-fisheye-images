# camera-ready

import torch
import torch.nn as nn

import numpy as np

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

# function for colorizing a label image:
def label_img_to_color(img):
    label_to_color = {
        0: [255, 255, 255],
        1: [0, 0, 255],
        2: [255, 255, 0],
        3: [255, 0, 255],
        4: [0, 255, 255],
        5: [255, 204, 204],
        6: [204, 255, 204],
        7: [128, 128, 128],
        8: [102, 102, 102],
        9: [128, 128, 102],
        10: [128, 102, 128],
        11: [102, 128, 128],
        12: [128, 102, 0],
        13: [102, 0, 128],
        14: [  0, 128, 102],
        15: [  128, 0, 0],
        16: [  0, 128, 0]
        }

    # [255, 255, 255],
    # [0, 0, 255],
    # [255, 255, 0],
    # [255, 0, 255],
    # [0, 255, 255],
    # [255, 204, 204],
    # [204, 255, 204],
    # [128, 128, 128],
    # [102, 102, 102],
    # [128, 128, 102],
    # [128, 102, 128],
    # [102, 128, 128],
    # [128, 102, 0],
    # [102, 0, 128],
    # [0, 128, 102],
    # [128, 0, 0],
    # [0, 128, 0]]

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    # for row in range(img_height):
    #     for col in range(img_width):
    #         label = img[row, col]
    #
    #         img_color[row, col] = np.array(label_to_color[label])
    for i in range(len(label_to_color)):
        mask = (img == i)
        img_color[mask] = label_to_color[i]

    return img_color
