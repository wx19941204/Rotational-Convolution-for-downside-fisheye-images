# camera-ready

import sys

from datasets import DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

from utils.utils import label_img_to_color

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import unet

if __name__ == '__main__':

    batch_size = 16

    network = network_plus.modeling.deeplabv3_resnet18(num_classes=17).cuda()
    # network.load_state_dict(torch.load("pretrained_models/model_13_2_2_2_epoch_580.pth"))
    network.load_state_dict(torch.load("runs/training_logs/rotdeeplabv3_87.73/checkpoints/model_22_val_loss_0.037069093.pth"))

    val_dataset = DatasetVal(val_data_path="data",
                             val_meta_path="data")

    num_val_batches = int(len(val_dataset) / batch_size)
    print("num_val_batches:", num_val_batches)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=1)

    # with open("data/meta/class_weights.pkl", "rb") as file:  # (needed for python3)
    #     class_weights = np.array(pickle.load(file))
    # class_weights = torch.from_numpy(class_weights)
    # class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

    # loss function
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs, img_ids) in enumerate(tqdm(val_loader)):
        with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            # loss = loss_fn(outputs, label_imgs)
            # loss_value = loss.data.cpu().numpy()
            # batch_losses.append(loss_value)

            ########################################################################
            # save data for visualization:
            ########################################################################
            outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
            pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
            pred_label_imgs = pred_label_imgs.astype(np.uint8)

            for i in range(pred_label_imgs.shape[0]):
                pred_label_img = pred_label_imgs[i]  # (shape: (img_h, img_w))
                img_id = img_ids[i]
                # img = imgs[i]  # (shape: (3, img_h, img_w))
                #
                # img = img.data.cpu().numpy()
                # img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
                # img = img * np.array([0.229, 0.224, 0.225])
                # img = img + np.array([0.485, 0.456, 0.406])
                # img = img * 255.0
                # img = img.astype(np.uint8)
                #
                pred_label_img_color = label_img_to_color(pred_label_img)
                pred_label_img_color = pred_label_img_color[:, :, ::-1]  # BGR2RGB
                # overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
                # overlayed_img = overlayed_img.astype(np.uint8)

                cv2.imwrite("./pred_color_imgs/" + img_id + "_color.png", pred_label_img_color)

    #
    # val_loss = np.mean(batch_losses)
    # print("val loss: %g" % val_loss)
