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
import unet
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

    batch_size = 4
    model_dir = 'runs/training_logs/model_eval_val_for_metrics'

    # network = unet.UNet(n_channels=3, n_classes=17).cuda()
    network = unet.RotUNet(n_channels=3, n_classes=17).cuda()

    network.load_state_dict(torch.load("runs/training_logs/model_1/checkpoints/model_1_epoch_40.pth"))

    val_dataset = DatasetVal(val_data_path="data",
                             val_meta_path="data")

    num_val_batches = int(len(val_dataset) / batch_size)
    print("num_val_batches:", num_val_batches)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=1)



    network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
    for step, (imgs, label_imgs, img_ids) in enumerate(tqdm(val_loader)):
        with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))

            ########################################################################
            # save data for visualization:
            ########################################################################
            outputs = F.upsample(outputs, size=(1024, 1024),
                                 mode="bilinear")

            outputs = outputs.data.cpu().numpy()
            pred_label_imgs = np.argmax(outputs, axis=1)
            pred_label_imgs = pred_label_imgs.astype(np.uint8)

            for i in range(pred_label_imgs.shape[0]):
                pred_label_img = pred_label_imgs[i]
                img_id = img_ids[i]
                pred_label_img = pred_label_img.astype(np.uint8)

                cv2.imwrite(model_dir + "/" + img_id + "_layer.png", pred_label_img)

