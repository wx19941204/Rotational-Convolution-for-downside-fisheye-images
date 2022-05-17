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

    # trainId_to_id = {
    #     0: 0,
    #     1: 1,
    #     2: 2,
    #     3: 3,
    #     4: 4,
    #     5: 5,
    #     6: 6,
    #     7: 7,
    #     8: 8,
    #     9: 9,
    #     10: 10,
    #     11: 11,
    #     12: 12,
    #     13: 13,
    #     14: 14,
    #     15: 15,
    #     16: 16,
    # }
    # trainId_to_id_map_func = np.vectorize(trainId_to_id.get)

    batch_size = 4
    model_dir = 'runs/training_logs/model_eval_val_for_metrics'

    # network = unet.UNet(n_channels=3, n_classes=17).cuda()
    network = unet.RotUNet(n_channels=3, n_classes=17).cuda()
    # print(network)
    network.load_state_dict(torch.load("runs/training_logs/model_1/checkpoints/model_1_epoch_40.pth"))

    val_dataset = DatasetVal(val_data_path="data",
                             val_meta_path="data")

    num_val_batches = int(len(val_dataset) / batch_size)
    print("num_val_batches:", num_val_batches)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=1)

    # with open("data/class_weights.pkl", "rb") as file:  # (needed for python3)
    #     class_weights = np.array(pickle.load(file))
    # class_weights = torch.from_numpy(class_weights)
    # class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()
    #
    # # loss function
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
    # batch_losses = []
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
            outputs = F.upsample(outputs, size=(1024, 1024),
                                 mode="bilinear")  # (shape: (batch_size, num_classes, 1024, 2048))

            outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, 1024, 2048))
            pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, 1024, 2048))
            pred_label_imgs = pred_label_imgs.astype(np.uint8)

            for i in range(pred_label_imgs.shape[0]):
                pred_label_img = pred_label_imgs[i]  # (shape: (1024, 2048))
                img_id = img_ids[i]

                # convert pred_label_img from trainId to id pixel values:
                # pred_label_img = trainId_to_id_map_func(pred_label_img)  # (shape: (1024, 2048))
                pred_label_img = pred_label_img.astype(np.uint8)

                cv2.imwrite(model_dir + "/" + img_id + "_layer.png", pred_label_img)


    # val_loss = np.mean(batch_losses)
    # print("val loss: %g" % val_loss)
