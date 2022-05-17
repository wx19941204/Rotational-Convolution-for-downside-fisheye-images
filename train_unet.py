# camera-ready

import sys

from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from utils.utils import add_weight_decay

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import unet


import time, os

if __name__ == '__main__':

    # Save  config
    model_id = "1"
    project_dir = 'runs'
    logs_dir = project_dir + "/training_logs"
    model_dir = logs_dir + "/model_%s" % model_id
    checkpoints_dir = model_dir + "/checkpoints"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(checkpoints_dir)

    # Train config
    num_epochs = 150
    batch_size = 4
    min_val_loss = 0.1
    learning_rate = 0.0001
    multi_GPU = False

    # network = unet.UNet(n_channels=3, n_classes=17).cuda()
    network = unet.RotUNet(n_channels=3, n_classes=17).cuda()

   # initial learning rate
    pg0, pg1, pg2, pg_deform = [], [], [], []  # optimizer parameter groups
    for k, v in network.named_parameters():
        if v.requires_grad:
            if 'offset_conv' in k or 'mask_conv' in k:
                pg_deform.append(v)
            elif '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else
    optimizer = torch.optim.Adam(params=pg1, lr=learning_rate, weight_decay=0.0001)
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    optimizer.add_param_group({'params': pg_deform, 'lr': 0.1 * learning_rate})
    print('Optimizer groups: %g .bias, %g conv.weight, %g other, %g deform_params' % (
    len(pg2), len(pg1), len(pg0), len(pg_deform)))
    del pg0, pg1, pg2

    if multi_GPU:
        network = nn.DataParallel(network)

    train_dataset = DatasetTrain(train_data_path="data",
                                 train_meta_path="data")
    val_dataset = DatasetVal(val_data_path="data",
                             val_meta_path="data")

    num_train_batches = int(len(train_dataset) / batch_size)
    num_val_batches = int(len(val_dataset) / batch_size)
    print("num_train_batches:", num_train_batches)
    print("num_val_batches:", num_val_batches)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=1)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=1)

    loss_fn = nn.CrossEntropyLoss()

    epoch_losses_train = []
    epoch_losses_val = []
    for epoch in range(num_epochs):
        print("###########################")
        print("######## NEW EPOCH ########")
        print("###########################")
        print("epoch: %d/%d" % (epoch + 1, num_epochs))

        ############################################################################
        # train:
        ############################################################################
        network.train()  # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs) in enumerate(tqdm(train_loader)):
            # current_time = time.time()
            # print(step)

            imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad()  # (reset gradients)
            loss.backward()  # (compute gradients)
            optimizer.step()  # (perform optimization step)

            # print (time.time() - current_time)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % model_dir)
        plt.close(1)

        print("####")

        ############################################################################
        # val:
        ############################################################################
        network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
            with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))

                outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))

                # compute the loss:
                loss = loss_fn(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        with open("%s/epoch_losses_val.pkl" % model_dir, "wb") as file:
            pickle.dump(epoch_losses_val, file)
        print("val loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_val, "k^")
        plt.plot(epoch_losses_val, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.savefig("%s/epoch_losses_val.png" % model_dir)
        plt.close(1)

        # save the model weights to disk:
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoints_dir + "/model_" + model_id + "_epoch_" + str(epoch + 1) + ".pth"
            if multi_GPU:
                torch.save(network.module.state_dict(), checkpoint_path)
            else:
                torch.save(network.state_dict(), checkpoint_path)

        if epoch_loss < min_val_loss:
            min_val_loss = epoch_loss
            checkpoint_path = checkpoints_dir + "/model_" + model_id + "_val_loss_" + str(min_val_loss) + ".pth"
            if multi_GPU:
                torch.save(network.module.state_dict(), checkpoint_path)
            else:
                torch.save(network.state_dict(), checkpoint_path)

