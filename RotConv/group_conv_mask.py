import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time

'''
Generate pre-calculated coefficient matrix for weighted bitwise summation.
Mask visualization is in main function.
'''

# mask for RC4
def group_conv_mask(input_size=416):
    h = input_size
    half_h = h // 2
    mask0_0 = np.zeros((half_h, half_h))
    mask0_0[:, 0] = 1
    for i in range(0, half_h-1):
        mask0_0[i, :half_h-i] = np.linspace(1, 0.5, num=half_h-i)
    for j in range(1, half_h):
        mask0_0[(half_h-j-1):half_h, j] = np.linspace(0.5, 0, num=j+1)
    mask0_0[half_h-1, 0] = 0.5
    mask0_1 = np.flip(mask0_0, axis=1)
    mask0_2 = np.hstack((mask0_1, mask0_0))
    mask_0 = np.zeros_like(mask0_2)
    mask_0 = np.vstack((mask0_2, mask_0))

    mask_1 = np.rot90(mask_0)
    mask_2 = np.rot90(mask_1)
    mask_3 = np.rot90(mask_2)

    mask = [mask_0, mask_1, mask_2, mask_3]
    return mask

# mask for RC8
def group_conv_8mask(input_size=416):
    h = input_size
    half_h = h // 2

    # 0, 90, 180, 270
    mask0_0 = np.zeros((half_h, half_h))
    for i in range(0, half_h-1):
        mask0_0[i, :half_h-i] = np.linspace(1, 0, num=half_h-i)
    mask0_0[half_h-1, 0] = 0.5
    mask0_1 = np.flip(mask0_0, axis=1)
    mask0_2 = np.hstack((mask0_1, mask0_0))
    mask_0 = np.zeros_like(mask0_2)
    mask_0 = np.vstack((mask0_2, mask_0))

    # 45， 135， 225， 315
    mask7_0 = np.zeros((half_h, half_h))
    for i in range(0, half_h-1):
        mask7_0[i, :half_h-i] = np.linspace(0, 1, num=half_h-i)
    for j in range(1, half_h):
        mask7_0[(half_h-j-1):half_h, j] = np.linspace(1, 0, num=j+1)
    mask7_1 = np.zeros_like(mask7_0)
    mask7_2 = np.hstack((mask7_1, mask7_0))
    mask_7 = np.zeros_like(mask7_2)
    mask_7 = np.vstack((mask7_2, mask_7))

    mask_1 = np.rot90(mask_0)
    mask_2 = np.rot90(mask_1)
    mask_3 = np.rot90(mask_2)

    mask_4 = np.rot90(mask_7)
    mask_5 = np.rot90(mask_4)
    mask_6 = np.rot90(mask_5)

    # 0, 90, 180, 270，45， 135， 225， 315
    mask = [mask_0, mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7]
    return mask

# mask for RC2
def group_conv_2mask(input_size=416):
    h = input_size
    half_h = h // 2
    mask0_0 = np.zeros((half_h, half_h))
    mask0_0[:, 0] = 1
    for i in range(0, half_h-1):
        mask0_0[i, :half_h-i] = np.linspace(1, 0.75, num=half_h-i)
    for j in range(1, half_h):
        mask0_0[(half_h-j-1):half_h, j] = np.linspace(0.75, 0.5, num=j+1)
    mask0_0[half_h-1, 0] = 0.5
    mask0_1 = np.flip(mask0_0, axis=1)
    mask0_2 = np.hstack((mask0_1, mask0_0))
    mask0_3 = np.rot90(mask0_1) - 0.5
    mask0_3 = np.flip(mask0_3, axis=1)
    mask0_4 = np.flip(mask0_3, axis=1)
    mask_0 = np.hstack((mask0_4, mask0_3))
    mask_0 = np.vstack((mask0_2, mask_0))

    mask_1 = np.flip(mask_0)
    mask = [mask_0, mask_1]
    return mask


if __name__ == '__main__':

    # mask = group_conv_mask(input_size=6)
    # mask = torch.tensor(mask)
    # print(mask[0])
    #
    # input = torch.zeros(size=[5, 3, 4, 6, 6])
    # output = input + mask
    # output = output * mask
    # print(output[4, 2, 3])

    #### mask visualization
    mask = group_conv_mask(input_size=608)
    # mask = group_conv_2mask(input_size=608)
    # mask = group_conv_8mask(input_size=608)
    for (i, maski) in enumerate(mask):
        cv2.imshow(str(i), maski)
    cv2.waitKey(0)
