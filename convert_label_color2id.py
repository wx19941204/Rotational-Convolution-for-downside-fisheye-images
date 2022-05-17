import cv2
import os
import numpy as np
from tqdm import tqdm

color_map = np.asarray([
    [255, 255, 255], # 0
    [0, 0, 255],  # 1
    [255, 255, 0],  # 2
    [255, 0, 255],  # 3
    [0, 255, 255],  # 4
    [255, 204, 204],  # 5
    [204, 255, 204],  # 6
    [128, 128, 128],  # 7
    [102, 102, 102],  # 8
    [128, 128, 102],  # 9
    [128, 102, 128],  # 10
    [102, 128, 128],  # 11
    [128, 102, 0],  # 12
    [102, 0, 128],  # 13
    [0, 128, 102],  # 14
    [128, 0, 0],  # 15
    [0, 128, 0],  # 16
], dtype=np.uint8)

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


path_color_mark1 = "data/gtFine/"
path_save = "data/label_imgs/"

list_color = os.listdir(path_color_mark1)
for cnt, img_name in enumerate(tqdm(list_color)):
    # print(cnt, img_name)
    path_img = path_color_mark1 + img_name
    img_color = cv2.imread(path_img)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # , cv2.COLOR_BGR2RGB
    h, w, _ = img_color.shape

    len_color = len(color_map)
    gray_img = np.zeros([h, w], np.uint8)
    for i in range(len_color):
        mask = np.all(img_color == color_map[i], axis=2)
        label = i
        gray_img[mask] = label

    cv2.imwrite(path_save + img_name, gray_img)