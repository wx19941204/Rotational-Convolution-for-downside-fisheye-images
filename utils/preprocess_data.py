# camera-ready

import pickle
import numpy as np
import cv2
import os
from collections import namedtuple

# (NOTE! this is taken from the official Cityscapes scripts:)
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is taken from the official Cityscapes scripts:)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'            ,  0 ,      0 , 'void'            , 0       , False        , True         , (255, 255, 255) ),
    Label(  'Person'                ,  1 ,      1 , 'void'            , 0       , False        , True         , (0, 0, 255) ),
    Label(  'Door'                  ,  2 ,      2 , 'void'            , 0       , False        , True         , (255, 255, 0) ),
    Label(  'Chair'                 ,  3 ,      3 , 'void'            , 0       , False        , True         , (255, 0, 255) ),
    Label(  'Wall'                  ,  4 ,      4 , 'void'            , 0       , False        , True         , (0, 255, 255) ),
    Label(  'Floor'                 ,  5 ,      5 , 'void'            , 0       , False        , True         , (255, 204, 204) ),
    Label(  'Table'                 ,  6 ,      6 , 'void'            , 0       , False        , True         , (204, 255, 204) ),
    Label(  'Sofa'                  ,  7 ,      7 , 'flat'            , 1       , False        , False        , (128, 128, 128) ),
    Label(  'Furniture'             ,  8 ,      8 , 'flat'            , 1       , False        , False        , (102, 102, 102) ),
    Label(  'Lamp'                  ,  9 ,      9 , 'flat'            , 1       , False        , True         , (128, 128, 102) ),
    Label(  'Decoration'            , 10 ,      10 , 'flat'            , 1       , False        , True         , (128, 102, 128) ),
    Label(  'Plant'                 , 11 ,      11 , 'construction'    , 2       , False        , False        , (102, 128, 128) ),
    Label(  'Screen'                , 12 ,      12 , 'construction'    , 2       , False        , False        , (128, 102, 0) ),
    Label(  'Bed'                   , 13 ,      13 , 'construction'    , 2       , False        , False        , (102, 0, 128) ),
    Label(  'Fridge'                , 14 ,      14 , 'construction'    , 2       , False        , True         , (0, 128, 102) ),
    Label(  'Wheeled Walker'        , 15 ,      15 , 'construction'    , 2       , False        , True         , (128, 0, 0) ),
    Label(  'Armchair'              , 16 ,      16 , 'construction'    , 2       , False        , True         , (0, 128, 0) )
]

# create a function which maps id to trainId:
id_to_trainId = {label.id: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

train_meta_path = "../data/sample_label_imgs/"

path_list = os.listdir(train_meta_path)
path_list = [train_meta_path + x for x in path_list]

################################################################################
# compute the class weigths:
################################################################################
print ("computing class weights")

num_classes = 17

trainId_to_count = {}
for trainId in range(num_classes):
    trainId_to_count[trainId] = 0

# get the total number of pixels in all train label_imgs that are of each object class:
for step, label_img_path in enumerate(path_list):
    if step % 100 == 0:
        print(step)

    label_img = cv2.imread(label_img_path, -1)

    for trainId in range(num_classes):
        # count how many pixels in label_img which are of object class trainId:
        trainId_mask = np.equal(label_img, trainId)
        trainId_count = np.sum(trainId_mask)

        # add to the total count:
        trainId_to_count[trainId] += trainId_count
    if step > 1000:
        break

# compute the class weights according to the ENet paper:
class_weights = []
total_count = sum(trainId_to_count.values())
print('total_count', total_count)
for trainId, count in trainId_to_count.items():
    print('trainid:{}, count:{}'.format(trainId, count))
    trainId_prob = float(count)/float(total_count)
    trainId_weight = 1/np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)

print(class_weights)

# with open("../data/class_weights.pkl", "wb") as file:
#     pickle.dump(class_weights, file) # (protocol=2 is needed to be able to open this file with python2)
