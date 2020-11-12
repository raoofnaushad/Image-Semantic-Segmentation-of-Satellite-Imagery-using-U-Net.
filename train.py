import os
import numpy as np
import glob
import shutil

from fastai.vision.all import *

import config


## Path for data loader to train.
path = Path(config.DATA_PATH)
## Codes are the classes required.
codes = np.array(["building", "woodland", "water", "Background"])


## get_image_files load data from file paths
fnames = get_image_files(path/"images")
## We also need a function to return the label name from the folder.
def label_func(fn): return path/"labels"/f"{fn.stem}_m{'.png'}"


## SegmentationDataLoaders is used as the data loader.
dls = SegmentationDataLoaders.from_label_func(
    path, bs=config.BATCH_SIZE, fnames = fnames, label_func = label_func, codes = codes
)
## Showing batches of data (Batch size = 3)
dls.show_batch()

## Since it is image segmentation we use dataloader and resnet18 which is a widely used transfer leanring model.
learn = unet_learner(dls, resnet18)
learn.fine_tune(3)

## For visualizing results
learn.show_results( )
