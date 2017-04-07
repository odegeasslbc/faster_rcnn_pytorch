# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import numpy as np

from .sealion import sealion

from .pascal_voc import pascal_voc
from .imagenet3d import imagenet3d
from .kitti import kitti
from .kitti_tracking import kitti_tracking
from .nthu import nthu
from .coco import coco
from .kittivoc import kittivoc



#imageset = 'SeaLionTrain'
#devkit = '/home/bingchen/kaggle/sea-lion/Train/'

def get_imdb(name, devkit):
    """Get an imdb (image database) by name."""
    __sets[name] = (lambda imageset = name, devkit_path = devkit: sealion(imageset, devkit_path))
    if not __sets.has_key(name):
        raise keyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
