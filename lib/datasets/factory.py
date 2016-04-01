# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.detections_base_roidb import CaltechPedestrians, TownCenterPedestrians, RTSDSigns, FacesDataset
import numpy as np

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                pascal_voc(split, year))

__sets['caltech10x_trainval'] = lambda: CaltechPedestrians('/home/local/work/caltech/caltech10x', 'train')
__sets['caltech10x_test'] = lambda: CaltechPedestrians('/home/local/work/caltech/caltech10x', 'test')

__sets['caltech1x_trainval'] = lambda: CaltechPedestrians('/home/local/work/caltech/caltech1x', 'train')
__sets['caltech1x_test'] = lambda: CaltechPedestrians('/home/local/work/caltech/caltech1x', 'test')

__sets['caltech_all_trainval'] = lambda: CaltechPedestrians('/home/local/work/caltech/caltech_all', 'train')
__sets['caltech_all_test'] = lambda: CaltechPedestrians('/home/local/work/caltech/caltech_all', 'test')

__sets['towncenter_train'] = lambda: TownCenterPedestrians('/home/local/work/data/town_centre', 'train')
__sets['towncenter_test'] = lambda: TownCenterPedestrians('/home/local/work/data/town_centre', 'test')

__sets['rtsd_train'] = lambda: RTSDSigns('/home/local/work/rtsd_signs/gt/gt_base_all.pickle', 'train')
__sets['rtsd_test'] = lambda: RTSDSigns('/home/local/work/rtsd_signs/gt/gt_base_all.pickle', 'test')

__sets['faces_aflw'] = lambda: FacesDataset('/home/local/work/data/faces/AFLW_p3lr11')

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
