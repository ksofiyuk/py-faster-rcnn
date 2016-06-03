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

__sets['faces_aflw_train'] = lambda: FacesDataset('/home/local/work/data/faces/AFLW_p3lr11', 'train')
__sets['faces_aflw_test'] = lambda: FacesDataset('/home/local/work/data/faces/AFLW_p3lr11', 'test')

__sets['faces_aflw_fixed_train'] = lambda: FacesDataset('/home/local/work/data/faces/AFLWfixed_p3lr11', 'train')
__sets['faces_aflw_fixed_test'] = lambda: FacesDataset('/home/local/work/data/faces/AFLWfixed_p3lr11', 'test')

__sets['faces_fddb_train'] = lambda: FacesDataset('/home/local/work/data/faces/FDDB/FDDB_p3lr11', 'train')
__sets['faces_fddb_test'] = lambda: FacesDataset('/home/local/work/data/faces/FDDB/FDDB_p3lr11', 'test')

__sets['faces_afw_train'] = lambda: FacesDataset('/home/local/work/data/faces/AFW/AFW_p3lr11', 'train')
__sets['faces_afw_test'] = lambda: FacesDataset('/home/local/work/data/faces/AFW/AFW_p3lr11', 'test')

__sets['faces_ijbai_train'] = lambda: FacesDataset('/home/local/work/data/faces/IJBAI_p3lr11', 'train')
__sets['faces_ijbai_test'] = lambda: FacesDataset('/home/local/work/data/faces/IJBAI_p3lr11', 'test')

__sets['faces_bstest_train'] = lambda: FacesDataset('/home/local/work/data/faces/BSTest_p3lr11', 'train')
__sets['faces_bstest_test'] = lambda: FacesDataset('/home/local/work/data/faces/BSTest_p3lr11', 'test')

__sets['faces_bsfntest_train'] = lambda: FacesDataset('/home/local/work/data/faces/BigSampleFNTest_p3lr11', 'train')
__sets['faces_bsfntest_test'] = lambda: FacesDataset('/home/local/work/data/faces/BigSampleFNTest_p3lr11', 'test')

__sets['faces_bsfntrain_train'] = lambda: FacesDataset('/home/local/work/data/faces/BigSampleFNTrain_p3lr11', 'train')
__sets['faces_bsfntrain_test'] = lambda: FacesDataset('/home/local/work/data/faces/BigSampleFNTrain_p3lr11', 'test')

__sets['faces_yfs_train'] = lambda: FacesDataset('/home/local/work/data/faces/YFS/YFS_p3lr11', 'train')
__sets['faces_yfs_test'] = lambda: FacesDataset('/home/local/work/data/faces/YFS/YFS_p3lr11', 'test')

__sets['faces_videoset_train'] = lambda: FacesDataset('/home/local/work/data/faces/VideoSet1/VideoSet1_p3lr11', 'train')
__sets['faces_videoset_test'] = lambda: FacesDataset('/home/local/work/data/faces/VideoSet1/VideoSet1_p3lr11', 'test')

__sets['faces_itv_train'] = lambda: FacesDataset('/home/local/work/data/faces/ITV_mips2016_p3lr11', 'train')
__sets['faces_itv_test'] = lambda: FacesDataset('/home/local/work/data/faces/ITV_mips2016_p3lr11', 'test')

__sets['faces_aflw_ext_train'] = lambda: FacesDataset('/home/local/work/data/faces/AFLW_extended', 'train')
__sets['faces_aflw_ext_test'] = lambda: FacesDataset('/home/local/work/data/faces/AFLW_extended', 'test')

__sets['nightclub_test'] = lambda: FacesDataset('/home/local/work/data/faces/nightclub_video', 'test')

__sets['heads_hh_train'] = lambda: FacesDataset('/home/local/work/data/heads/HH/train', 'train')
__sets['heads_hh_test'] = lambda: FacesDataset('/home/local/work/data/heads/HH/test', 'test')

__sets['heads_hh_val'] = lambda: FacesDataset('/home/local/work/data/heads/HH/val', 'test')
__sets['heads_original24test'] = lambda: FacesDataset('/home/local/work/data/heads/OriginalTest_24', 'test')
__sets['heads_tc'] = lambda: FacesDataset('/home/local/work/data/town_centre', 'test')



# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxesIJBAI
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
