# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
import random
from core.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from datasets.iterators import MultiRandomOrderIterator
from datasets.iterators import InfinityLoopIterator
from datasets.image_sample import FlippedImageSample


class RoIDataLayer(caffe.Layer):
    """ data layer used for training."""

    def _get_next_normal_minibatch(self):
        minibatch = []
        while len(minibatch) < cfg.TRAIN.IMS_PER_BATCH:
            sample = next(self._imdbs_iter)

            if cfg.TRAIN.USE_FLIPPED and np.random.randint(0, 2):
                sample = FlippedImageSample(sample)

            if cfg.TRAIN.ENABLE_SMART_ORDER:
                loss = self._roidb_losses.get(sample.id, 1e6)

                if loss < 0.5 * self._score_mean and \
                        np.random.uniform() < cfg.TRAIN.SO_GOOD_SKIP_PROB:
                    # print('Skipped', sample.id, 'last loss', loss, 'mean loss', self._score_mean)
                    continue

            minibatch.append(sample)

        return minibatch

    def _get_next_force_minibatch(self):
        minibatch = []
        while len(minibatch) < cfg.TRAIN.IMS_PER_BATCH:
            if self._force_cur >= len(self._forced_samples):
                self._force_cur = 0
                random.shuffle(self._forced_samples)

            minibatch.append(self._forced_samples[self._force_cur])
            self._force_cur += 1

        return minibatch

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        if self._force_mode:
            minibatch_db = self._get_next_force_minibatch()
        else:
            minibatch_db = self._get_next_normal_minibatch()

        return get_minibatch(minibatch_db, self._num_classes)

    def enable_force_mode(self, forced_samples):
        self._forced_samples = forced_samples

        self._force_cur = 0
        self._force_mode = True

        print('Force mode was enabled with %d samples' % len(self._forced_samples))

    def disable_force_mode(self):
        self._force_mode = False

    def set_images_dbs(self, images_dbs):
        """Set the roidb to be used by this layer during training."""
        self._imdbs = images_dbs
        self._imdbs_iter = InfinityLoopIterator(
            MultiRandomOrderIterator(self._imdbs))

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}
        self._forward_images = []
        self._losses = []
        self._blobs = None
        self._roidb_losses = {}
        self._score_mean = 0
        self._force_mode = False
        self._forced_samples = []
        self._force_cur = 0
        self._imdbs = None
        self._imdbs_iter = None

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 5)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1

            top[idx].reshape(1, 5)
            self._name_to_top_map['ignored_boxes'] = idx
            idx += 1

        print('RoiDataLayer: name_to_top:', self._name_to_top_map)
        assert len(top) == len(self._name_to_top_map)

    def next_blob(self):
        pass

    def get_losses(self):
        iters_count = len(self._forward_images)

        losses = self._losses[-iters_count+1:] + [self.get_last_loss()]
        ret = list(zip(self._forward_images, losses))

        self._forward_images = []
        self._losses = []
        return ret

    def get_last_loss(self):
        bbox_loss = float(self.net.blobs['rpn_loss_bbox'].data.copy())
        cls_loss = float(self.net.blobs['rpn_cls_loss'].data.copy())
        return (cls_loss, bbox_loss)

    def update_roidb_losses(self, loss_data):
        for sample, losses in loss_data:
            self._roidb_losses[sample.id] = losses[0]

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        self._blobs, samples = self._get_next_minibatch()
        self._forward_images += samples

        # for sample in samples:
        #     print(sample.id, len(sample.marking), sample.max_size)

        self._losses.append(self.get_last_loss())

        for blob_name, blob in self._blobs.items():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass