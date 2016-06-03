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
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue


class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_normal_minibatch(self):
        minibatch_db = []
        while len(minibatch_db) < cfg.TRAIN.IMS_PER_BATCH:
            if self._cur >= len(self._roidb):
                self._shuffle_roidb_inds()

            if np.random.uniform() < cfg.TRAIN.GENERATED_FRACTION:
                minibatch_db.append(self._imdb.generate_frame())
                continue

            inds = self._perm[self._cur]
            self._cur += 1

            if self._roidb[inds]['background']:
                x = np.random.uniform()
                if x > cfg.TRAIN.BG_PROB:
                    continue

            if cfg.TRAIN.USE_FLIPPED:
                self._roidb[inds]['flipped'] = np.random.randint(0, 2)

            if cfg.TRAIN.ENABLE_SMART_ORDER:
                sample = (self._roidb[inds]['image'], self._roidb[inds]['flipped'])
                loss = self._roidb_losses.get(sample, 1e6)

                if loss < 0.5 * self._score_mean and \
                         np.random.uniform() < cfg.TRAIN.SO_GOOD_SKIP_PROB:
                    # print('Skipped', sample, 'last loss', loss, 'mean loss', self._score_mean)
                    continue

            minibatch_db.append(self._roidb[inds])

        return minibatch_db

    def _get_next_force_minibatch(self):
        minibatch_db = []
        while len(minibatch_db) < cfg.TRAIN.IMS_PER_BATCH:
            if self._force_cur >= len(self._force_inds):
                self._force_cur = 0
                random.shuffle(self._force_inds)

            inds, is_flipped = self._force_inds[self._force_cur]
            self._force_cur += 1

            self._roidb[inds]['flipped'] = is_flipped
            minibatch_db.append(self._roidb[inds])

        return minibatch_db

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            assert False, "not implemented"
            return self._blob_queue.get()
        else:
            if self._force_mode:
                minibatch_db = self._get_next_force_minibatch()
            else:
                minibatch_db = self._get_next_normal_minibatch()

            return get_minibatch(minibatch_db, self._num_classes)

    def enable_force_mode(self, force_samples):
        force_samples = {path: flipped for path, flipped in list(force_samples)}

        self._force_cur = 0
        self._force_inds = []
        self._force_mode = True

        for indx, record in enumerate(self._roidb):
            flipped = force_samples.get(record['image'], None)
            if flipped is not None:
                self._force_inds.append((indx, flipped))
        print('Force mode was enabled with %d samples' % len(self._force_inds))

    def disable_force_mode(self):
        self._force_mode = False

    def set_roidb(self, imdb):
        """Set the roidb to be used by this layer during training."""
        self._imdb = imdb
        self._roidb = imdb.roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print('Terminating BlobFetcher')
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

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
        self._force_inds = []
        self._force_cur = 0

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

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
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
            self._roidb_losses[sample] = losses[0]

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        self._blobs, im_path = self._get_next_minibatch()
        self._forward_images.append(im_path)

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

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print('BlobFetcher started')
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)
