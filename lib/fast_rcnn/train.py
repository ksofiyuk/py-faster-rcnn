# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import pickle

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self._score_std = 0
        self._score_mean = 0
        self._so_force_iter = 0
        self._so_force_mode = False
        self._so_bad_samples = set()

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print('Computing bounding-box regression targets...')
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print('done')

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print(('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model))
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self._data_layer = self.solver.net.layers[0]
        self._data_layer.set_roidb(roidb)
        self._data_layer.net = self.solver.net

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             'bbox_pred' in net.params)

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        iters_info = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.net.layers[0].next_blob()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

            iters_losses = self.solver.net.layers[0].get_losses()
            if not self._so_force_mode:
                iters_info += iters_losses

            if self.solver.iter % 100 == 0:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                save_path = os.path.join(self.output_dir, 'iters_info.pickle')
                with open(save_path, 'wb') as f:
                    pickle.dump(iters_info, f)

            if cfg.TRAIN.ENABLE_SMART_ORDER:
                tail_len = cfg.TRAIN.SO_TAIL_LEN
                cls_scores = np.array([x[1][0] for x in iters_info])
                self._score_std = np.std(cls_scores[-tail_len:])
                self._score_mean = np.mean(cls_scores[-tail_len:])

                if not self._so_force_mode:
                    # if len(iters_losses) >= tail_len:
                    for sample, (cls_loss, bbox_loss) in iters_losses:
                        if cls_loss > self._score_mean + 1 * self._score_std:
                            self._so_bad_samples.add(sample)

                    self._data_layer.update_roidb_losses(iters_losses)
                    self._data_layer._score_mean = self._score_mean

                    if len(self._so_bad_samples) > cfg.TRAIN.SO_FORCE_BATCHSIZE:
                        self._so_force_mode = True
                        self._so_force_iter = 0
                        self._data_layer.enable_force_mode(self._so_bad_samples)

                    if self.solver.iter % 100 == 0:
                        print('bad sample cnt', len(self._so_bad_samples),
                                                self._score_mean, self._score_std)
                else:
                    self._so_force_iter += len(iters_losses)

                    max_force_iter = len(self._so_bad_samples) * cfg.TRAIN.SO_FORCE_ROUNDS
                    if self._so_force_iter > max_force_iter:
                        self._so_force_mode = False
                        self._so_bad_samples = set()
                        self._data_layer.disable_force_mode()


        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    return imdb.roidb


def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=None):
    if max_iters is None:
        max_iters = cfg.TRAIN.MAX_ITERS
    if pretrained_model is None:
        pretrained_model = cfg.WEIGHTS_PATH

    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print('Solving...')
    model_paths = sw.train_model(max_iters)
    print('done solving')
    return model_paths
