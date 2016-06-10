# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
from core.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(samples, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    assert cfg.TRAIN.HAS_RPN, "RPN only!"
    assert len(samples) == 1, "Single batch only"

    sample = samples[0]
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(sample.scales))

    # Get the input image blob, formatted for caffe
    im_blob, gt_boxes, ignored_boxes, im_scale = \
        _convert_sample(sample, random_scale_inds)

    blobs = {'data': im_blob}
    blobs['gt_boxes'] = gt_boxes
    blobs['ignored_boxes'] = ignored_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scale]],
        dtype=np.float32)

    return blobs, samples


def _convert_sample(sample, scale_indx):
    target_size = sample.scales[scale_indx]

    im, im_scale = prep_im_for_blob(sample.bgr_data, cfg.PIXEL_MEANS,
                                    target_size, sample.max_size)

    gt_boxes = []
    ignored_boxes = []
    for x in sample.marking:
        if x['object_class'] < 1:
            continue
        box = [x['x'], x['y'],
               x['x'] + x['w'] - 1, x['y'] + x['h'] - 1,
               x['object_class']]
        if x['ignore']:
            ignored_boxes.append(box)
        else:
            gt_boxes.append(box)

    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    ignored_boxes = np.array(ignored_boxes, dtype=np.float32)
    gt_boxes[0:4] *= im_scale
    ignored_boxes[0:4] *= im_scale

    blob = im_list_to_blob([im])

    return blob, gt_boxes, ignored_boxes, im_scale


# def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
#     """Generate a random sample of RoIs comprising foreground and background
#     examples.
#     """
#     # label = class RoI has max overlap with
#     labels = roidb['max_classes']
#     overlaps = roidb['max_overlaps']
#     rois = roidb['boxes']
#
#     # Select foreground RoIs as those with >= FG_THRESH overlap
#     fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
#     # Guard against the case when an image has fewer than fg_rois_per_image
#     # foreground RoIs
#     fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
#     # Sample foreground regions without replacement
#     if fg_inds.size > 0:
#         fg_inds = npr.choice(
#                 fg_inds, size=fg_rois_per_this_image, replace=False)
#
#     # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
#     bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
#                        (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
#     # Compute number of background RoIs to take from this image (guarding
#     # against there being fewer than desired)
#     bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
#     bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
#                                         bg_inds.size)
#     # Sample foreground regions without replacement
#     if bg_inds.size > 0:
#         bg_inds = npr.choice(
#                 bg_inds, size=bg_rois_per_this_image, replace=False)
#
#     # The indices that we're selecting (both fg and bg)
#     keep_inds = np.append(fg_inds, bg_inds)
#     # Select sampled values from various arrays:
#     labels = labels[keep_inds]
#     # Clamp labels for the background RoIs to 0
#     labels[fg_rois_per_this_image:] = 0
#     overlaps = overlaps[keep_inds]
#     rois = rois[keep_inds]
#
#     bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
#             roidb['bbox_targets'][keep_inds, :], num_classes)
#
#     return labels, overlaps, rois, bbox_targets, bbox_inside_weights


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
