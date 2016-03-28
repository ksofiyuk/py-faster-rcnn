import os
from datasets.imdb import imdb
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import random
import subprocess
import uuid
from .voc_eval import voc_eval
from fast_rcnn.config import cfg
from copy import deepcopy

from ground_truth import DetectionsBase
from extract_positives import extract_positives_gt
from collect_negatives import collect_random_negatives_from_gt
from augmentation import jitter_sample
from filesystem_utils import mkdir
import cv2


class ImageLoader:
    def __init__(self, image_path, image=None):
        self.path = image_path
        self.image = image

    def __call__(self):
        if self.image is None:
            return cv2.imread(self.path)
        else:
            return self.image


class caltech_pedestrians(imdb):

    def __init__(self, dataset_path, split, include_backgrounds=False, town_center=False):
        imdb.__init__(self, os.path.basename(dataset_path) + '_' + split)

        self._classes = ['__background__', 'pedestrian']
        self._image_ext = '.jpg'
        self._generated_cnt = 0

        self._image_index = []

        filter_conditions = [('height', 50, 6000), ('width', 10, 5000),
                             ('sign_class', '(?!((person\?)|(people)))', None),
                     ('overlapped', 0, 1), ('score', 0, 0.35)]

        if town_center:
            filter_conditions = [('sign_class', '(?!((person\?)|(people)))', None),
                                 ('beyond_borders', 0, 0)]

        self._gt_base = DetectionsBase(os.path.join(dataset_path, split), True)
        self._gt_base.set_filter_conditions(filter_conditions)

        self._backgrounds = self._gt_base.get_backgrounds(pure_backgrounds=True)
        self._backgrounds = [ip for ip, _ in self._backgrounds.items()]

        gt = self._gt_base.get_gt(keep_ignore=True,
                                  include_backgrounds=include_backgrounds)

        self._positives = []
        self._negatives = []
        if cfg.TRAIN.GENERATED_FRACTION > 0:
            self._positives = extract_positives_gt(gt, 80, 6, sratio=1.5)
            self._negatives = collect_random_negatives_from_gt(gt, 30000, 10, (80, 120))

        self._gt_list = sorted(list(gt.items()))
        self._image_index = list(range(len(self._gt_list)))

    def get_roidb_element(self, bboxes, filepath, image_getter=None):
        is_background = (len(bboxes) == 0 or all(box[4] for box in bboxes))
        num_objs = len(bboxes)

        if is_background:
            num_objs += 1
            bboxes.append([0,0,1,1,False])

        overlaps = np.zeros((len(bboxes), self.num_classes), dtype=np.float32)
        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        # assert num_objs > 0

        for ix, bbox in enumerate(bboxes):
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + max(2, bbox[2]) - 1
            y2 = y1 + max(2, bbox[3]) - 1
            gt_classes[ix] = 1

            assert(x2 > x1 and y2 > y1)

            if bbox[4]:
                y2 *= -1

            boxes[ix, :] = [x1, y1, x2, y2]
            overlaps[ix, 1] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        if image_getter is None:
            image_getter = ImageLoader(filepath)

        return {'image_getter': image_getter,
                         'image' : filepath,
                         'boxes' : boxes,
                         'gt_classes': gt_classes,
                         'gt_overlaps' : overlaps,
                         'flipped' : False,
                         'background': is_background,
                         'max_overlaps': overlaps.toarray().max(axis=1),
                         'max_classes': overlaps.toarray().argmax(axis=1)}

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = pickle.load(fid)
        #     print('{} gt roidb loaded from {}'.format(self.name, cache_file))
        #     return roidb

        gt_roidb = []
        for index, (filepath, bboxes) in enumerate(self._gt_list):
            gt_roidb.append(self.get_roidb_element(bboxes, filepath))

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def generate_frame(self):
        num_pos = 10
        num_neg = int(num_pos * 0.8)
        samples = random.sample(self._positives, num_pos)

        samples += [(x, []) for x in random.sample(self._negatives, num_neg)]
        samples = [jitter_sample(sample, jittering_angle=6,
                                         jittering_contrast=0.2,
                                         jittering_illumination=0.2)
                       for sample in samples]

        avg_height_k = sum(min(b[3] for b in boxes if not b[4])/i.shape[0]
                               for i, boxes in samples[:num_pos]) / num_pos
        samples[:num_pos] = [random_scale_sample(sample, 50, 90)
                                for sample in samples[:num_pos]]
        samples[num_pos:] = [random_scale_sample(sample, 50, 90, avg_height_k)
                                for sample in samples[num_pos:]]
        random.shuffle(samples)

        background = cv2.imread(random.choice(self._backgrounds))
        image, boxes = combine_samples(samples, 640, 480, background)

        background = cv2.imread(random.choice(self._backgrounds))
        image = np.hstack((image, background))

        image_path = './generated/%d.jpg' % self._generated_cnt
        roidb_element = self.get_roidb_element(boxes, image_path,
                                               ImageLoader(image_path, image))

        mkdir('./generated')
        cv2.imwrite(image_path, image)
        self._generated_cnt += 1

        return roidb_element

    def image_path_at(self, i):
        return self._gt_list[self._image_index[i]][0]


def combine_horizontal(samples, out_width):
    line_height = max(i.shape[0] for i, b in samples)
    sum_width = sum(i.shape[1] for i, b in samples)
    max_shift = np.round(2 * (out_width - sum_width) / len(samples))
    res_width = out_width - sum_width

    shifts = []
    last_x = 0
    for image, _ in samples:
        h_shift = np.random.randint(0, line_height - image.shape[0] + 1)
        w_shift = np.random.randint(0, min(max_shift, res_width) + 1)
        shifts.append((last_x + w_shift, h_shift))
        last_x += w_shift + image.shape[1]
        res_width -= w_shift
    return shifts


def combine_samples(samples, out_width, out_height, background=None):
    heights = [i.shape[0] for i, b in samples]
    order = np.argsort(heights)[::-1]

    horizontal_boxes = []
    line_boxes = []

    total_width = 0
    total_height = 0
    max_height = 0
    for indx in order:
        width = samples[indx][0].shape[1]
        height = samples[indx][0].shape[0]

        if total_height + max_height > out_height:
            break

        if total_width + width > out_width:
            horizontal_boxes.append(line_boxes)
            total_width = width
            total_height += max_height
            max_height = height
            line_boxes = [samples[indx]]
        else:
            max_height = max(height, max_height)
            total_width += width
            line_boxes.append(samples[indx])

    if total_height + max_height <= out_height:
        horizontal_boxes.append(line_boxes)

    random.shuffle(horizontal_boxes)
    for line in horizontal_boxes:
        random.shuffle(line)

    last_h = 0
    if background is None:
        out_image = np.zeros((out_height, out_width, 3),
                             dtype=samples[0][0].dtype)
    else:
        out_image = background.copy()

    gt_boxes = []
    for line in horizontal_boxes:
        shifts = combine_horizontal(line, out_width)
        line_height = 0

        for indx, (image, boxes) in enumerate(line):
            sx = shifts[indx][0]
            sy = last_h + shifts[indx][1]

            out_image[sy:sy + image.shape[0],
                      sx:sx + image.shape[1], :] = image
            for box in boxes:
                box = box.copy()
                box[0] += sx
                box[1] += sy
                gt_boxes.append(box)

            line_height = max(line_height, image.shape[0])
        last_h += line_height

    return out_image, gt_boxes


def random_scale_sample(sample, min_height, max_height, height_mul=None):
    image = sample[0]
    boxes = deepcopy(sample[1])
    height = np.random.randint(min_height, max_height + 1)

    if not boxes:
        sheight = image.shape[0] * height_mul
    else:
        sheight = min(b[3] for b in boxes if not b[4])
    scale = height / sheight

    for box in boxes:
        for i in range(4):
            box[i] *= scale

    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return (image, boxes)