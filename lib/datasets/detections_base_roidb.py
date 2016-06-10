import os
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import random
import bisect
import os.path as osp

from pprint import pprint
from collections import defaultdict
from core.config import cfg
from copy import deepcopy

from ground_truth import DetectionsBase
from extract_positives import extract_positives_gt
from collect_negatives import collect_random_negatives_from_gt
from augmentation import jitter_sample
from filesystem_utils import mkdir
from bbox_utils import get_max_iou
from lmdb_utils import LMDBStore
from visual_utils import plot_bboxes
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


class DetectionsBaseRoidb(imdb):

    def __init__(self, dataset_name):
        super(DetectionsBaseRoidb, self).__init__(dataset_name)

        self._generated_cnt = 0
        self._neg_indx = 0
        self._image_index = []

        self.init_params()

        self._pos_lmdb = None
        self._pos_prob = None
        self._pos_types = None
        self._pos_types_indxs = defaultdict(list)
        self._positives = []
        self._negatives = []
        self._gt = {}
        self._backgrounds = []

        self.gt_base_init()

        if cfg.TRAIN.USE_LMDB:
            lmdb_path = osp.abspath(osp.join(cfg.ROOT_DIR, 'exps',
                                             cfg.EXP_DIR, 'output',
                                             'lmdb_pos_' + dataset_name))
            lmdb_exists = osp.exists(lmdb_path)

            self._pos_lmdb = LMDBStore(lmdb_path)
            if not lmdb_exists:
                for i, sample in enumerate(self._positives):
                    self._pos_lmdb.set_value(str(i), sample)
                    self._pos_types_indxs[sample[1][0][5]].append(i)
                self._pos_lmdb.set_value('_pos_types_indxs', self._pos_types_indxs)
            else:
                self._pos_types_indxs = self._pos_lmdb.get_value('_pos_types_indxs')
            self._pos_lmdb.commit()
            self._num_positives = sum(len(x) for key, x in self._pos_types_indxs.items())
        else:
            self._positives = list(self._positives)
            self._num_positives = len(self._positives)
            for i, (_, boxes) in enumerate(self._positives):
                self._pos_types_indxs[boxes[0][5]].append(i)

        if cfg.TRAIN.REDISTRIBUTE_CLASSES:
            cnt_by_types = [(stype, len(indxs)) for stype, indxs in self._pos_types_indxs.items()]
            stypes, cnt = tuple(map(list, zip(*cnt_by_types)))
            prob = np.array(cnt, dtype=np.float) / np.sum(cnt)

            thresh = 0.1 / len(prob)
            while np.min(prob) < thresh:
                low_fr = (np.sum(prob < thresh) + 1) / len(prob)
                prob = redistribute(prob, low_fr, 0.05, 0.01)

            self._pos_prob = prob
            self._pos_types = stypes
            pprint(dict(zip(stypes, list(zip(cnt, prob)))))


        self._gt_list = sorted(list(self._gt.items()))
        self._image_index = list(range(len(self._gt_list)))

    def gt_base_init(self):
        pass

    def init_params(self):
        self.gen_num_pos = 1
        self.gen_num_neg = 1
        self.gen_width = 200
        self.gen_height = 200
        self.gen_min_height = 20
        self.gen_max_height = 80

    def get_roidb_element(self, bboxes, filepath, image_getter=None):
        is_background = (len(bboxes) == 0 or all(box[4] for box in bboxes))
        num_objs = len(bboxes)

        if is_background:
            num_objs += 1
            bboxes.append([0,0,1,1,False])

        overlaps = np.zeros((len(bboxes), self.num_classes), dtype=np.float32)
        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

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

        gt_roidb = []
        for index, (filepath, bboxes) in enumerate(self._gt_list):
            gt_roidb.append(self.get_roidb_element(bboxes, filepath))

        return gt_roidb

    def sample_jittering(self, sample):
        return sample

    def collect_negatives(self):
        return [None]

    def generate_frame(self):
        num_pos = self.gen_num_pos
        num_neg = self.gen_num_neg
        samples_indx = []

        if cfg.TRAIN.REDISTRIBUTE_CLASSES:
            samples_types = prob_sample(self._pos_types, self._pos_prob, num_pos)
            for stype in samples_types:
                samples_indx.append(random.choice(self._pos_types_indxs[stype]))
        else:
            samples_indx = random.sample(range(self._num_positives), num_pos)

        if cfg.TRAIN.USE_LMDB:
            samples = [self._pos_lmdb.get_value(str(i)) for i in samples_indx]
        else:
            samples = [self._positives[i] for i in samples_indx]

        neg_samples = []
        while len(neg_samples) < num_neg:
            if self._neg_indx >= len(self._negatives):
                self._negatives = self.collect_negatives()
                self._neg_indx = 0
            neg_samples.append((self._negatives[self._neg_indx], []))
            self._neg_indx += 1

        samples += neg_samples
        samples = [self.sample_jittering(sample) for sample in samples]

        avg_height_k = sum(min(b[3] for b in boxes if not b[4])/i.shape[0]
                               for i, boxes in samples[:num_pos]) / num_pos
        samples[:num_pos] = [random_scale_sample(sample, self.gen_min_height, self.gen_max_height)
                                for sample in samples[:num_pos]]
        samples[num_pos:] = [random_scale_sample(sample, self.gen_min_height, self.gen_max_height,
                                                 avg_height_k)
                                for sample in samples[num_pos:]]
        random.shuffle(samples)

        if self._backgrounds is not None:
            background_path, background_boxes = random.choice(self._backgrounds)

            background = cv2.imread(background_path)
            if background.shape[1] > self.gen_width:
                background = background[:, :self.gen_width, :]

            scale_x = self.gen_width / background.shape[1]
            scale_y = self.gen_height / background.shape[0]
            background = cv2.resize(background, (self.gen_width, self.gen_height))

            image, boxes, inserted_boxes = combine_samples(samples, self.gen_width, self.gen_height, background)

            for box in background_boxes:
                if box[0] + box[2] * 0.5 > self.gen_width:
                    continue

                box[0] *= scale_x
                box[1] *= scale_y
                box[2] *= scale_x
                box[3] *= scale_y
                iou = get_max_iou(box, inserted_boxes)
                if iou < 0.05:
                    boxes.append(box)
        else:
            sample_path, sample_boxes = random.choice(self._gt_list)
            sample_image = cv2.imread(sample_path)

            background = np.full((sample_image.shape[0], self.gen_width, 3), 128, dtype=sample_image.dtype)
            image, boxes, inserted_boxes = combine_samples(samples, self.gen_width, sample_image.shape[0], background)

            for box in sample_boxes:
                box[0] += self.gen_width

            image = np.hstack((image, sample_image))
            boxes += sample_boxes


        if cfg.TRAIN.DOUBLE_GENERATE:
            background_path, background_boxes = random.choice(self._backgrounds)
            background = cv2.imread(background_path)
            scale_y = self.gen_height / background.shape[0]
            new_width = int(scale_y * background.shape[1])
            scale_x = new_width / background.shape[1]
            background = cv2.resize(background, (new_width, self.gen_height))

            for box in background_boxes:
                box[0] *= scale_x
                box[1] *= scale_y
                box[2] *= scale_x
                box[3] *= scale_y
                box[0] += self.gen_width
            boxes += background_boxes
            image = np.hstack((image, background))

        image_path = './generated/%d.jpg' % self._generated_cnt
        roidb_element = self.get_roidb_element(boxes, image_path,
                                               ImageLoader(image_path, image))

        if self._generated_cnt < 1000:
            mkdir('./generated')
            timage = image.copy()
            timage = plot_bboxes(timage, boxes)
            cv2.imwrite(image_path, timage)
        self._generated_cnt += 1

        return roidb_element

    def image_path_at(self, i):
        return self._gt_list[self._image_index[i]][0]


class RTSDSigns(DetectionsBaseRoidb):
    def __init__(self, dataset_path, mode):
        assert mode in ['train', 'test'], "Unknown mode"

        self._mode = mode
        self._dataset_path = dataset_path

        dataset_name = os.path.basename(dataset_path) + '_' + mode
        super(RTSDSigns, self).__init__(dataset_name)

    def gt_base_init(self):
        self._classes = ['__background__', 'sign']

        filter_conditions = [('sign_class', r'(?!((.*unknown.*)|(6_16|6_8_.*|6_15_3)|(8_.*)))', None), # (?!(.*unknown.*))
                           ('width', 24, 600), ('height', 24, 600),
                           ('false_positive', 0, 0), ('blurred',0,0,),
                           ('beyond_borders', 0,0), ('narrow',0,0),
                           ('badly_visible',0,0), ('overlapped', 0,1)]

        gt_base = DetectionsBase('/home/local/work/rtsd-frames')
        gt_base.load(self._dataset_path)

        gt_base.merge_object_classes(['1_20', '1_20_2', '1_20_3', '1_21'], '1_20-21')
        gt_base.merge_object_classes(['1_24', '1_25', '1_26', '1_27', '1_28',
                                      '1_29', '1_30', '1_31', '1_32', '1_33'], '1_24-33')

        gt_base.merge_object_classes(['2_3_2','2_3_3','2_3_4','2_3_5','2_3_6'], '2_3')

        gt_base.merge_object_classes(['3_3', '3_5', '3_6', '3_7',
                                      '3_4_1', '3_4_n', '3_4_n2', '3_4_n5', '3_4_n7',
                                      '3_4_n8'], '3_3-7')
        gt_base.merge_object_classes(['3_24_n', '3_24_n10', '3_24_n15', '3_24_n30',
                                      '3_24_n45'], '3_24_n_rare')
        gt_base.merge_object_classes(['3_25_n', '3_25_n20', '3_25_n40',
                                      '3_25_n50', '3_25_n70', '3_25_n80'], '3_25')
        gt_base.merge_object_classes(['3_12_n3', '3_12_n5', '3_12_n6', '3_12_n10',
                                      '3_13_r', '3_13_r2.5', '3_13_r3', '3_13_r3.3',
                                      '3_13_r3.5', '3_13_r3.7', '3_13_r3.9', '3_13_r4',
                                      '3_13_r4.0', '3_13_r4.1', '3_13_r4.2', '3_13_r4.3',
                                      '3_13_r4.5', '3_13_r5', '3_13_r5.2',
                                      '3_14_r2.7', '3_14_r3', '3_14_r3.5', '3_14_r3.7',
                                      '3_16_n', '3_16_n1', '3_16_n3'], '3_12_13_14_16')
        gt_base.merge_object_classes(['3_11_n5', '3_11_n8','3_11_n9','3_11_n13',
                                      '3_11_n17','3_11_n20','3_11_n23'], '3_11')

        gt_base.merge_object_classes(['4_1_2_1', '4_1_2_2'], '4_1_2_12')
        gt_base.merge_object_classes(['4_1_2', '4_1_3'], '4_1_23')
        gt_base.merge_object_classes(['4_1_4', '4_1_5'], '4_1_45')
        gt_base.merge_object_classes(['4_2_1', '4_2_2'], '4_2_12')
        gt_base.merge_object_classes(['4_3', '4_4', '4_5'], '4_345')
        gt_base.merge_object_classes(['4_8_1', '4_8_2', '4_8_3'], '4_8')

        gt_base.merge_object_classes(['5_21', '5_22'], '5_2(1|2)')
        gt_base.merge_object_classes(['5_3', '5_4'], '5_(3|4)')
        gt_base.merge_object_classes(['5_16', '5_17', '5_18'], '5_16_17_18')
        gt_base.merge_object_classes(['5_11', '5_12'], '5_11_12')

        gt_base.merge_object_classes(['6_2_n20', '6_2_n40', '6_2_n50', '6_2_n60',
                                      '6_2_n70'], '6_2_n')
        gt_base.merge_object_classes(['6_15_1', '6_15_2'], '6_15')

        gt_base.merge_object_classes(['7_11', '7_12', '7_13', '7_14', '7_15', '7_18',
                                      '7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7'], '7_all')

        self._gt_base = gt_base
        self._gt_base.set_filter_conditions(filter_conditions)

        gt_train, gt_test = self._gt_base.get_traintest(keep_ignore=True, train_fraction=0.80)
        self._backgrounds = list(self._gt_base.get_backgrounds(pure_backgrounds=True).items())

        if self._mode == 'train':
            self._gt = gt_train
        else:
            self._gt = gt_test

        if self._mode == 'train' and cfg.TRAIN.GENERATED_FRACTION > 0:
            self._positives = extract_positives_gt(self._gt, 96, 5, sratio=1.5)

    def sample_jittering(self, sample):
        return jitter_sample(sample, jittering_angle=8,
                                     jittering_contrast=0.2,
                                     jittering_illumination=0.2)

    def collect_negatives(self):
        return collect_random_negatives_from_gt(self._gt_base.get_backgrounds(pure_backgrounds=True),
                                                4000, 200, (96, 144))

    def init_params(self):
        self.gen_num_pos = 20
        self.gen_num_neg = 6
        self.gen_width = 500
        self.gen_height = 720
        self.gen_min_height = 16
        self.gen_max_height = 72


class CaltechPedestrians(DetectionsBaseRoidb):
    def __init__(self, dataset_path, mode):
        assert mode in ['train', 'test'], "Unknown mode"

        self._mode = mode
        self._dataset_path = dataset_path

        dataset_name = os.path.basename(dataset_path) + '_' + mode
        super(CaltechPedestrians, self).__init__(dataset_name)

    def gt_base_init(self):
        self._classes = ['__background__', 'pedestrian']

        filter_conditions = [('height', 50, 6000), ('width', 10, 5000),
                             ('sign_class', '(?!((person\?)|(people)))', None),
                             ('overlapped', 0, 1), ('score', 0, 0.35)]

        self._gt_base = DetectionsBase(os.path.join(self._dataset_path, self._mode), load_gt=True)
        self._gt_base.set_filter_conditions(filter_conditions)

        self._backgrounds = list(self._gt_base.get_backgrounds(pure_backgrounds=True).items())

        include_backgrounds = self._mode == 'test'
        self._gt = self._gt_base.get_gt(keep_ignore=True,
                                        include_backgrounds=include_backgrounds)

        if self._mode == 'train' and cfg.TRAIN.GENERATED_FRACTION > 0:
            self._positives = extract_positives_gt(self._gt, 80, 5, sratio=1.5)

    def sample_jittering(self, sample):
        return jitter_sample(sample, jittering_angle=6,
                                     jittering_contrast=0.2,
                                     jittering_illumination=0.2)

    def collect_negatives(self):
        return collect_random_negatives_from_gt(self._gt, 4000, 200, (80, 120))

    def init_params(self):
        self.gen_num_pos = 10
        self.gen_num_neg = int(self.gen_num_pos * 0.8)
        self.gen_width = 640
        self.gen_height = 480
        self.gen_min_height = 50
        self.gen_max_height = 90


class TownCenterPedestrians(DetectionsBaseRoidb):
    def __init__(self, dataset_path, mode):
        assert mode in ['train', 'test'], "Unknown mode"

        self._mode = mode
        self._dataset_path = dataset_path

        dataset_name = os.path.basename(dataset_path) + '_' + mode
        super(TownCenterPedestrians, self).__init__(dataset_name)

    def gt_base_init(self):
        self._classes = ['__background__', 'pedestrian']

        filter_conditions = [('sign_class', '(?!((person\?)|(people)))', None),
                             ('beyond_borders', 0, 0)]

        self._gt_base = DetectionsBase(os.path.join(self._dataset_path, self._mode), load_gt=True)
        self._gt_base.set_filter_conditions(filter_conditions)

        include_backgrounds = self._mode == 'test'
        self._gt = self._gt_base.get_gt(keep_ignore=True,
                                        include_backgrounds=include_backgrounds)
        self._backgrounds = list(self._gt.items())

    def init_params(self):
        self.gen_num_pos = 10
        self.gen_num_neg = int(self.gen_num_pos * 0.8)
        self.gen_width = 1280
        self.gen_height = 720
        self.gen_min_height = 20
        self.gen_max_height = 100


class FacesDataset(DetectionsBaseRoidb):
    def __init__(self, dataset_path, mode):
        self._dataset_path = dataset_path
        self._mode = mode

        dataset_name = osp.basename(dataset_path)
        super(FacesDataset, self).__init__(dataset_name)

    def gt_base_init(self):
        self._classes = ['__background__', 'face']

        imgs_dir = osp.join(self._dataset_path, 'imgs')
        gt_path = osp.join(self._dataset_path, 'gt.pickle')
        self._gt_base = DetectionsBase(images_directory=imgs_dir)
        self._gt_base.load(gt_path)

        blacklist_path = osp.join(self._dataset_path, 'blacklist.pickle')
        if osp.exists(blacklist_path):
            self._gt_base.load_blacklist(blacklist_path)
            print('%d images was ignored' % len(self._gt_base._blacklist_images))

        filter_conditions = [('sign_class', 'face', None)]
        self._gt_base.set_filter_conditions(filter_conditions)

        self._gt = self._gt_base.get_gt(keep_ignore=True,
                                        include_backgrounds=True)
        self._backgrounds = None

        if cfg.TRAIN.GENERATED_FRACTION > 0:
            self._positives = extract_positives_gt(self._gt, 128, 3, sratio=1.0)

    def init_params(self):
        self.gen_num_pos = 10
        self.gen_num_neg = 5
        self.gen_width = 400
        self.gen_height = 500
        self.gen_min_height = 64
        self.gen_max_height = 200

    def sample_jittering(self, sample):
        return jitter_sample(sample, jittering_angle=6,
                                     jittering_contrast=0.2,
                                     jittering_illumination=0.2)

    def collect_negatives(self):
        return collect_random_negatives_from_gt(self._gt, 4000, 200, (128, 128))


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
    order = list(range(len(heights)))#np.argsort(heights)[::-1]

    horizontal_boxes = []
    line_boxes = []

    total_width = 0
    total_height = 0
    max_height = 0
    for indx in order:
        width = samples[indx][0].shape[1]
        height = samples[indx][0].shape[0]

        if width > out_width:
            continue

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
    inserted_boxes = []
    for line in horizontal_boxes:
        shifts = combine_horizontal(line, out_width)
        line_height = 0

        for indx, (image, boxes) in enumerate(line):
            sx = shifts[indx][0]
            sy = last_h + shifts[indx][1]

            inserted_boxes.append([sx, sy, image.shape[1], image.shape[0]])
            out_image[sy:sy + image.shape[0],
                      sx:sx + image.shape[1], :] = image
            for box in boxes:
                box = box.copy()
                box[0] += sx
                box[1] += sy
                gt_boxes.append(box)

            line_height = max(line_height, image.shape[0])
        last_h += line_height

    return out_image, gt_boxes, inserted_boxes


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


def redistribute(prob, low_fraction, high_fraction, redistr_k):
    prob = prob.copy()
    order = np.argsort(prob)

    high_len = int(high_fraction * len(prob))
    vsum = np.sum(prob[order[-high_len:]]) * redistr_k
    prob[order[-high_len:]] -= vsum / high_len

    low_len = int(low_fraction * len(prob))
    prob[order[:low_len]] += vsum / low_len

    return prob


def prob_sample(population, weights, num_samples):
    assert len(population) == len(weights)
    cdf_vals = np.cumsum(weights) / np.sum(weights)

    ans = [population[bisect.bisect(cdf_vals, np.random.uniform())]
           for i in range(num_samples)]

    return ans