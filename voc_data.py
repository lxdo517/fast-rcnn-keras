import numpy as np
from PIL import Image
import config as cfg
from ss.selectivesearch import selective_search
from utils.bbox_overlaps import bbox_overlaps
from utils.bbox_transform import bbox_transform


class VocData(object):
    def __init__(self, data_path, voc_annotation):
        self._num_classes = len(voc_annotation.class_names)
        self._annotations = self._parse_data_path(data_path)
        self.example_nums = len(self._annotations)
        self._indicator = 0
        self._random_shuffle()

    def _parse_data_path(self, data_path):
        annotations = open(data_path).readlines()
        annotations = [annotation.strip() for annotation in annotations]
        return np.asarray(annotations)

    def _random_shuffle(self):
        x = np.random.permutation(self.example_nums)
        self._annotations = self._annotations[x]

    def data_generator_wrapper(self, batch_size=1):
        # assert batch_size == 1, 'batch_size should be 1'
        return self._data_generator(batch_size)

    def _data_generator(self, batch_size):
        i = 0
        n = self.example_nums
        while True:
            image_data = []
            gt_box_data = []
            for b in range(batch_size):
                annotation = self._annotations[i]
                image, gt_boxes = self._parse_annotation(annotation)
                image_data.append(image)
                gt_box_data.append(gt_boxes)
                i = (i + 1) % n
            image_data = np.array(image_data)
            gt_box_data = np.array(gt_box_data)
            labels, regions_target, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
                self._process_gt_bboxes(image_data, gt_box_data)
            yield [image_data, labels, regions_target, bbox_targets, bbox_inside_weights,
                              bbox_outside_weights], np.zeros(batch_size)

    def _parse_annotation(self, annotation):
        """
        返回图像数据 和 gt_boxes
        :param annotation:
        :return:
        """
        lines = annotation.strip().split()
        img_path = lines[0]
        img = Image.open(img_path)
        iw, ih = img.size
        scale_w = cfg.DEFAUTL_IMAGE_SIZE / iw
        scale_h = cfg.DEFAUTL_IMAGE_SIZE / ih
        img = img.resize((cfg.DEFAUTL_IMAGE_SIZE, cfg.DEFAUTL_IMAGE_SIZE), Image.BICUBIC)
        flip = np.random.rand()
        if flip < .5:
            # 图像以0.5概率水平翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_data = np.asarray(img, dtype=np.float32)
        # 将其变成0~1
        img_data = img_data / 255.0
        gt_boxes = np.asarray([np.asarray(list(map(float, box.split(',')))) for box in lines[1:]])
        # 将gt_boxes映射到resize后的图像上
        gt_boxes = np.vstack((gt_boxes[:, 0] * scale_w, gt_boxes[:, 1] * scale_h, gt_boxes[:, 2] * scale_w,
                              gt_boxes[:, 3] * scale_h, gt_boxes[:, 4])).transpose()
        if flip:
            # 修正boxes
            gt_boxes[:, [0, 2]] = cfg.DEFAUTL_IMAGE_SIZE - gt_boxes[:, [2, 0]]
        return img_data, gt_boxes

    def _process_gt_bboxes(self, img_data, gt_boxes):
        # 图像缩小的尺寸
        for idx, img in enumerate(img_data):
            # 可能参数需要调整 很多图像采样不到满足设置要求的正负样本数
            _, regions = selective_search(img, scale=80., sigma=0.8, min_size=50)
            candidates = set()
            for r in regions:
                if r['rect'] in candidates:
                    continue
                if r['size'] < 220:
                    continue
                if (r['rect'][2] * r['rect'][3]) < 500:
                    continue
                # r['rect']形式如下:x1, y1, w, h
                candidates.add(r['rect'])
            # 获取选中的regions
            regions = list(candidates)
            # 得到训练用的标签数据
            labels, regions_target, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
                self._get_labels(regions, gt_boxes[idx])
            return labels, regions_target, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def _get_labels(self, regions, gt_boxes):
        """
        获取训练用的标签数据
        :param regions:  n * 4  (x1, y1, w, h)
        :param gt_boxes: m * 5  (x1, y1, x2, y2, cls)
        :return:
        """
        # 将gt_boxes添加进regions增加正样本数量
        all_regions = np.vstack((regions, gt_boxes[:, :4]))
        # 1. 计算iou
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_regions[:, :], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        # 为每个anchor设置所属类别  与哪个gt_boxes相交iou最大就是对应的class
        labels = gt_boxes[gt_assignment, 4]
        # 2. 设置正负样本数目
        fg_inds = np.where(max_overlaps >= cfg.TRAIN_FG_THRESH)[0]
        # 128 * 0.25
        fg_rois_per_image = cfg.TRAIN_BATCH_SIZE * cfg.TRAIN_FG_FRACTION
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        if fg_inds.size > 0:
            # 随机抽样
            fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
        # [0.1, 0.5] 的region为背景  取不到足够的样本
        # bg_inds = np.where((max_overlaps < cfg.TRAIN_BG_THRESH_HI) &
        #                    (max_overlaps >= cfg.TRAIN_BG_THRESH_LO))[0]
        bg_inds = np.where(max_overlaps < cfg.TRAIN_BG_THRESH_HI)[0]
        bg_rois_per_this_image = cfg.TRAIN_BATCH_SIZE - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)

        if bg_inds.size > 0:
            bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        # 得到128个labels 和 regions用来训练
        # fast-rcnn论文中说batch_size=2, 每张图片64个训练样本 这里没有采用这种方法 直接使用faster-rcnn论文中
        # batch_size=1 , 每张图片采用128个
        labels = labels[keep_inds]
        labels[fg_rois_per_this_image:] = 0
        regions_target = all_regions[keep_inds]
        # 将regions转成回归值 tx ty tw th
        bbox_target_data = self._transform_regions(regions_target, gt_boxes[gt_assignment[keep_inds], :4])
        bbox_targets, bbox_inside_weights = self._get_bbox_regression_labels(bbox_target_data, labels)
        regions_target = np.vstack((np.zeros(regions_target.shape[0], ), regions_target[:, 0], regions_target[:, 1],
                                    regions_target[:, 2], regions_target[:, 3])).transpose()

        labels = labels.reshape((1, -1))
        regions_target = regions_target.reshape((1, -1, 5))
        # 减去背景类别
        bbox_targets = bbox_targets.reshape((1, -1, (self._num_classes - 1) * 4))
        bbox_inside_weights = bbox_inside_weights.reshape((1, -1, (self._num_classes - 1) * 4))
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
        return labels, regions_target, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def _transform_regions(self, regions, gt_boxes):
        regions_target = bbox_transform(regions, gt_boxes)
        return regions_target

    def _get_bbox_regression_labels(self, bbox_target_data, labels):
        # -1 是为了去掉背景
        clss = labels
        bbox_targets = np.zeros((clss.size, 4 * (self._num_classes - 1)), dtype=np.float32)
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        # 仅仅当bbox中含有物体的框才参与损失函数的计算
        inds = np.where(clss > 0)[0]
        for ind in inds:
            # 每个类回归4个坐标 按照顺序排序
            # 设置对应的坐标回归值
            cls = clss[ind] - 1
            start = int(4 * cls)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, :]
            # 仅仅当rois有物体才计算损失
            bbox_inside_weights[ind, start:end] = (1, 1, 1, 1)
        return bbox_targets, bbox_inside_weights

# from voc_annotation import VOCAnnotation
#
# voc_annotation = VOCAnnotation(2007, 'train', '/Users/lx/segment_data', './data/voc_classes.txt')
# voc_data = VocData('./data/2007_train.txt', voc_annotation)
#
# g = voc_data.data_generator_wrapper(1)

# for i in range(100):
#     image_data, labels, regions_target, bbox_targets, bbox_inside_weights, bbox_outside_weights = next(g)[0]
#     # (1, 576, 576, 3) (99,) (99, 5) (99, 80) (99, 80)
#     print(image_data.shape, labels.shape, regions_target.shape, bbox_targets.shape, bbox_inside_weights.shape,
#           bbox_outside_weights.shape)
