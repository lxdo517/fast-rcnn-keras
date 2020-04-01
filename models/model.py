import tensorflow as tf
from tensorflow.keras import Model
from models.vgg16_body import get_model_body
from tensorflow.keras.layers import Dense, Flatten, Dropout, Lambda
from roi.roi_proposal import roi_proposal
from fast_loss import fast_loss as loss


class FastRCNN(Model):
    def __init__(self, num_classes, keep_prob=0.5):
        super(FastRCNN, self).__init__()
        self._num_classes = num_classes
        self._vgg16 = get_model_body()
        # roi pooling 不参与反向传播
        self._roi_pooling = Lambda(roi_proposal)
        self._flatten = Flatten()
        self._fc1 = Dense(4096, activation='tanh')
        self._dropout1 = Dropout(keep_prob)
        self._fc2 = Dense(4096, activation='tanh')
        self._dropout2 = Dropout(keep_prob)
        # predict k + 1 categories  k个类别加上背景
        # (None, 128, 21)
        self._fc_cls = Dense(num_classes + 1, name='cls_output')
        # predict 4 * k 个值 每个类4个坐标回归值
        # (None, 128, 80)
        self._fc_bbox = Dense(num_classes * 4, name='bbox_output')
        # 计算损失
        self._loss = Lambda(loss, name='fast_loss')

    def call(self, inputs, mask=None):
        image_data, labels, regions_target, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
        # (None, 36, 36, 512)
        x = self._vgg16(image_data)
        # seletvie_search 貌似有点问题参数可能不对 不能够采样128个满足条件样本
        # (None, 128, 7, 7, 512)
        x = self._roi_pooling([x, regions_target])
        x = self._flatten(x)
        x = self._fc1(x)
        x = self._dropout1(x)
        x = self._fc2(x)
        x = self._dropout2(x)
        # (batch_size, 128, 21)
        cls_output = self._fc_cls(x)
        # (batch_size, 128, 80)
        bbox_output = self._fc_bbox(x)
        loss = self._loss([cls_output, labels, bbox_output, bbox_targets, bbox_inside_weights, bbox_outside_weights])
        return loss