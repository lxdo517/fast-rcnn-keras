import tensorflow as tf
import config as cfg


def fast_loss(args):
    cls_output, labels, bbox_output, bbox_targets, bbox_inside_weights, bbox_outside_weights = args
    labels = tf.cast(labels, dtype=tf.int32)
    # 分类损失
    cls_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_output, labels=tf.squeeze(labels)))
    diff = tf.multiply(bbox_inside_weights, bbox_output - bbox_targets)
    # 对于论文中提到为什么要用L1 smooth loss 而不是L2损失
    # 论文中是说 L1 loss that is less sensitive to outliers than the L2 loss used in R-CNN and SPPnet
    # 1 说起来就是对于网络初期, 由于真实值和预测值之间误差较大, L2 损失的梯度也会过大可能导致梯度爆炸, 网络模型不稳定
    # 2 而对于网络训练后期, 损失已经很小了, 在lr不变的情况下, 梯度绝对值1, 损失函数将在稳定值附近继续波动, 达不到更高的精度
    diff_l1 = smooth_l1(diff, 1.0)
    # 边框回归损失
    roi_bbox_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(bbox_outside_weights, diff_l1), axis=1))
    fast_loss = cls_loss + roi_bbox_loss
    return fast_loss


def smooth_l1(x, sigma):
    '''
                      0.5 * (sigma * x)^2  if |x| < 1/sigma^2
      smoothL1(x) = {
                      |x| - 0.5/sigma^2    otherwise
    '''

    with tf.variable_scope('smooth_l1'):
        conditional = tf.less(tf.abs(x), 1 / sigma ** 2)
        close = 0.5 * (sigma * x) ** 2
        far = tf.abs(x) - 0.5 / sigma ** 2
        return tf.where(conditional, close, far)
