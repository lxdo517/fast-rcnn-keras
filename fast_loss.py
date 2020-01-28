import tensorflow as tf
import config as cfg


def fast_loss(args):
    cls_output, labels, bbox_output, bbox_targets, bbox_inside_weights, bbox_outside_weights = args
    labels = tf.cast(labels, dtype=tf.int32)
    # 分类损失
    cls_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_output, labels=tf.squeeze(labels)))
    diff = tf.multiply(bbox_inside_weights, bbox_output - bbox_targets)
    diff_l1 = smooth_l1(diff, 1.0)
    # 边框回归损失
    roi_bbox_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(bbox_outside_weights, diff_l1), axis=1))
    roi_bbox_loss = cfg.TRAIN_RPN_BBOX_LAMBDA * roi_bbox_loss
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
