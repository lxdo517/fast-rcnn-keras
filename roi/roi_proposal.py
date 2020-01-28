from roi.roi_tf import roi_pool_tf


def roi_proposal(args):
    pooled_features = roi_pool_tf(args[0], args[1])
    return pooled_features
