import numpy as np


def bbox_transform(ex_rois, gt_rois):
    """
      返回生成的anchor和真实box之间的线性映射
      Gˆx =Pwdx(P)+Px (1)
      Gˆy =Phdy(P)+Py (2)
      Gˆw = Pw exp(dw(P))(3)
      Gˆh = Ph exp(dh(P))
      tx = (Gx − Px)/Pw (6)
      ty=(Gy−Py)/Ph (7)
      tw = log(Gw/Pw) (8)
      th = log(Gh/Ph).
      :param ex_rois:
      :param gt_rois:
      :return:
    """
    # 首先转换成中心点坐标
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1)

    dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    dw = np.log(gt_widths / ex_widths)
    dh = np.log(gt_heights / ex_heights)
    targets = np.vstack((dx, dy, dw, dh)).transpose()
    return targets
