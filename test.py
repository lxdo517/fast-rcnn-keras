import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from utils.nms import nms
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from ss.selectivesearch import selective_search
import config as cfg

parse = argparse.ArgumentParser(description='fast rcnn model params')
parse.add_argument('--model_path', default='./logs/fast_rcnn_model.h5', help='')
parse.add_argument('--img_path', default='', help='')

classes = ['bg', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def get_proposal(img):
    # BGR -> RGB 做简单处理
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img = img / 255.

    # regions 里面 是 x1, y1, x2, y2
    _, regions = selective_search(img, scale=200, sigma=0.9, min_size=50)

    rects = np.asarray([list(region['rect']) for region in regions])
    selected_imgs = []
    candidates = set()
    # 过滤掉一些框
    for r in rects:
        x1, y1, x2, y2 = r
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if (x1, y1, x2, y2) in candidates:
            continue
        if (x2 - x1) * (y2 - y1) < 220:
            continue
        crop_img = img[y1:y2, x1:x2, :]
        # 裁剪后进行resize
        crop_img = cv2.resize(crop_img, (cfg.DEFAUTL_IMAGE_SIZE, cfg.DEFAUTL_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        selected_imgs.append(crop_img)
        candidates.add((x1, y1, x2, y2))

    rects = [list(candidate) for candidate in candidates]
    return np.asarray(selected_imgs), np.asarray(rects)


def show_rect(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def softmax(cls_output):
    rpn_cls_output = np.exp(cls_output)
    rpn_cls_output = rpn_cls_output / np.sum(rpn_cls_output, axis=-1, keepdims=True)
    return rpn_cls_output


def get_inputs(img_path):
    img = cv2.imread(img_path)
    # height/width/channel
    height, width, _ = img.shape
    # img resize
    img = cv2.resize(img, (cfg.DEFAUTL_IMAGE_SIZE, cfg.DEFAUTL_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    # image_data, labels, regions_target, bbox_targets, bbox_inside_weights, bbox_outside_weights = next(g)[0]
    # (1, 576, 576, 3) (1, 128) (1, 128, 5) (1, 128, 80) (1, 128, 80) (1, 128, 80)
    # 测试是不需要 labels bbox_targets, bbox_inside_weights, bbox_outside_weights 所以全部取0
    imgs, rects = get_proposal(img)
    imgs = imgs[np.newaxis, ...]
    m, n = rects.shape
    rects = rects[np.newaxis, ...]
    return [imgs, np.zeros((1, m)), rects, np.zeros((1, m, 80)), np.zeros((1, m, 80)), np.zeros((1, m, 80))]


def main():
    args = parse.parse_args()
    model_path = args.model_path
    img_path = args.img_path
    if model_path.strip() == '':
        raise ValueError('model path should not be null')
    if img_path.strip() == '':
        raise ValueError('test img path should not be null')
    model = load_model(model_path)

    test_model = Model(model.input, [model.get_layer('cls_output').output, model.get_layer('bbox_output').output])
    test_model.load_weights(model_path, by_name=True, skip_mismatch=True)
    # 获取输入信息
    inputs = get_inputs()
    # shape (1, 128, 21) shape (1, 128, 80)
    cls_output, bbox_ouput = test_model(inputs)
    # shape (128, 21)
    cls_output = np.squeeze(cls_output, axis=0)
    # shape (128, 80)
    bbox_ouput = np.squeeze(bbox_ouput, axis=0)
    # 进行softmax
    cls_output = softmax(cls_output)
    # 找出128个边框的最大类别 shape (128, )
    argmax_cls = np.argmax(cls_output, axis=1)

    cls_output = cls_output[argmax_cls > 0]
    # (n, ) n <= 128
    argmax_cls = argmax_cls[argmax_cls > 0]
    # (n, 80)
    bbox_ouput = cls_output[bbox_ouput > 0]
    scores = np.max(cls_output, axis=1)
    rects = []
    for i, bbox in enumerate(bbox_ouput):
        # 去掉背景
        cls = argmax_cls[i] - 1
        start = cls * 4
        end = start + 4
        bbox = bbox[start:end]
        rects.append(bbox)
    rects = np.asarray(rects)
    # 非极大值抑制
    keep_ind = nms(rects, scores, 0.5)
    rects = rects[keep_ind, :]
    show_rect(img_path, rects)


if __name__ == '__main__':
    main()
