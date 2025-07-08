#!/usr/bin/env python3
import argparse
from typing import Callable, Tuple
import unittest

import numpy as np

# Pat a Mat:
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

B = np  # The backend to use; you can use `tf` for TensorFlow implementation.

# Bounding boxes and anchors are expected to be Numpy/TensorFlow tensors,
# where the last dimension has size 4.
Tensor = np.ndarray  # or use `tf.Tensor` if you use TensorFlow backend

# For bounding boxes in pixel coordinates, the 4 values correspond to:
TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3


def bboxes_area(bboxes: Tensor) -> Tensor:
    """ Compute area of given set of bboxes.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    If the bboxes.shape is [..., 4], the output shape is bboxes.shape[:-1].
    """
    return B.maximum(bboxes[..., BOTTOM] - bboxes[..., TOP], 0) \
        * B.maximum(bboxes[..., RIGHT] - bboxes[..., LEFT], 0)


def bboxes_iou(xs: Tensor, ys: Tensor) -> Tensor:
    """ Compute IoU of corresponding pairs from two sets of bboxes `xs` and `ys`.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    Note that broadcasting is supported, so passing inputs with
    `xs.shape=[num_xs, 1, 4]` and `ys.shape=[1, num_ys, 4]` produces an output
    with shape `[num_xs, num_ys]`, computing IoU for all pairs of bboxes from
    `xs` and `ys`. Formally, the output shape is `np.broadcast(xs, ys).shape[:-1]`.
    """
    intersections = B.stack([
        B.maximum(xs[..., TOP], ys[..., TOP]),
        B.maximum(xs[..., LEFT], ys[..., LEFT]),
        B.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
        B.minimum(xs[..., RIGHT], ys[..., RIGHT]),
    ], axis=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)


def bboxes_to_fast_rcnn(anchors: Tensor, bboxes: Tensor) -> Tensor:
    """ Convert `bboxes` to a Fast-R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use the TOP, LEFT, BOTTOM, RIGHT constants as indices of the
    respective coordinates.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)

    If the `anchors.shape` is `[anchors_len, 4]` and `bboxes.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """
    #mine:
    #print("BBOXES->FAST_RCNNS:")
    #print("anchors+shape:", anchors, anchors.shape)
    #print("bbooxes+shape:", bboxes, bboxes.shape)
    bbox_height = B.abs(bboxes[..., TOP] - bboxes[..., BOTTOM])
    bbox_width = B.abs(bboxes[..., LEFT] - bboxes[..., RIGHT])
    anchor_height = B.abs(anchors[..., TOP] - anchors[..., BOTTOM])
    anchor_width = B.abs(anchors[..., LEFT] - anchors[..., RIGHT])
    bbox_y_center = bboxes[..., BOTTOM] - bbox_height / 2
    bbox_x_center = bboxes[..., LEFT] + bbox_width / 2
    anchor_y_center = anchors[..., BOTTOM] - anchor_height / 2
    anchor_x_center = anchors[..., LEFT] + anchor_width / 2
    #print("a_h, a_w, a_xc, a_yc:", anchor_height, anchor_width, anchor_x_center, anchor_y_center)
    #print("b_h, b_w, b_xc, b_yc:", bbox_height, bbox_width, bbox_x_center, bbox_y_center,)
    t_x = (bbox_y_center - anchor_y_center) / anchor_height
    t_y = (bbox_x_center - anchor_x_center) / anchor_width
    t_h = B.log(bbox_height / anchor_height)
    t_w = B.log(bbox_width / anchor_width)
    #print("t_x, t_y, t_h, t_w:", t_x, t_y, t_h, t_w)

    return B.stack([t_x, t_y, t_h, t_w], -1)

    # TODO: Implement according to the docstring.
    #raise NotImplementedError()


def bboxes_from_fast_rcnn(anchors: Tensor, fast_rcnns: Tensor) -> Tensor:
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`.

    The `anchors.shape` is `[anchors_len, 4]`, `fast_rcnns.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """

    #mine:
    #- (bbox_y_center - anchor_y_center) / anchor_height
    #- (bbox_x_center - anchor_x_center) / anchor_width
    #- log(bbox_height / anchor_height)
    #- log(bbox_width / anchor_width)

    #print("FAST_RCNNS->BBOXES:")
    #print("anchors+shape:", anchors, anchors.shape)
    #print("fast_rcnns+shape:", fast_rcnns, fast_rcnns.shape)
    anchor_height = B.abs(anchors[..., TOP] - anchors[..., BOTTOM])
    anchor_width = B.abs(anchors[..., LEFT] - anchors[..., RIGHT])
    anchor_y_center = anchors[..., BOTTOM] - anchor_height / 2
    anchor_x_center = anchors[..., LEFT] + anchor_width / 2
    #print("a_h, a_w, a_cx, a_cy:", anchor_height, anchor_width, anchor_x_center, anchor_y_center)
    bbox_y_center = fast_rcnns[..., 0] * anchor_height + anchor_y_center
    bbox_x_center = fast_rcnns[..., 1] * anchor_width + anchor_x_center
    bbox_height = B.exp(fast_rcnns[..., 2]) * anchor_height
    bbox_width = B.exp(fast_rcnns[..., 3]) * anchor_width
    #print("b_h, b_w, b_cx, b_cy:", bbox_height, bbox_width, bbox_x_center, bbox_y_center)
    b_L = bbox_x_center - bbox_width / 2
    b_R = bbox_x_center + bbox_width / 2
    b_T = bbox_y_center - bbox_height / 2
    b_B = bbox_y_center + bbox_height / 2
    #print("b_T, b_L, b_B, b_R:", b_T, b_L, b_B, b_R)
    #return B.stack([b_T, b_L, b_B, b_R])
    return B.stack([b_T, b_L, b_B, b_R], -1)



    # TODO: Implement according to the docstring.
    #raise NotImplementedError()


def bboxes_training(
    anchors: Tensor, gold_classes: Tensor, gold_bboxes: Tensor, iou_threshold: float
) -> Tuple[Tensor, Tensor]:
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` is assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN; zeros if no
      gold object was assigned to the anchor

    Algorithm:
    - First, for each gold object, assign it to an anchor with the largest IoU
      (the one with smaller index if there are several). In case several gold
      objects are assigned to a single anchor, use the gold object with smaller
      index.
    - For each unused anchor, find the gold object with the largest IoU
      (again the one with smaller index if there are several), and if the IoU
      is >= iou_threshold, assign the object to the anchor.
    """

    # TODO: First, for each gold object, assign it to an anchor with the
    # largest IoU (the one with smaller index if there are several). In case
    # several gold objects are assigned to a single anchor, use the gold object
    # with smaller index.

    #print("anchors, gold_classes:", anchors, anchors.shape, gold_classes, gold_classes.shape)
    #print("gold_bboxes, iou_threshold:", gold_bboxes, gold_bboxes.shape, iou_threshold)

    IoUs = np.zeros([gold_bboxes.shape[0], anchors.shape[0]])
    for i in range(gold_bboxes.shape[0]):
        for j in range(anchors.shape[0]):
            IoUs[i][j] = bboxes_iou(gold_bboxes[i], anchors[j])

    #print("IoU:", IoUs)

    #anchor_bboxes = [[0, 0] for _ in range(anchors.shape[0])]
    anchor_bboxes = - B.ones(anchors.shape[0])

    anchs_for_each_gold = B.argmax(IoUs, axis=-1)
    #print("anchs_for_each_gold:", anchs_for_each_gold)

    for i in range(gold_bboxes.shape[0]):
        if anchor_bboxes[anchs_for_each_gold[i]] == -1:
            anchor_bboxes[anchs_for_each_gold[i]] = i
    #print("anchor_bboxes:", anchor_bboxes)


    #golds_with_anch = []
    #for i in range(gold_bboxes.shape[0]):
    #    best_anchor = anchors[B.argmax(IoUs[i, :])]
    #    golds_with_anch.append(best_anchor)

    #golds_reduced = []
    #for gold in golds_with_anch:
    #    if gold not in golds_reduced:
    #        golds_reduced.append(gold)
    #    else:


    # TODO: For each unused anchor, find the gold object with the largest IoU
    # (again the one with smaller index if there are several), and if the IoU
    # is >= threshold, assign the object to the anchor.


    ind = np.where(anchor_bboxes == -1)[0]
    #print("ind:", ind[0])
    #print("ind+shape:", ind, ind.shape)
    for i in ind:
        argmax = B.argmax(IoUs, axis=0)[i]
        if IoUs[argmax, i] >= iou_threshold:
            anchor_bboxes[i] = argmax
    #print("anchor_bboxes:", anchor_bboxes)

    boxes_class = []
    boxes = []
    for i in range(anchors.shape[0]):
        #print("anch_bbox[i]:", anchor_bboxes[i])
        if anchor_bboxes[i] == -1:
            boxes_class.append(0)
            boxes.append(B.zeros(4))
        else:
            boxes_class.append(gold_classes[int(anchor_bboxes[i])] + 1)
            box = gold_bboxes[int(anchor_bboxes[i])]
            boxes.append(bboxes_to_fast_rcnn(anchors[i], box))
    anchor_classes = B.stack(boxes_class, 0)
    anchor_bboxes = B.stack(boxes, 0)
    #print("anchor_classes:", anchor_classes)
    #print("anchor_bboxes:", anchor_bboxes)



    #anchs_with_gold = []
    #for i in range(anchors.shape[0]):
    #    anch = anchors[i]
    #    if anch not in golds_with_anch:
    #        i_best = B.argmax(IoUs[:, i])
    #        if IoUs[i_best, i] >= iou_threshold:
    #            best_gold = gold_bboxes[i_best]
    #        a

    return anchor_classes, anchor_bboxes


def main(args: argparse.Namespace) -> Tuple[Callable, Callable, Callable]:
    return bboxes_to_fast_rcnn, bboxes_from_fast_rcnn, bboxes_training


class Tests(unittest.TestCase):
    def test_bboxes_to_from_fast_rcnn(self):
        data = [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 30], [.5, 1, np.log(2), np.log(3)]],
            [[0, 9, 10, 19], [2, 10, 5, 16], [-0.15, -0.1, -1.20397, -0.51083]],
            [[5, 3, 15, 13], [7, 7, 10, 9], [-0.15, 0, -1.20397, -1.60944]],
            [[7, 6, 17, 16], [9, 10, 12, 13], [-0.15, 0.05, -1.20397, -1.20397]],
            [[5, 6, 15, 16], [7, 7, 10, 10], [-0.15, -0.25, -1.20397, -1.20397]],
            [[6, 3, 16, 13], [8, 5, 12, 8], [-0.1, -0.15, -0.91629, -1.20397]],
            [[5, 2, 15, 12], [9, 6, 12, 8], [0.05, 0, -1.20397, -1.60944]],
            [[2, 10, 12, 20], [6, 11, 8, 17], [0, -0.1, -1.60944, -0.51083]],
            [[10, 9, 20, 19], [12, 13, 17, 16], [-0.05, 0.05, -0.69315, -1.20397]],
            [[6, 7, 16, 17], [10, 11, 12, 14], [0, 0.05, -1.60944, -1.20397]],
            [[2, 2, 12, 12], [3, 5, 8, 8], [-0.15, -0.05, -0.69315, -1.20397]],
        ]
        # First run on individual anchors, and then on all together
        for anchors, bboxes, fast_rcnns in [map(lambda x: [x], row) for row in data] + [zip(*data)]:
            anchors, bboxes, fast_rcnns = [np.array(data, np.float32) for data in [anchors, bboxes, fast_rcnns]]
            np.testing.assert_almost_equal(bboxes_to_fast_rcnn(anchors, bboxes), fast_rcnns, decimal=3)
            np.testing.assert_almost_equal(bboxes_from_fast_rcnn(anchors, fast_rcnns), bboxes, decimal=3)

    def test_bboxes_training(self):
        anchors = np.array([[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]], np.float32)
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
                [[1], [[14., 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(.2), np.log(.2)]], 0.5],
                [[2], [[0., 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
                [[2], [[0., 0, 20, 20]], [3, 3, 3, 3],
                 [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 0, 1],
                 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-0.35, -0.45, 0.53062, 0.40546]], 0.5],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 2, 1],
                 [[0, 0, 0, 0], [0, 0, 0, 0], [-0.1, 0.6, -0.22314, 0.69314], [-0.35, -0.45, 0.53062, 0.40546]], 0.3],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 1, 2, 1],
                 [[0, 0, 0, 0], [0.65, -0.45, 0.53062, 0.40546], [-0.1, 0.6, -0.22314, 0.69314],
                  [-0.35, -0.45, 0.53062, 0.40546]], 0.17],
        ]:
            gold_classes, anchor_classes = np.array(gold_classes, np.int32), np.array(anchor_classes, np.int32)
            gold_bboxes, anchor_bboxes = np.array(gold_bboxes, np.float32), np.array(anchor_bboxes, np.float32)
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            np.testing.assert_almost_equal(computed_classes, anchor_classes, decimal=3)
            np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)


if __name__ == '__main__':
    unittest.main()
