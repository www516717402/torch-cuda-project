import numpy as np
import torch
from cuda_example._ext import nms as ops_nms


def nms_forward(
    bboxes, scores, iou_threshold, offset, score_threshold, max_num
):
    is_filtering_by_score = score_threshold > 0
    if is_filtering_by_score:
        valid_mask = scores > score_threshold
        bboxes, scores = bboxes[valid_mask], scores[valid_mask]
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

    inds = ops_nms(
        bboxes, scores, iou_threshold=float(iou_threshold), offset=offset
    )

    if max_num > 0:
        inds = inds[:max_num]
    if is_filtering_by_score:
        inds = valid_inds[inds]
    return inds


def nms(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        tuple: kept dets (boxes and scores) and indice, which always have
        the same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    inds = nms_forward(
        boxes, scores, iou_threshold, offset, score_threshold, max_num
    )
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


def test_nms():
    np_boxes = np.array(
        [
            [6.0, 3.0, 8.0, 7.0],
            [3.0, 6.0, 9.0, 11.0],
            [3.0, 7.0, 10.0, 12.0],
            [1.0, 4.0, 13.0, 7.0],
        ],
        dtype=np.float32,
    )
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    np_inds = np.array([1, 0, 3])
    np_dets = np.array(
        [
            [3.0, 6.0, 9.0, 11.0, 0.9],
            [6.0, 3.0, 8.0, 7.0, 0.6],
            [1.0, 4.0, 13.0, 7.0, 0.2],
        ]
    )
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)
    dets, inds = nms(boxes, scores, iou_threshold=0.3, offset=0)
    assert np.allclose(dets, np_dets)  # test cpu
    assert np.allclose(inds, np_inds)  # test cpu
    dets, inds = nms(boxes.cuda(), scores.cuda(), iou_threshold=0.3, offset=0)
    assert np.allclose(dets.cpu().numpy(), np_dets)  # test gpu
    assert np.allclose(inds.cpu().numpy(), np_inds)  # test gpu
    print('test nms finish.')


if __name__ == '__main__':
    test_nms()
