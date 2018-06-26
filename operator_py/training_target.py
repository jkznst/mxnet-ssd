import mxnet as mx
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def bbox_transform(ex_rois, gt_rois, box_stds):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14) / box_stds[0]
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14) / box_stds[1]
    targets_dw = np.log(gt_widths / ex_widths) / box_stds[2]
    targets_dh = np.log(gt_heights / ex_heights) / box_stds[3]

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


class TrainingTargets(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, overlap_threshold, negative_mining_ratio, negative_mining_thresh, variances):
        super(TrainingTargets, self).__init__()
        self.overlap_threshold = overlap_threshold
        self.negative_mining_ratio = negative_mining_ratio
        self.negative_mining_thresh = negative_mining_thresh
        self.variances = variances

        self.eps = 1e-14

    def forward(self, is_train, req, in_data, out_data, aux):

        anchors = in_data[0].asnumpy()    # 1 x num_anchors x 4
        anchors = np.reshape(anchors, newshape=(-1, 4)) # num_anchors x 4
        class_preds = in_data[1].asnumpy()    # batchsize x num_class x num_anchors
        labels = in_data[2].asnumpy()     # batchsize x 8 x 5

        batchsize = class_preds.shape[0]
        num_class = class_preds.shape[1]    # including background class
        num_anchors = class_preds.shape[2]

        # label: >0 is positive, 0 is negative, -1 is dont care
        cls_target = np.ones((batchsize, num_anchors), dtype=np.float32) * -1
        box_target = np.zeros((batchsize, num_anchors, 4), dtype=np.float32)
        box_mask = np.zeros((batchsize, num_anchors, 4), dtype=np.float32)

        for cls_preds_per_batch, labels_per_batch, cls_target_per_batch, box_target_per_batch, \
            box_mask_per_batch in zip(class_preds, labels, cls_target, box_target, box_mask):
            # filter out padded gt_boxes with cid -1
            valid_labels = np.where(labels_per_batch[:, 0] >= 0)[0]
            gt_boxes = labels_per_batch[valid_labels, 1:5]
            num_valid_gt = gt_boxes.shape[0]

            # overlap between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
            # sample for positive labels
            if num_valid_gt > 0:
                gt_flags = np.zeros(shape=(num_valid_gt, 1), dtype=np.bool)
                max_matches = np.ones(shape=(num_anchors, 2), dtype=np.float32) * -1
                anchor_flags = np.ones(shape=(num_anchors, 1), dtype=np.int8) * -1  # -1 means dont care
                num_positive = 0

                temp_overlaps = overlaps
                while np.count_nonzero(gt_flags) < num_valid_gt:
                    # ground-truth not fully matched
                    best_anchor = -1
                    best_gt = -1
                    max_overlap = 1e-6  # start with a very small positive overlap

                    max_iou = np.max(temp_overlaps)
                    if max_iou > max_overlap:
                        max_overlap = max_iou
                        best_anchor, best_gt = np.where(temp_overlaps == max_overlap)
                        best_anchor = best_anchor[0]
                        best_gt = best_gt[0]
                        temp_overlaps[:, best_gt] = -1
                        temp_overlaps[best_anchor, :] = -1

                    if int(best_anchor) == -1:
                        assert int(best_gt) == -1
                        break   # no more good match
                    else:
                        assert int(max_matches[best_anchor, 0]) == -1
                        assert int(max_matches[best_anchor, 1]) == -1
                        max_matches[best_anchor, 0] = max_overlap
                        max_matches[best_anchor, 1] = best_gt
                        num_positive += 1
                        # mark as visited
                        gt_flags[best_gt] = True
                        anchor_flags[best_anchor] = 1
                # end while

                assert self.overlap_threshold > 0
                # find positive matches based on overlaps
                max_iou = np.max(overlaps, axis=1)
                best_gt = np.argmax(overlaps, axis=1)

                max_matches[:, 0] = np.where(anchor_flags.flatten() == 1, max_matches[:, 0], max_iou)
                max_matches[:, 1] = np.where(anchor_flags.flatten() == 1, max_matches[:, 1], best_gt)

                overlap_inds = np.where((anchor_flags.flatten() != 1) & (max_iou > self.overlap_threshold))[0]
                num_positive += overlap_inds.size
                # mark as visited
                gt_flags[best_gt[overlap_inds]] = True
                anchor_flags[overlap_inds] = 1

                # for j in range(num_anchors):
                #     if int(anchor_flags[j]) == 1:
                #         continue    # already matched this anchor
                #
                #     if max_iou[j] > self.overlap_threshold:
                #         num_positive += 1
                #         # mark as visited
                #         gt_flags[best_gt[j]] = True
                #         anchor_flags[j] = 1

                if self.negative_mining_ratio > 0:
                    assert self.negative_mining_thresh > 0
                    num_negative = num_positive * self.negative_mining_ratio
                    if num_negative > (num_anchors - num_positive):
                        num_negative = num_anchors - num_positive

                    if num_negative > 0:
                        # use negative mining, pick "best" negative samples
                        bg_probs = []
                        for j in range(num_anchors):
                            if int(anchor_flags[j]) == 1:
                                continue # already matched this anchor
                            if max_matches[j, 0] < 0:
                                # not yet calculated
                                best_gt = -1
                                max_iou = -1.0

                                iou = np.max(overlaps[j])
                                if iou > max_iou:
                                    max_iou = iou
                                    best_gt = np.argmax(overlaps[j])

                                if int(best_gt) != -1:
                                    assert int(max_matches[j, 0]) == -1
                                    assert int(max_matches[j, 1]) == -1
                                    max_matches[j, 0] = max_iou
                                    max_matches[j, 1] = best_gt

                            if (max_matches[j, 0] < self.negative_mining_thresh) & \
                                (int(anchor_flags[j]) == -1):
                                # calculate class predictions
                                # max_val = cls_preds_per_batch[0, j] # background cls preds
                                max_val = np.max(cls_preds_per_batch[:, j])
                                # for k in range(1, num_class):
                                #     tmp = cls_preds_per_batch[k, j]
                                #     if tmp > max_val:
                                #         max_val = tmp

                                p_sum = np.sum(np.exp(cls_preds_per_batch[:, j] - max_val))
                                # for k in range(num_class):
                                #     tmp = cls_preds_per_batch[k, j]
                                #     p_sum += np.exp(tmp - max_val)

                                bg_prob = np.exp(cls_preds_per_batch[0, j] - max_val) / p_sum
                                # loss should be -log(x), but value does not matter, skip log
                                bg_probs += [bg_prob, j]
                        # end iterate anchors

                        bg_probs = np.reshape(np.array(bg_probs), newshape=(-1,2))
                        # default ascend order
                        neg_indx = np.lexsort((bg_probs[:, 1].flatten(), bg_probs[:, 0].flatten()))
                        bg_probs = bg_probs[neg_indx]

                        for i in range(int(num_negative)):
                            anchor_flags[int(bg_probs[i, 1])] = 0 # mark as negative sample

                else:
                    # use all negative samples
                    anchor_flags = np.where(anchor_flags.astype(np.int8) == 1, 1, 0)

                # assign training target
                fg_inds = np.where(anchor_flags.astype(np.int8) == 1)[0]
                bg_inds = np.where(anchor_flags.astype(np.int8) == 0)[0]

                # assign class target
                cls_target_per_batch[fg_inds] = labels_per_batch[max_matches[fg_inds, 1].astype(np.int8), 0] + 1
                cls_target_per_batch[bg_inds] = 0

                # assign bbox mask
                box_mask_per_batch[fg_inds, :] = 1

                # assign bbox target
                box_target_per_batch[fg_inds, :] = bbox_transform(anchors[fg_inds, :],
                                                                  gt_boxes[max_matches[fg_inds, 1].astype(np.int8), :],
                                                                  box_stds=(1.0, 1.0, 1.0, 1.0))

        # box_target, box_mask, cls_target = mx.nd.contrib.MultiBoxTarget(anchors, labels, class_preds,
        #                                                                     overlap_threshold=self.overlap_threshold,
        #                                                                     ignore_label=-1,
        #                                                                     negative_mining_ratio=self.negative_mining_ratio,
        #                                                                     minimum_negative_samples=0,
        #                                                                     negative_mining_thresh=self.negative_mining_thresh,
        #                                                                     variances=self.variances,
        #                                                                     name="multibox_target")

        # anchor_mask = box_mask.reshape(shape=(0, -1, 4))  # batchsize x num_anchors x 4
        # bb8_mask = mx.nd.repeat(data=anchor_mask, repeats=4, axis=2)  # batchsize x num_anchors x 16
        # # anchor_mask = mx.nd.mean(data=anchor_mask, axis=2, keepdims=False, exclude=False)
        #
        # anchors_in_use = mx.nd.broadcast_mul(lhs=anchor_mask, rhs=anchors)  # batchsize x num_anchors x 4
        #
        # # transform the anchors from [xmin, ymin, xmax, ymax] to [cx, cy, wx, hy]
        #
        # centerx = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1) +
        #            mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3)) / 2
        # centery = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2) +
        #            mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4)) / 2
        # width = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3) -
        #          mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1)) + 1e-8
        # height = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4) -
        #           mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2)) + 1e-8
        #
        # anchors_in_use_transformed = mx.nd.concat(centerx, centery, width, height, dim=2)   # batchsize x num_anchors x 4
        #
        # bb8_target = mx.nd.zeros_like(data=bb8_mask)    # batchsize x num_anchors x 16
        # bb8_label = mx.nd.slice_axis(data=labels, axis=2, begin=8, end=24)  # batchsize x 8 x 16
        #
        # # calculate targets for OCCLUSION dataset
        # for cid in range(1, 9):
        #     cid_target_mask = (cls_target == cid)
        #     cid_target_mask = cid_target_mask.reshape(shape=(0,-1,1))
        #     # cid_anchors_in_use_transformed = mx.nd.broadcast_mul(lhs=cid_target_mask, rhs=anchors_in_use_transformed)
        #     cid_anchors_in_use_transformed = mx.nd.where(condition=mx.nd.broadcast_to(cid_target_mask, shape=anchors_in_use_transformed.shape),
        #                                                 x=anchors_in_use_transformed,
        #                                                 y=mx.nd.zeros_like(anchors_in_use_transformed))
        #     cid_label_mask = (mx.nd.slice_axis(data=labels, axis=2, begin=0, end=1) == cid - 1)
        #     cid_bb8_label = mx.nd.broadcast_mul(lhs=cid_label_mask, rhs=bb8_label)
        #     # TODO: currently only support single instance per class, and clip by 0
        #     cid_bb8_label = mx.nd.sum(cid_bb8_label, axis=1, keepdims=True) # batchsize x 1 x 16
        #
        #     # substract center
        #     cid_bb8_target = mx.nd.broadcast_sub(cid_bb8_label, mx.nd.tile(  # repeat single element !! error
        #         data=mx.nd.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=0, end=2),
        #         reps=(1, 1, 8)))
        #     # divide by w and h
        #     cid_bb8_target = mx.nd.broadcast_div(cid_bb8_target, mx.nd.tile(
        #         data=mx.nd.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=2, end=4),
        #         reps=(1, 1, 8))) / 0.1  # variance
        #
        #     cid_bb8_target = mx.nd.where(condition=mx.nd.broadcast_to(cid_target_mask, shape=cid_bb8_target.shape),
        #                                  x=cid_bb8_target,
        #                                  y=mx.nd.zeros_like(cid_bb8_target))
        #     bb8_target = bb8_target + cid_bb8_target
        #
        # condition = bb8_mask > 0.5
        # bb8_target = mx.nd.where(condition=condition, x=bb8_target, y=mx.nd.zeros_like(data=bb8_target))
        #
        # bb8_target = bb8_target.flatten()  # batchsize x (num_anchors x 16)
        # bb8_mask = bb8_mask.flatten()  # batchsize x (num_anchors x 16)

        box_target = np.reshape(box_target, newshape=(batchsize, -1))
        box_mask = np.reshape(box_mask, newshape=(batchsize, -1))

        self.assign(out_data[0], req[0], mx.nd.array(box_target))
        self.assign(out_data[1], req[1], mx.nd.array(box_mask))
        self.assign(out_data[2], req[2], mx.nd.array(cls_target))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("training_targets")
class TrainingTargetsProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, overlap_threshold=0.5, negative_mining_ratio=3,
                 negative_mining_thresh=0.5, variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(TrainingTargetsProp, self).__init__(need_top_grad=False)
        self.overlap_threshold = float(overlap_threshold)
        self.negative_mining_ratio = float(negative_mining_ratio)
        self.negative_mining_thresh = float(negative_mining_thresh)
        self.variances = variances

    def list_arguments(self):
        return ['anchors', 'cls_preds', 'labels']

    def list_outputs(self):
        return ['box_target', 'box_mask', 'cls_target']

    def infer_shape(self, in_shape):
        anchors_shape = in_shape[0]
        data_shape = in_shape[1]
        label_shape = in_shape[2]

        box_target_shape = (data_shape[0], 4 * data_shape[2])
        box_mask_shape = (data_shape[0], 4 * data_shape[2])
        cls_target_shape = (data_shape[0], data_shape[2])

        return [anchors_shape, data_shape, label_shape], \
               [box_target_shape, box_mask_shape,
                cls_target_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return TrainingTargets(self.overlap_threshold, self.negative_mining_ratio, self.negative_mining_thresh, self.variances)
