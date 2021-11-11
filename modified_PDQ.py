"""
Link to original code: https://github.com/jskinn/rvchallenge-evaluation/blob/master/pdq.py
This code transforms cv-version into point-cloud version
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import gmean
from bbox_IoU import bbox_IoU


_SMALL_VAL = 1e-3   # 1e-14


class PDQ(object):
    """
    Class for calculating PDQ for a set of timestamps
    """
    def __init__(self):
        """
        Initialisation function for PDQ evaluator.
        """
        super(PDQ, self).__init__()
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_fg_quality = 0.0
        self._tot_bg_quality = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0
        self._det_evals = []
        self._gt_evals = []

    def add_img_eval(self, gt_seg_mat, gt_label_vec, det_seg_heatmap_mat, det_label_prob_mat, gt_bbox, det_bbox,
                     bbox_comparison):
        """
        calculate a single timestamp's detections and ground-truth to the overall evaluation analysis.

        :param gt_seg_mat: list of GroundTruthInstance objects present in the given timestamp, Boolean type.
        :param gt_label_vec: the corresponding gt-object label, int type.
        :param det_seg_heatmap_mat: list of DetectionInstance objects provided for the given timestamp, 0-1 type.
        :param det_label_prob_mat: the corresponding detected-object lable, one-hot coding.
        :param gt_bbox: the bbox of the gt-object
        :param det_bbox: the bbox of the detected object
        :param bbox_comparison: if Ture, calculate IoU, if False, no IoU

        :return: None
        """
        results = _calc_qual_img(gt_seg_mat, gt_label_vec, det_seg_heatmap_mat, det_label_prob_mat, gt_bbox, det_bbox,
                                 bbox_comparison)
        self._tot_overall_quality += results['overall']
        self._tot_spatial_quality += results['spatial']
        self._tot_label_quality += results['label']
        self._tot_fg_quality += results['fg']
        self._tot_bg_quality += results['bg']
        self._tot_TP += results['TP']
        self._tot_FP += results['FP']
        self._tot_FN += results['FN']
        self._det_evals.append(results['img_det_evals'])
        self._gt_evals.append(results['img_gt_evals'])

    def reset(self):
        """
        Reset all internally stored evaluation measures to zero.
        :return: None
        """
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_fg_quality = 0.0
        self._tot_bg_quality = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0
        self._det_evals = []
        self._gt_evals = []

    def get_spatial_score(self):
        """
        Get the spatial quality score for all assigned detections in all timestamps analysed at the current sequence.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average spatial quality of every assigned detection, which is averaged Qs.
        """
        if self._tot_TP > 0.0:
            return self._tot_spatial_quality / float(self._tot_TP)
        return 0.0

    def get_label_score(self):
        """
        Get the average label quality score for all assigned detections in all timestamps analysed at the sequence.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average label quality of every assigned detection, which is averaged Ql.
        """
        if self._tot_TP > 0.0:
            return self._tot_label_quality / float(self._tot_TP)
        return 0.0

    def get_pdq_score(self):
        """
        Get the average overall pairwise quality score for all assigned detections
        in all timestamps analysed at the current sequence
        Note that this is averaged over the the full set of TPs, FPs, and FNs.
        :return: average final pdq score of the whole sequence
        """
        tot_pairs = self._tot_TP + self._tot_FP + self._tot_FN
        return self._tot_overall_quality / tot_pairs

    def get_fg_quality_score(self):
        """
        Get the average foreground spatial quality score for all assigned detections
        in all timestamps analysed at the current sequence
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average overall pairwise foreground spatial quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_fg_quality / float(self._tot_TP)
        return 0.0

    def get_bg_quality_score(self):
        """
        Get the average background spatial quality score for all assigned detections
        in all timestamps analysed at the current sequence
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average overall pairwise background spatial quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_bg_quality/ float(self._tot_TP)
        return 0.0

    def get_assignment_counts(self):
        """
        Get the total number of TPs, FPs, and FNs across all timestamps analysed at the current sequence
        :return: tuple containing (TP, FP, FN)
        """
        return self._tot_TP, self._tot_FP, self._tot_FN


def _safe_log(mat):
    """
    Function for performing safe log (avoiding infinite loss) for all elements of a given matrix by adding _SMALL_VAL
    to all elements.
    :param mat: matrix of values
    :return: safe log of matrix elements
    """
    return np.log(mat + _SMALL_VAL)


def _calc_bg_loss(bg_seg_mat, det_seg_heatmap_mat):
    """
    Calculate the background pixel loss for all detections on all ground truth objects for a given timestamp.
    :param bg_seg_mat: (g x points) vectorized background masks for each gt object in the timestamp, Boolean.
    :param det_seg_heatmap_mat: (d x points) vectorized segmented heatmaps for each detection in the timestamp, 0-1.
    :return: bg_loss_sum: (g x d) total background loss between each of the g ground truth objects and d detections.
    """
    bg_log_loss_mat = _safe_log(1-det_seg_heatmap_mat) * (det_seg_heatmap_mat > 0)
    bg_loss_sum = np.dot(bg_seg_mat, bg_log_loss_mat.T)
    return bg_loss_sum


def _calc_fg_loss(gt_seg_mat, det_seg_heatmap_mat):
    """
    Calculate the foreground pixel loss for all detections on all ground truth objects for a given timestamp.
    :param gt_seg_mat: (g x points) vectorized segmentation masks for each gt object in the timestamp, Boolean.
    :param det_seg_heatmap_mat: (d x points) vectorized segmented heatmaps for each detection in the timestamp, 0-1.
    :return: fg_loss_sum: (g x d) total foreground loss between each of the g ground truth objects and d detections.
    """
    log_heatmap_mat = _safe_log(det_seg_heatmap_mat)
    fg_loss_sum = np.dot(gt_seg_mat, log_heatmap_mat.T)
    return fg_loss_sum


def _calc_spatial_qual(fg_loss_sum, bg_loss_sum, num_fg_pixels_vec):
    """
    Calculate the spatial quality for all detections on all ground truth objects for a given timestamp.
    :param fg_loss_sum: (g x d) total foreground loss between each of the g ground truth objects and d detections.
    :param bg_loss_sum: (g x d) total background loss between each of the g ground truth objects and d detections.
    :param num_fg_pixels_vec: (g x 1) number of fg pixels for each of the g ground truth objects.
    :return: spatial_quality: (g x d) spatial quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    total_loss = fg_loss_sum + bg_loss_sum
    loss_per_gt_pixel = total_loss/num_fg_pixels_vec
    spatial_quality = np.exp(loss_per_gt_pixel)

    # Deal with tiny floating point errors or tiny errors caused by _SMALL_VAL that prevent perfect 0 or 1 scores
    spatial_quality[np.isclose(spatial_quality, 0)] = 0
    spatial_quality[np.isclose(spatial_quality, 1)] = 1
    return spatial_quality


def _calc_label_qual(gt_label_vec, det_label_prob_mat):
    """
    Calculate the label quality for all detections on all ground truth objects for a given timestamp.
    :param gt_label_vec:  g, containing the class label as an integer for each object.
    :param det_label_prob_mat: (d x c) numpy array of label probability scores across all c classes
    for each of the d detections.
    :return: label_qual_mat: (g x d) label quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    """

    label_qual_mat = det_label_prob_mat[:, gt_label_vec].T.astype(np.float32)
    label_qual_mat = np.squeeze(label_qual_mat, axis=0)
    return label_qual_mat


def _calc_overall_qual(label_qual, spatial_qual):
    """
    Calculate the overall quality (pPDQ) for all detections on all ground truth objects for a given timestamp
    :param label_qual: (g x d) label quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    :param spatial_qual: (g x d) spatial quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    :return: overall_qual_mat: (g x d)  pPDQ between zero and one for each possible combination of
    g ground truth objects and d detections.
         0: this combination is totally wrong
         1: this combination matches perfect
    """
    combined_mat = np.dstack((label_qual, spatial_qual))
    # Calculate the geometric mean between label quality and spatial quality.
    # Note we ignore divide by zero warnings here for log(0) calculations internally.
    with np.errstate(divide='ignore'):
        overall_qual_mat = gmean(combined_mat, axis=2)
    return overall_qual_mat


def _gen_cost_tables(gt_seg_mat, gt_label_vec, det_seg_heatmap_mat, det_label_prob_mat):
    """
    Generate the cost tables containing the cost values (1 - quality) for each combination of ground truth objects and
    detections within a given timestamp.
    :param gt_seg_mat: (g x points) gt-objects present in the given timestamp, Boolean type.
    :param gt_label_vec: vector g, the corresponding gt-object label, int type.
    :param det_seg_heatmap_mat: (d x points) detected-objects provided for the given timestamp, 0-1 type.
    :param det_label_prob_mat: (d x c) the corresponding detected-object lable, one-hot coding.
    :return: dictionary of (g x d) cost tables for each combination of ground truth objects and detections.
    Note that all costs are simply (1 - quality) scores (required for Hungarian algorithm implementation)
    Format: {'overall': overall pPDQ cost table, 'spatial': spatial quality cost table,
    'label': label quality cost table, 'fg': foreground quality cost table, 'bg': background quality cost table}
    """
    # Initialise cost tables
    n_pairs = max(len(gt_seg_mat), len(det_seg_heatmap_mat))
    overall_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    spatial_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    label_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    bg_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    fg_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    num_gt_objects = len(gt_seg_mat)
    num_det_objects = len(det_seg_heatmap_mat)

    # Generate bg segmentation mask & fg pixels vector from gt-objects
    bg_seg_mat = np.logical_not(gt_seg_mat)
    num_fg_pixels = []
    for i in range(len(gt_seg_mat)):
        count = 0
        for j in range(len(gt_seg_mat[i])):
            if gt_seg_mat[i][j]:
                count += 1
        num_fg_pixels.append(count)
    num_fg_pixels = np.array(num_fg_pixels)
    num_fg_pixels_vec = num_fg_pixels.reshape((num_gt_objects,1))

    # Calculate spatial and label qualities, which are Qs and Ql
    label_qual_mat = _calc_label_qual(gt_label_vec, det_label_prob_mat)
    fg_loss = _calc_fg_loss(gt_seg_mat, det_seg_heatmap_mat)
    bg_loss = _calc_bg_loss(bg_seg_mat, det_seg_heatmap_mat)
    spatial_qual = _calc_spatial_qual(fg_loss, bg_loss, num_fg_pixels_vec)

    # Calculate foreground quality
    fg_loss_per_gt_pixel = fg_loss/num_fg_pixels_vec
    fg_qual = np.exp(fg_loss_per_gt_pixel)
    fg_qual[np.isclose(fg_qual, 0)] = 0
    fg_qual[np.isclose(fg_qual, 1)] = 1

    # Calculate background quality
    bg_loss_per_gt_pixel = bg_loss/num_fg_pixels_vec
    bg_qual = np.exp(bg_loss_per_gt_pixel)
    bg_qual[np.isclose(bg_qual, 0)] = 0
    bg_qual[np.isclose(bg_qual, 1)] = 1

    # Generate the overall cost table (1 - overall quality(pPDQ))
    overall_cost_table[:num_gt_objects, :num_det_objects] -= _calc_overall_qual(label_qual_mat, spatial_qual)

    # Generate the spatial and label cost tables
    spatial_cost_table[:num_gt_objects, :num_det_objects] -= spatial_qual
    label_cost_table[:num_gt_objects, :num_det_objects] -= label_qual_mat

    # Generate foreground and background cost tables
    fg_cost_table[:num_gt_objects, :num_det_objects] -= fg_qual
    bg_cost_table[:num_gt_objects, :num_det_objects] -= bg_qual

    return {'overall': overall_cost_table, 'spatial': spatial_cost_table, 'label': label_cost_table,
            'fg': fg_cost_table, 'bg': bg_cost_table}


def _calc_qual_img(gt_seg_mat, gt_label_vec, det_seg_heatmap_mat, det_label_prob_mat, gt_bbox, det_bbox,
                   bbox_comparison):
    """
    Calculates the sum of qualities for the best matches between gt objects and detections for one timestamp
    Each gt object can only be matched to a single detection and vice versa as an object-detection pair.
    Note that if a gt object or detection does not have a match, the quality is counted as zero.
    Any provided detection with a zero-quality match will be counted as a false positive (FP).
    Any ground-truth object with a zero-quality match will be counted as a false negative (FN).
    All other matches are counted as "true positives" (TP)
    If there are no gt objects or detections for the given timestamp, the system returns zero and this timestamp
    will not contribute to average_PDQ.

    :param gt_seg_mat: (g x points) gt-objects present in the given timestamp, Boolean type.
    :param gt_label_vec: vector g, the corresponding gt-object label, int type.
    :param det_seg_heatmap_mat: (d x points) detected-objects provided for the given timestamp, 0-1 type.
    :param det_label_prob_mat: (d x c) the corresponding detected-object label, one-hot coding.
    :param gt_bbox: the bbox of the gt-object
    :param det_bbox: the bbox of the detected object
    :param bbox_comparison: if Ture, calculate IoU, if False, no IoU

    :return: results dictionary containing total overall PDQ (needed to scale),
    total spatial quality on positively assigned detections Qs (needed to scale),
    total label quality on positively assigned detections Ql (needed to scale),
    total foreground quality on positively assigned detections (needed to scale),
    total background quality on positively assigned detections (needed to scale),
    number of TPs, number of FPs, number FNs,
    detection evaluation summary, and ground-truth evaluation summary for for the given image.
    Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
    'fg':<tot_tp_foreground_quality>, 'bg':<tot_tp_background_quality>, 'TP': <num_true_positives>,
    'FP': <num_false_positives>, 'FN': <num_false_positives>, 'img_det_evals':<detection_evaluation_summary>,
    'img_gt_evals':<ground-truth_evaluation_summary>}
    """

    img_det_evals = []
    img_gt_evals = []
    # if there are no detections or gt instances respectively the quality is zero
    if (len(gt_label_vec) == 0) or (len(det_label_prob_mat) == 0):
        FP = 0
        FN = 0
        # no gt objects, but detections exist
        if det_label_prob_mat.any() > 0:
            img_det_evals = [{"det_id": idx, "gt_id": None, "matched": False,
                              "pPDQ": 0.0, "spatial": 0.0, "label": 0.0, "correct_class": None,
                              'bg': 0.0, 'fg': 0.0}
                             for idx in range(len(det_seg_heatmap_mat))]
            FP = len(det_label_prob_mat)
        # gt objects exist, no detections
        if len(gt_label_vec) > 0:
            for gt_idx, gt_instance in enumerate(gt_seg_mat):
                gt_eval_dict = {"det_id": None, "gt_id": gt_idx, "matched": False,
                                "pPDQ": 0.0, "spatial": 0.0, "label": 0.0, "correct_class": gt_label_vec[gt_idx],
                                'fg': 0.0, 'bg': 0.0}
                FN += 1
                img_gt_evals.append(gt_eval_dict)

        return {'overall': 0.0, 'spatial': 0.0, 'label': 0.0, 'fg': 0.0, 'bg': 0.0, 'TP': 0, 'FP': FP,
                'FN': FN, "img_det_evals": img_det_evals, "img_gt_evals": img_gt_evals}

    # For each possible pairing, calculate the quality of that pairing and convert it to a cost
    # to enable use of the Hungarian algorithm.
    cost_tables = _gen_cost_tables(gt_seg_mat, gt_label_vec, det_seg_heatmap_mat, det_label_prob_mat)

    # Use the Hungarian algorithm with the cost table to find the best match between gt object and detection
    # (lowest overall cost representing highest overall pairwise quality)
    row_idxs, col_idxs = linear_sum_assignment(cost_tables['overall'])

    # Transform the loss tables back into quality tables with values between 0 and 1
    overall_quality_table = 1 - cost_tables['overall']
    spatial_quality_table = 1 - cost_tables['spatial']
    label_quality_table = 1 - cost_tables['label']
    fg_quality_table = 1 - cost_tables['fg']
    bg_quality_table = 1 - cost_tables['bg']

    # Go through all optimal assignments and summarize all pairwise statistics
    # Calculate the number of TPs, FPs, and FNs for the timestamp during the process
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for match_idx, match in enumerate(zip(row_idxs, col_idxs)):
        row_id, col_id = match
        det_eval_dict = {"det_id": int(col_id), "gt_id": int(row_id), "matched": True,
                         "pPDQ": float(overall_quality_table[row_id, col_id]),
                         "spatial": float(spatial_quality_table[row_id, col_id]),
                         "label": float(label_quality_table[row_id, col_id]),
                         'fg': float(fg_quality_table[row_id, col_id]),
                         'bg': float(bg_quality_table[row_id, col_id]),
                         "correct_class": None}
        gt_eval_dict = det_eval_dict.copy()
        if overall_quality_table[row_id, col_id] > 0:
            det_eval_dict["correct_class"] = gt_label_vec[row_id]
            gt_eval_dict["correct_class"] = gt_label_vec[row_id]
            if row_id < len(gt_seg_mat):
                true_positives += 1
                # bbox IoU calculation
                if bbox_comparison:
                    iou = bbox_IoU(gt_bbox[row_id],det_bbox[col_id])
                    print('IoU:', iou)
            else:
                # Set the overall quality table value to zero so it does not get included in final total
                overall_quality_table[row_id, col_id] = 0.0
            img_det_evals.append(det_eval_dict)
            img_gt_evals.append(gt_eval_dict)
        else:
            if row_id < len(gt_seg_mat):
                gt_eval_dict["correct_class"] = gt_label_vec[row_id]
                gt_eval_dict["det_id"] = None
                gt_eval_dict["matched"] = False
                false_negatives += 1
                img_gt_evals.append(gt_eval_dict)

            if col_id < len(det_seg_heatmap_mat):
                det_eval_dict["gt_id"] = None
                det_eval_dict["matched"] = False
                false_positives += 1
                img_det_evals.append(det_eval_dict)

    # Calculate the sum of quality at the best matching pairs to calculate total qualities for the image
    tot_overall_img_quality = np.sum(overall_quality_table[row_idxs, col_idxs])
    # print('overall_quality_table',overall_quality_table,'tot_overall_img_quality',tot_overall_img_quality)

    # Force spatial and label qualities to zero for total calculations as there is no actual association between
    # detections and therefore no TP when this is the case.
    spatial_quality_table[overall_quality_table == 0] = 0.0
    label_quality_table[overall_quality_table == 0] = 0.0
    fg_quality_table[overall_quality_table == 0] = 0.0
    bg_quality_table[overall_quality_table == 0] = 0.0

    # Calculate the sum of spatial and label qualities only for TP samples
    tot_tp_spatial_quality = np.sum(spatial_quality_table[row_idxs, col_idxs])
    tot_tp_label_quality = np.sum(label_quality_table[row_idxs, col_idxs])
    tot_tp_fg_quality = np.sum(fg_quality_table[row_idxs, col_idxs])
    tot_tp_bg_quality = np.sum(bg_quality_table[row_idxs, col_idxs])

    # Sort the evaluation details to match the order of the detections and ground truths
    img_det_eval_idxs = [det_eval_dict["det_id"] for det_eval_dict in img_det_evals]
    img_gt_eval_idxs = [gt_eval_dict["gt_id"] for gt_eval_dict in img_gt_evals]
    img_det_evals = [img_det_evals[idx] for idx in np.argsort(img_det_eval_idxs)]
    img_gt_evals = [img_gt_evals[idx] for idx in np.argsort(img_gt_eval_idxs)]

    return {'overall': tot_overall_img_quality, 'spatial': tot_tp_spatial_quality, 'label': tot_tp_label_quality,
            'fg': tot_tp_fg_quality, 'bg': tot_tp_bg_quality,
            'TP': true_positives, 'FP': false_positives, 'FN': false_negatives,
            'img_gt_evals': img_gt_evals, 'img_det_evals': img_det_evals}
