# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Evaluation codes for the detection task"""
import numpy as np
from scipy import interpolate

from maskrcnn.config import cfg


def sens_at_FP(boxes_all, gts_all, avgFP, iou_th):
    """compute the sensitivity at avgFP (average FP per image)"""
    sens, fp_per_img, _ = FROC(boxes_all, gts_all, iou_th)
    avgFP_in = [a for a in avgFP if a <= fp_per_img[-1]] # avgFP quantities <= actual avgFP
    avgFP_out = [a for a in avgFP if a > fp_per_img[-1]] # avgFP quantities > actual avgFP --> == 1
    f = interpolate.interp1d(fp_per_img, sens)
    res = np.hstack([f(np.array(avgFP_in)), np.ones((len(avgFP_out, )))*sens[-1]])
    return res

# NOTE: Breast CT function [detection task] - gets only average sensitivity across all matched bboxes
def avg_sens_only(boxes_all, gts_all, iou_th):
    sens, _, _ = FROC(boxes_all, gts_all, iou_th)
    return sens[-1]

# NOTE: Breast CT function [detection task] - gets average iou, confidence and dice coefficient
def avg_iou_dice_conf(boxes_all, gts_all, iou_th):
    _, _, results = FROC(boxes_all, gts_all, iou_th)
    return results

# NOTE: Breast CT modified FROC -> to get average sensitivity, iou, confidence and dice coefficient
def FROC(boxes_all, gts_all, iou_th):
    """Compute the Free ROC curve, for single class only"""
    nImg = len(boxes_all)
    # to denote which boxes are generated from the same image
    img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    boxes_cat = np.vstack(boxes_all)

    # sort boxes_cat according to highers to lowest conf score
    scores = boxes_cat[:, -1]
    scores_sorted = sorted(scores, reverse=True)
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs[ord]

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    IOUs = 0 
    confs = 0
    tps = []
    fps = []
    for i in range(len(boxes_cat)):
        # gets IOU areas (in the case of multiple ground truths overlapping)
        overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
        if len(overlaps) == 0 or overlaps.max() < iou_th:
            # FP = no overlap with gt or overlap < 0.5 (test iou threshold)
            nMiss += 1
        else:
            for j in range(len(overlaps)):
                if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
                    # TP = overlap > 0.5 and gt has not been matched yet
                    hits[img_idxs[i]][j] = True
                    # match gt to a prediction
                    nHits += 1
                    IOUs += overlaps[j]
                    confs += scores_sorted[i]

        tps.append(nHits)
        fps.append(nMiss)

    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt # nGt = num ground truths = TP + FN
    # cumulative sensitivity (over all images)
    # note: for BreastCT with only 1 gt per image - sens increases at each image by either 0 or 1
    fp_per_img = np.array(fps, dtype=float) / nImg 
    # cumulative FPs over the number of images - fp_per_img[-1] == total no. of fp over images
    avg_IOU = IOUs/nHits
    # average IOU across the true positives
    avg_conf = confs/nHits
    # average confidence across the true positives
    avg_dice = (2*avg_IOU)/(avg_IOU+1)

    return sens, fp_per_img, [avg_IOU, avg_conf, avg_dice]


def IOU(box1, gts):
    """compute overlaps over intersection"""
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)

    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps
