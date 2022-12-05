# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Evaluation code of the DeepLesion dataset"""
import logging
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torchmetrics

from .detection_eval import sens_at_FP, avg_sens_only,  avg_iou_dice_conf
from .tagging_eval import compute_all_acc_wt, compute_thresholds, print_accs
from maskrcnn.config import cfg
from maskrcnn.utils.print_info import save_acc_to_file


def do_evaluation(
    dataset,
    predictions,
    is_validation
):
    logger = logging.getLogger("maskrcnn.inference")
    assert len(dataset) == len(predictions)

    fns = list(predictions.keys())
    if 'is_gt' in predictions[fns[0]]['result'].extra_fields.keys():
        for i, fn in enumerate(fns):
            is_gt = predictions[fn]['result'].get_field('is_gt')
            predictions[fn]['gt_result'] = predictions[fn]['result'][is_gt]
            predictions[fn]['result'] = predictions[fn]['result'][~is_gt]

    # NOTE: Breast CT - modification to include more metrics
    simple_sens = eval_DL_detection(predictions, logger, is_validation)

    # multi-label lesion tagging
    if cfg.MODEL.TAG_ON:
        tag_res = eval_DL_tagging(predictions, logger, is_validation)

    if cfg.MODEL.MASK_ON:
        seg_res = eval_BC_segmentation(predictions, logger)

    return  simple_sens

# NOTE: Breast CT pipeline addition - evaluation of segmentation by calculating DICE score 
def eval_BC_segmentation(predictions, logger):
    fns = sorted(predictions.keys())
    all_masks = [predictions[fn]['result'].get_field('mask').cpu().numpy() for fn in fns]
    all_gt_masks = [predictions[fn]['target'].get_field('masks').cpu().numpy() for fn in fns]
    all_scores = [predictions[fn]['result'].get_field('scores').numpy() for fn in fns]

    all_chosen_masks = []
    for score, mask in zip(all_scores, all_masks):
        if len(score)==0:
            all_chosen_masks.append(torch.zeros((1, 512, 512)))
        else:
            max_conf_idx = np.argmax(score)
            all_chosen_masks.append([mask[max_conf_idx]])

    all_chosen_masks = np.vstack(all_chosen_masks)
    all_chosen_masks = torch.from_numpy(all_chosen_masks.astype('uint8'))
    print(all_chosen_masks.shape)

    final_gt_masks = []
    for mask in all_gt_masks:
        gt_mask = np.array(mask)
        gt_mask = torch.from_numpy(gt_mask.astype('uint8'))
        gt_mask = torch.nn.functional.interpolate(gt_mask.unsqueeze(0), size=(512, 512))
        final_gt_masks.append(gt_mask[0])
    
    all_gt_masks = np.vstack(final_gt_masks)
    all_gt_masks = torch.from_numpy(all_gt_masks.astype('uint8'))
    print(all_gt_masks.shape)
    assert all_chosen_masks.shape == all_gt_masks.shape

    num_identical = 0
    for i in range(0, all_gt_masks.shape()[0]):
        if all_gt_masks[i] == all_chosen_masks[i]:
            num_identical += 1

    print("number of identical masks:")
    print(num_identical)

    torch_dice = torchmetrics.functional.dice(all_chosen_masks, all_gt_masks)
    logger.info('segmentation dice coefficient: %.4f', torch_dice)
    return torch_dice




def eval_DL_detection(predictions, logger, is_validation):
    fns = sorted(predictions.keys())
    all_boxes = [predictions[fn]['result'].bbox.numpy() for fn in fns]
    all_scores = [predictions[fn]['result'].get_field('scores').numpy() for fn in fns]
    all_boxes = [np.hstack((b, s.reshape((-1, 1)))) for (b, s) in zip(all_boxes, all_scores)]
    all_gts = [predictions[fn]['target'].bbox.cpu().numpy() for fn in fns]

    # detection
    logger.info('\nDetection accuracy:')

    # NOTE: for DeepLesion - not suited for BC where we assume 1 lesion per image
    logger.info('Sensitivity @ %s average FPs per image:', str(cfg.TEST.VAL_FROC_FP))
    det_res = sens_at_FP(all_boxes, all_gts, cfg.TEST.VAL_FROC_FP, cfg.TEST.IOU_TH)  # cls 0 is background
    logger.info(', '.join(['%.4f'%v for v in det_res]))
    logger.info('mean of %s: %.4f', str(cfg.TEST.VAL_FROC_FP[:4]), np.mean(det_res[:4]))

    # NOTE: Breast CT modification - addition of more metrics into the logs
    simple_sens = avg_sens_only(all_boxes, all_gts, cfg.TEST.IOU_TH)
    logger.info('average sensitivity: %.4f', simple_sens)
    iou_over_TPs, conf_over_TPs, avg_dice_over_TPs = avg_iou_dice_conf(all_boxes, all_gts, cfg.TEST.IOU_TH)
    logger.info('average IOU over TP instances: %.4f', iou_over_TPs)
    logger.info('average Dice coeff over TP instances: %.4f', avg_dice_over_TPs)
    logger.info('average confidences over TP instances: %.4f', conf_over_TPs)

    # detection accuracy per tag
    compute_det_acc_per_tag = cfg.TEST.COMPUTE_DET_ACC_PER_TAG
    if compute_det_acc_per_tag and is_validation and cfg.MODEL.TAG_ON:
        cfg.runtime_info.det_acc_per_tag = np.empty((cfg.runtime_info.num_tags, len(cfg.TEST.VAL_FROC_FP)), dtype=float)
        logger.info('\nComputing detection accuracy per tag:')
        for i, t in enumerate(cfg.runtime_info.tag_list):
            tag_mask = {fn: predictions[fn]['target'].get_field('tags')[:, i] == 1 for fn in fns}
            gts_per_tag = [predictions[fn]['target'].bbox[tag_mask[fn]].cpu().numpy() for fn in fns]
            det_res_per_tag = sens_at_FP(all_boxes, gts_per_tag, cfg.TEST.VAL_FROC_FP, cfg.TEST.IOU_TH)
            cfg.runtime_info.det_acc_per_tag[i, :] = det_res_per_tag
            if i % 10 == 0:
                logger.info('%d, ', i)
        logger.info('\n')

    return simple_sens


def eval_DL_tagging(predictions, logger, is_validation):
    logger.info('\nTagging accuracy:')

    fns = sorted(predictions.keys())
    prob_all = [predictions[fn]['gt_result'].get_field('tag_scores') for fn in fns]
    pred_all = [predictions[fn]['gt_result'].get_field('tag_predictions') for fn in fns]
    prob_all = torch.cat(prob_all).cpu().numpy()
    pred_all = torch.cat(pred_all).cpu().numpy()

    if is_validation:  # compute best class thresholds for next test
        target_all = [predictions[fn]['target'].get_field('tags') for fn in fns]
        target_all = torch.cat(target_all).cpu().numpy()
        pred_wt_all = target_all >= 0
        logger.info('mined tags from reports:')
        accs = compute_all_acc_wt(target_all, pred_all, prob_all, pred_wt_all)
        print_accs(accs, logger)
        save_acc_to_file(accs, 'val_mined')

        if cfg.TEST.TAG.CALIBRATE_TH:
            tag_sel_val = compute_thresholds(target_all, prob_all, pred_wt_all)
            cfg.runtime_info.tag_sel_val = torch.from_numpy(tag_sel_val).to(torch.float)

    else:  # doing final test using manual annotations
        # in the test set, only 500 hand-labeled tags are released
        assert 'manual_annot_test_tags' in predictions[fns[0]]['target'].extra_fields.keys(), \
            "Currently we evaluate tagging accuracy on manual_annot_test_tags, which is only in the test set of DeepLesion"
        target_all = [predictions[fn]['target'].get_field('manual_annot_test_tags') for fn in fns]
        target_all = torch.cat(target_all).cpu().numpy()
        pred_wt_all = target_all >= 0
        accs = compute_all_acc_wt(target_all, pred_all, prob_all, pred_wt_all)
        save_acc_to_file(accs, 'test_handlabeled')
        logger.info('hand-labeled tags:')
        print_accs(accs, logger)

    return accs['mean_auc']


def eval_DL_segmentation(predictions, logger):
    min_dists = []
    diam_errs = []
    fns = sorted(predictions.keys())
    for fn in fns:
        d = predictions[fn]
        # coordinate offset caused by IMG_DO_CLIP is not considered
        spacing = d['info']['spacing']
        im_scale = d['info']['im_scale']
        gt_recists = d['info']['recists']
        gt_recists_mm = gt_recists/im_scale*spacing

        if not cfg.TEST.EVAL_SEG_TAG_ON_GT:
            raise NotImplementedError
        contours = d['gt_result'].get_field('contour_mm')
        predicted_diameters = d['gt_result'].get_field('diameter_mm')
        for gt_idx in range(len(gt_recists)):
            contour = contours[gt_idx][contours[gt_idx][:,0]>0, :]
            gt_recist = gt_recists_mm[gt_idx]
            min_dists.append(compute_recist_contour_dist(contour, gt_recist))

            # compute the error of lesion diameter estimation
            predicted_diameter = predicted_diameters[gt_idx]
            diam_errs.append(compute_diameter_error(predicted_diameter, gt_recist))

    # print(min_dists)
    # print(diam_errs)
    min_dists_avg = np.mean(min_dists)
    diam_errs_avg = np.mean(diam_errs)
    logger.info('avg min distance (mm) from groundtruth recist points to predicted contours in GT boxes:\n'
                'error of lesion diameter (mm) estimated from predicted contours in GT boxes:\n'
                '%.4f+-%.4f, %.4f+-%.4f',
                min_dists_avg, np.std(min_dists), diam_errs_avg, np.std(diam_errs))
    # print(np.sort(diam_errs)[::10])
    return np.mean([min_dists_avg, diam_errs_avg])


def compute_recist_contour_dist(contour, recist):
    """Avg min distance (mm) from groundtruth recist points to predicted contours"""
    recist = recist.reshape((4, 2))
    D = cdist(recist, contour)
    d4 = D.min(axis=1)
    return d4.mean()


def compute_diameter_error(predicted_diameter, gt_recist):
    """Error of lesion diameter (mm) estimated from predicted contours"""
    gt_recist = gt_recist.reshape((4,2))
    gt_diameters = np.sqrt(np.sum((gt_recist[::2] - gt_recist[1::2]) ** 2, axis=1))
    error = np.mean([np.abs(gt_diameters.max() - predicted_diameter.max()),
                     np.abs(gt_diameters.min() - predicted_diameter.min())])
    return error



