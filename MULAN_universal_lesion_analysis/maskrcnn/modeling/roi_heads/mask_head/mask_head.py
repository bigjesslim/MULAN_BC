# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn.structures.bounding_box import BoxList
from maskrcnn.modeling.roi_heads.mask_head.inference import Masker
from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator
from maskrcnn.config import cfg


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self):
        super(ROIMaskHead, self).__init__()
        self.feature_extractor = make_roi_mask_feature_extractor()
        self.predictor = make_roi_mask_predictor()
        self.post_processor = make_roi_mask_post_processor()
        self.loss_evaluator = make_roi_mask_loss_evaluator()

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        # NOTE: Breast CT pipeline modification
        # -> converting features from network layer to actual mask (codes derived from postprocessor)
        mask_prob = mask_logits.sigmoid()
        num_masks = mask_logits.shape[0] # 52
        labels = [bbox.get_field("labels") for bbox in proposals]

        labels = torch.cat(labels).to(torch.int64)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]
        boxes_per_image = [len(box) for box in proposals]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
        mask_prob = masker(mask_prob, proposals)

        loss_mask = self.loss_evaluator(proposals, mask_prob, targets)

        return mask_logits, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head():
    return ROIMaskHead()
