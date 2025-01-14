# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import torchmetrics

from maskrcnn.layers import smooth_l1_loss
from maskrcnn.modeling.matcher import Matcher
from maskrcnn.structures.boxlist_ops import boxlist_iou
from maskrcnn.modeling.utils import cat
from maskrcnn.config import cfg
from maskrcnn.structures.segmentation_mask import SegmentationMask


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix) # for multiple lesions

        # TODO: change for multiple lesions
        #matched_idxs = torch.argmax(match_quality_matrix[:-1])
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[torch.tensor(0).unsqueeze(0)]
        #matched_targets = target[ matched_idxs.clamp(min=0)] # for multiple lesions
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        preds = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if len(targets_per_image) > 0:
                matched_targets = self.match_targets_to_proposals(
                    proposals_per_image, targets_per_image
                )
                matched_idxs = matched_targets.get_field("matched_idxs")

                labels_per_image = matched_targets.get_field("labels")
                labels_per_image = labels_per_image.to(dtype=torch.int64)

                # this can probably be removed, but is left here for clarity
                # and completeness
                # neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
                # labels_per_image[neg_inds] = 0

                # mask scores are only computed on positive samples
                positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

                segmentation_masks = matched_targets.get_field("masks")
                segmentation_masks = segmentation_masks[positive_inds]

                positive_proposals = proposals_per_image[positive_inds]

                # NOTE: assumes only one gt 
                preds.append(positive_inds[0].item())

                if isinstance(segmentation_masks, SegmentationMask):
                    masks_per_image = project_masks_on_boxes(
                        segmentation_masks, positive_proposals, self.discretization_size
                    )
                else:
                    # NOTE: BreastCT pipeline additions - interpolating predicted masks to (512,512) for comparison with GT masks
                    segmentation_masks = segmentation_masks.unsqueeze(0)
                    segmentation_masks = torch.nn.functional.interpolate(segmentation_masks, size=(512, 512))
                    segmentation_masks = segmentation_masks[0]
                    device = proposals[0].bbox.device
                    masks_per_image = segmentation_masks.to(device, dtype=torch.float32)
                    # NOTE: End of BreastCT pipeline additions
            else:
                labels_per_image = torch.zeros(0, dtype=torch.int64).to(proposals_per_image.bbox.device)
                masks_per_image = torch.zeros(0, self.discretization_size, self.discretization_size,
                                              dtype=torch.float32).to(proposals_per_image.bbox.device)

            labels.append(labels_per_image)
            masks.append(masks_per_image)
        
        return labels, masks, preds  # NOTE: BreastCT pipeline addition of 'preds' variable to return processed prediction masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets, preds = self.prepare_targets(proposals, targets) # NOTE: BreastCT pipeline addition of 'preds' variable to return processed prediction masks

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0) 

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        # NOTE: BreastCT pipeline calculation of segmentation DICE loss for training
        pred_matched_masks = []
        mask_targets = mask_targets.int().to(device='cuda:0')

        for i in range(len(mask_logits)):
            pred_matched_masks.append(mask_logits[i][preds[i]].unsqueeze(0).clone())

        mask_logits = torch.vstack(pred_matched_masks)
        mask_logits = mask_logits.int().to(device='cuda:0')
        
        ## manual computation of DICE loss for BreastCT pipeline - removed
        # total_mask_loss = 0
        # for i in range(len(mask_targets)):
        #     mask_logit = mask_logits[i].unsqueeze(0).to(device='cuda:0')
        #     mask_target = mask_targets[i].unsqueeze(0)
        #     print("final shapes")
        #     print(mask_logit.shape)
        #     print(mask_target.shape)
        #     mask_loss = dice_loss( mask_logit, mask_target)
        #     total_mask_loss += mask_loss
        # mask_loss = dice_loss(mask_logits, mask_targets)
        # mask_loss = total_mask_loss/len(mask_targets)

        torch_dice = torchmetrics.functional.dice(mask_logits, mask_targets)
        mask_loss = 1-torch_dice

        # NOTE: End of BreastCT pipeline modification

        # mask_loss = F.binary_cross_entropy_with_logits(
        #     mask_logits[positive_inds, labels_pos], mask_targets
        # )
        return mask_loss


def make_roi_mask_loss_evaluator():
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
    )

    return loss_evaluator


# we found Dice loss works better than CE loss
def dice_loss(input, target):
    smooth = 0.01

    # # use all smp as a whole
    # iflat = input.sigmoid().view(-1)  # compute all samples as a whole??
    # tflat = target.view(-1)
    # intersection = (iflat * tflat).sum()
    #
    # return 1 - ((2. * intersection + smooth) /
    #             (iflat.sum() + tflat.sum() + smooth))

    # use avg of each smp
    input_dims = input.dim()
    target_dims = target.dim()
    assert input_dims == target_dims
    dims = (input_dims-2,input_dims-1)

    input = input.sigmoid()
    intersection = (input * target).sum(dim=dims)

    dice = ((2. * intersection + smooth) /
                (input.sum(dim=dims) + target.sum(dim=dims) + smooth))
    return 1 - dice.mean()
