# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""The DeepLesion dataset loader, include box, tag, and masks"""
import torch
import torchvision
import numpy as np
import os
import csv
import logging
import json
import nrrd
import random
import math
import pickle
from sklearn.model_selection import train_test_split

from maskrcnn.data.datasets.load_ct_img import load_prep_img
from maskrcnn.structures.bounding_box import BoxList
from maskrcnn.structures.segmentation_mask import SegmentationMask
from maskrcnn.config import cfg
from maskrcnn.data.datasets.DeepLesion_utils import load_tag_dict_from_xlsfile, gen_mask_polygon_from_recist, load_lesion_tags
from maskrcnn.data.datasets.DeepLesion_utils import gen_parent_list, gen_exclusive_list, gen_children_list

# NOTE: BreastCT.py is a new script cloned and modified from DeepLesion.py
# Its function is to allow for the dataset of Breast CT Scans of NCC Singapore (in the form of NRRD files) 
# to be loaded into a pytorch dataset object. 

class BreastCTDataset(object):

    def __init__(
        self, split, data_dir, ann_file, transforms=None
    ):
        self.transforms = transforms
        self.split = split
        self.data_path = data_dir
        self.classes = ['__background__',  # always index 0
                        'lesion']
        self.num_classes = len(self.classes)
        
        # NOTE: Breast CT modification for dataset initialization 
        # 1. Get persisted info of - patients + lesion slice indices per patient
        # MUST REFRESH persisted info when data is modified or if a different dataset is being used
        # TODO: Create script to parse through dataset and refresh info 
        dbfile = open('/home/tester/jessica/MULAN_BC/MULAN_universal_lesion_analysis/maskrcnn/data/datasets/lesion_slice_dict.pkl', 'rb')     
        lesion_slice_dict = pickle.load(dbfile)
        self.image_fn_list = lesion_slice_dict.keys()
        self.image_fn_list = list(self.image_fn_list)
        self.image_fn_list = sorted(self.image_fn_list, reverse=True) # fixes order --> permanent train-val-test set

        # 2. Dividing patients into train-val-test (60-20-20)
        train_fn_list, test_fn_list = train_test_split(self.image_fn_list, shuffle=False, test_size=0.2)
        train_fn_list, val_fn_list = train_test_split(train_fn_list, shuffle=False, test_size=0.25) 

        # 3. Assign train-val-test set based on configurations
        # "split" variable is assigned upon initialization call
        if self.split == 'train':
            self.image_fn_list = train_fn_list
        elif self.split == 'test':
            self.image_fn_list = test_fn_list
        elif self.split == 'val':
            self.image_fn_list = val_fn_list

        # 4. Use lesion_slice_dict to get slices for each file with lesions 
        # -> which forms image_fn_list = the list of CT slices (filename + slice index) -> forms the dataset
        # regenerate new lesion_slice dict for new dataset using gen_lesion_slice_dict.py
        lesion_ct_slices = []
        for image_fn in self.image_fn_list:
            for i in lesion_slice_dict[image_fn]:
                lesion_ct_slices.append([image_fn, i])

        self.image_fn_list = lesion_ct_slices
        # NOTE: End of Breast CT modifications 

        self.num_images = len(self.image_fn_list)
        self.logger = logging.getLogger(__name__)
        self.logger.info('DeepLesion %s num_images: %d' % (split, self.num_images))
    
    # NOTE: Breast CT modification - removed tag processing functions as this dataset has no tags

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, info).
        """
        
        # NOTE: Breast CT modification for getting individual CT slices 
        # 1. get image fn and slice idx from image_fn_list
        image_fn = self.image_fn_list[index][0]
        slice_idx = self.image_fn_list[index][1]

        # 2. Get and preprocess image nrrd data
        image_data =  nrrd.read(self.data_path + "/" + image_fn) 
        is_train = self.split =='train'

        assert slice_idx < len(image_data[0]) # check that slice_idx does not exceed the volume of image_data

        image, spacing, slice_intv = self.load_preprocess_nrrd(image_data)
        num_slice = cfg.INPUT.NUM_SLICES * cfg.INPUT.NUM_IMAGES_3DCE
        im, im_scale, crop = load_prep_img(image, slice_idx, spacing, slice_intv,
                                           cfg.INPUT.IMG_DO_CLIP, num_slice=num_slice, is_train=is_train)

        im -= cfg.INPUT.PIXEL_MEAN
        im = torch.from_numpy(im.transpose((2, 0, 1))).to(dtype=torch.float)
        
        ## get some metadata details from metadata file - eventually not used
        # spacing = img_info['space directions'][0][0]
        # slice_intv = img_info['space directions'][2][2]

        # 3. Get and preprocess mask data (= ground-truth (gt) lesion segmentations)
        mask_fn = image_fn.replace("image", "mask" )
        mask_data =  nrrd.read(self.data_path + "/" + mask_fn) 

        
        assert len(mask_data[0]) == len(image_data[0]) # check that image data CT volume is of the same size as the mask data CT volume

        mask, spacing, slice_intv = self.load_preprocess_nrrd(mask_data, False)
        mask = np.transpose(mask, (2, 0, 1))
        mask_slice = mask[slice_idx]

        # 4. Get ground-truth bounding boxes (a.k.a. bboxes) from mask data
        # Key assumption: each slice has one lesion and hence one bounding box (confirmed that all slices have lesions close to each other)
        # TODO: Modify the code to be able to locate multiple bounding boxes such that the above assumption is not made
        boxes0 = []
        diameters = []
        adjustment = ((im.shape[1])/(512))
        np_lesion = np.argwhere(mask_slice==1) # get indexes of the lesion within the mask
        left = (min(np_lesion[:, 1])-1)*adjustment # get indexes of left, right, up, down-most pixels
        down = (min(np_lesion[:, 0])-1)*adjustment
        right = (max(np_lesion[:, 1])-1)*adjustment
        up = (max(np_lesion[:, 0])-1)*adjustment
        
        diameters.append([right-left, up-down])# get diameters
        diameters = np.array(diameters)

        boxes0.append([left, down, right, up]) # get bboxes
        boxes0 = np.array(boxes0)
        boxes_new = boxes0.copy()
        boxes = torch.as_tensor(boxes_new).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (im.shape[2], im.shape[1]), mode= "xyxy")
        
        num_boxes = boxes.shape[0]
        classes = torch.ones(num_boxes, dtype=torch.int)  # lesion/nonlesion
        target.add_field("labels", classes)

        ## 5. Data augmentations (online) - TODO: try training first - comment out first if the code fails (requires modification)
        if is_train and cfg.INPUT.DATA_AUG_3D is not False:
            slice_radius = diameters.min() / 2 * spacing / slice_intv * abs(cfg.INPUT.DATA_AUG_3D)  # lesion should not be too small
            slice_radius = int(slice_radius)
            if slice_radius > 0:
                if cfg.INPUT.DATA_AUG_3D > 0:
                    delta = np.random.randint(0, slice_radius+1)
                else:  # give central slice higher prob
                    ar = np.arange(slice_radius+1)
                    p = slice_radius-ar.astype(float)
                    delta = np.random.choice(ar, p=p/p.sum())
                if np.random.rand(1) > .5:
                    delta = -delta

                dirname = image_fn.split("_")[0]
                image_fn1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
                if os.path.exists(os.path.join(self.data_path, image_fn1)):
                    image_fn = image_fn1


        # 6. Preparation for segmentation - preprocessing of mask
        masks = []
        if cfg.MODEL.MASK_ON:
            tensor_mask = mask_slice.copy()
            tensor_mask = torch.tensor(tensor_mask.astype('float32'))
            tensor_mask = torchvision.transforms.Resize((im.shape[-2], im.shape[-1]))(tensor_mask.unsqueeze(0))

            masks = tensor_mask
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=False)

        # 7. Returning CT slice image + segmentation gt mask + metadata
        # Note that image filename consists of both the actual nrrd image filename and the CT slice index
        infos = {'im_index': index, 'image_fn': image_fn + "_" + str(slice_idx), 'diameters': diameters*spacing,
                 'crop': crop, 'spacing': spacing, 'im_scale': im_scale}
        # NOTE: End of Breast CT modifications 
        return im, target, infos

    def __len__(self):
        return len(self.image_fn_list)

    # NOTE: Breast CT modification for preprocessing nrrd file data
    def load_preprocess_nrrd(self, data, image_conversion=True):
        if image_conversion:
            vol = (data[0].astype('int32') + 32768).astype('uint16')  # to be consistent with png files
        else:
            vol = data[0].astype('uint16')
        # spacing = -data.get_affine()[0,1]
        # slice_intv = -data.get_affine()[2,2]
        aff = data[1]['space directions'][:3, :3]
        spacing = np.abs(aff[:2, :2]).max()
        slice_intv = np.abs(aff[2, 2])

        if np.abs(aff[0, 0]) > np.abs(aff[0, 1]):
            vol = np.transpose(vol, (1, 0, 2))
            aff = aff[[1, 0, 2], :]
        if np.max(aff[0, :2]) > 0:
            vol = vol[::-1, :, :]
        if np.max(aff[1, :2]) > 0:
            vol = vol[:, ::-1, :]

        return vol, spacing, slice_intv

    def load_split_index(self):
        """
        need to group lesion indices to image indices, since one image can have multiple lesions
        :return:
        """

        split_list = ['train', 'val', 'test', 'small']
        index = split_list.index(self.split)
        if self.split != 'small':
            lesion_idx_list = np.where((self.train_val_test == index + 1) & ~self.noisy)[0]
        else:
            lesion_idx_list = np.arange(30)
        fn_list = self.filenames[lesion_idx_list]
        fn_list_unique, inv_ind = np.unique(fn_list, return_inverse=True)
        lesion_idx_grouped = [lesion_idx_list[inv_ind==i] for i in range(len(fn_list_unique))]
        return fn_list_unique, lesion_idx_grouped


