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

from maskrcnn.data.datasets.load_ct_img import load_prep_img
from maskrcnn.structures.bounding_box import BoxList
from maskrcnn.structures.segmentation_mask import SegmentationMask
from maskrcnn.config import cfg
from maskrcnn.data.datasets.DeepLesion_utils import load_tag_dict_from_xlsfile, gen_mask_polygon_from_recist, load_lesion_tags
from maskrcnn.data.datasets.DeepLesion_utils import gen_parent_list, gen_exclusive_list, gen_children_list


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

        # get list of image filenames
        self.image_fn_list = set()
        for file in os.listdir(data_dir):
            if (file.endswith(".nrrd")) and ('image' in str(file)):
                self.image_fn_list.add(file)
        self.image_fn_list = list(self.image_fn_list)

        self.num_images = len(self.image_fn_list)
        self.logger = logging.getLogger(__name__)
        self.logger.info('DeepLesion %s num_images: %d' % (split, self.num_images))
    
    # removed tag processing functions as this dataset has no tags

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, info).
        """

        # read image from nrrd file
        image_fn = self.image_fn_list[index]
        image, img_info =  nrrd.read(self.data_path + image_fn) 
        # lesion_idx_grouped = self.lesion_idx_grouped[index]
        
        # get some metadata details from metadata file
        spacing = img_info['space directions'][0][0]
        slice_intv = img_info['space directions'][2][2]

        # get details on each lesion from the mask nrrd file
        mask_fn = image_fn.replace("image", "mask" )
        mask, _ = nrrd.read(self.data_path + mask_fn) # gets numpy file
        mask = np.transpose(mask, (2, 0, 1))

        # get indices of slices with lesions from mask numpy
        lesion_slices = []
        for i in range(len(mask)):
            if np.any(mask[i]):
                lesion_slices.append(i)

        # get random slice from all slices with lesions
        chosen_slice_index = random.choice(lesion_slices)
        #image = image[chosen_slice_index]
        mask = mask[chosen_slice_index]

        # get lesions from chosen mask
        def get_lesion_pixels(mask):
            one_indices = np.argwhere(mask==1)
            one_indices = list(one_indices)
            one_indices = [tuple(x) for x in one_indices]
            all_lesions = []
            visited_indices = []
            while len(one_indices) > 0:
                start = one_indices.pop(0)
                lesion_indexes, visited_indices = bfs(start, visited_indices,mask)
                for index in lesion_indexes:
                    if index in one_indices:
                        one_indices.remove(index)
                if len(lesion_indexes) > 0:
                    all_lesions.append(lesion_indexes) # list of sets of indexes (each list = lesion denoted by set of pixels)
            
            return all_lesions
            

        def bfs(start, visited_indices, mask):
            lesion_indexes = set()
            lesion_indexes.add(start)
            bfs_queue = [start]
            curr_i = start
            while len(bfs_queue) > 0:
                # check up, down, left, right
                up = tuple([curr_i[0]-1,curr_i[1]]) if curr_i[0]>0 else None
                down = tuple([curr_i[0]+1,curr_i[1]]) if curr_i[0]<mask.shape[0]-1 else None
                left = tuple([curr_i[0],curr_i[1]-1]) if curr_i[1]>0 else None
                right = tuple([curr_i[0],curr_i[1]+1]) if curr_i[1]<mask.shape[1]-1 else None

                all_directions = [up, down, left, right]

                for next_i in all_directions:
                    if (next_i != None) and (mask[next_i[0]][next_i[1]] == 1):
                        lesion_indexes.add(next_i)
                        if (next_i not in visited_indices) and  (next_i not in bfs_queue):
                            bfs_queue.append((next_i))
                
                visited_indices.append(curr_i)
                curr_i = bfs_queue.pop(0)
            
            return lesion_indexes, visited_indices

        lesions = get_lesion_pixels(mask)

        # get bbox and diameters from lesion
        boxes0 = []
        diameters = []
        masks = []
        for lesion in lesions:
            lesion =[list(x) for x in lesion]
            np_lesion = np.array(lesion)

            left = (min(np_lesion[:, 1])-1)*spacing
            down = (min(np_lesion[:, 0])-1)*spacing
            right = (max(np_lesion[:, 1])-1)*spacing
            up = (max(np_lesion[:, 0])-1)*spacing
            # get bboxes
            boxes0.append([left, down, right, up])
            # get diameters
            diameters.append([right-left, up-down])
            # get individual masks
            def get_1d_index(x):
                index = x[0]*512 + x[1]
                return index
            lesion_indexes = np.apply_along_axis(get_1d_index, 1, np_lesion)
            lesion_mask = torch.zeros(mask[0].shape)
            tuple_lesion =tuple(torch.tensor(x) for x in lesion)
            lesion_mask.put_(torch.tensor(lesion_indexes), torch.ones(len(lesion_indexes)))
            masks.append(lesion_mask)


        #recists = self.d_coordinate[lesion_idx_grouped] - not required
        #window = self.DICOM_window[lesion_idx_grouped][0] - not sure
        #z_coord = self.norm_location[lesion_idx_grouped[0], 2] - not sure

        num_slice = cfg.INPUT.NUM_SLICES * cfg.INPUT.NUM_IMAGES_3DCE
        is_train = self.split =='train'

        ## DATA AUG (online) - TODO: try training first - comment out first if the code fails (requires modification)
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

                dirname, slicename = image_fn.split(os.sep)
                slice_idx = int(slicename[:-4])
                image_fn1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
                if os.path.exists(os.path.join(self.data_path, image_fn1)):
                    image_fn = image_fn1

        # TODO: modify this function and how it gets neighbouring slices
        im, im_scale, crop = load_prep_img(image, chosen_slice_index, spacing, slice_intv,
                                           cfg.INPUT.IMG_DO_CLIP, num_slice=num_slice, is_train=is_train, load_from_nrrd=True)

        im -= cfg.INPUT.PIXEL_MEAN
        # TODO: below is alr done before - check if need to do again or just remove
        im = torch.from_numpy(im.transpose((2, 0, 1))).to(dtype=torch.float)

        boxes_new = boxes0.copy()
        boxes_new *= im_scale
        boxes = torch.as_tensor(boxes_new).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (im.shape[2], im.shape[1]), mode="xyxy")

        num_boxes = boxes.shape[0]
        classes = torch.ones(num_boxes, dtype=torch.int)  # lesion/nonlesion
        target.add_field("labels", classes)

        # tagging portion removed

        # segmentation
        if cfg.MODEL.MASK_ON:
            ## getting mask from polygons from recist
            # masks = []
            # for recist in recists:
            #     mask = gen_mask_polygon_from_recist(recist)
            #     masks.append([mask])
            # masks = SegmentationMask(masks, (im.shape[-1], im.shape[-2]))
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=False)

        if self.transforms is not None:
            im, target = self.transforms(im, target)

        # infos = {'im_index': index, 'lesion_idxs': lesion_idx_grouped, 'image_fn': image_fn, 'diameters': diameters*spacing,
        #          'crop': crop, 'recists': recists, 'window': window, 'spacing': spacing, 'im_scale': im_scale,
        #          'z_coord': z_coord}

        infos = {'im_index': index, 'image_fn': image_fn, 'diameters': diameters*spacing,
                 'crop': crop, 'spacing': spacing, 'im_scale': im_scale}
        return im, target, infos

    def __len__(self):
        return len(self.image_fn_list)

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

    def loadinfo(self, path):
        """load annotations and meta-info from DL_info.csv"""
        info = []
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                filename = row[0]  # replace the last _ in filename with / or \
                idx = filename.rindex('_')
                row[0] = filename[:idx] + os.sep + filename[idx + 1:]
                info.append(row)
        info = info[1:]

        # the information not used in this project are commented
        self.filenames = np.array([row[0] for row in info])
        self.slice_idx = np.array([int(row[4]) for row in info])
        self.d_coordinate = np.array([[float(x) for x in row[5].split(',')] for row in info])
        self.d_coordinate -= 1
        self.boxes = np.array([[float(x) for x in row[6].split(',')] for row in info])
        self.boxes -= 1  # coordinates in info file start from 1
        self.diameter = np.array([[float(x) for x in row[7].split(',')] for row in info])
        self.norm_location = np.array([[float(x) for x in row[8].split(',')] for row in info])

        self.spacing3D = np.array([[float(x) for x in row[12].split(',')] for row in info])
        self.spacing = self.spacing3D[:, 0] # spacing = real world scale of each pixel
        self.slice_intv = self.spacing3D[:, 2]  # slice intervals
        self.DICOM_window = np.array([[float(x) for x in row[14].split(',')] for row in info])
        
        # no info on this 
        # self.age = np.array([float(row[16]) for row in info])  # may be NaN
        self.train_val_test = np.array([int(row[17]) for row in info])
