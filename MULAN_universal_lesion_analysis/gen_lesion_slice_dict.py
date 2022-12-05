import numpy as np
import nrrd
from PIL import Image
import os
import pickle

# get list of image filenames
image_fn_list = set()
for file in os.listdir("/home/tester/Data/new data"): 
    if (file.endswith(".nrrd")) and ('image' in str(file)):
        # exclude patient 105 - no lesion
        if "P105_image" not in file:
            image_fn_list.add(file)

    
image_fn_list = list(image_fn_list)

# get slices for each file with lesions
lesion_ct_slices = {}
count = 0
for image_fn in image_fn_list:
    count += 1
    mask_fn = image_fn.split("_")[0] + "_mask.nrrd"
    print(count)
    mask_data =  nrrd.read("/home/tester/Data/new data/" + mask_fn) 
    mask = mask_data[0]

    # get list of all ct slices with lesions into "lesion_ct_slices"
    mask = np.transpose(mask, (2, 0, 1))
    slices = []
    for i in range(mask.shape[0]):
        if np.argwhere(mask==1).shape[0] > 0:
            slices.append(i)
    lesion_ct_slices[image_fn] = slices

dbfile = open("/home/tester/jessica/MULAN_BC/MULAN_universal_lesion_analysis/maskrcnn/data/datasets/lesion_slice_dict.pkl", 'ab')
pickle.dump(lesion_ct_slices, dbfile)