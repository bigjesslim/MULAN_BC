# MULAN_BC Documentation

GitHub code repository: [https://github.com/bigjesslim/MULAN_BC](https://github.com/bigjesslim/MULAN_BC)

Documentation for modifications done to the [original MULAN code repository](https://github.com/rsummers11/CADLab) to run on NCC (National Cancer Centre) Singapore’s NRRD files of Breast CT Scans. 

# How to run code

1. Navigate to the folder `MULAN_universal_lesion_analysis`
2. To setup: 
    1. `./setup.sh`
    2. `source venv/bin/activate`
3. Modify `config.yml` according to your requirements
    - Note: Change pretrained weights to be used
    - For demo - produce visualizations for single CT volumes
        - `MODE = "demo"`
        - Key in the path to the CT volume image in the command line + choose“soft tissue” optioiin
        - Ensure that the CT volume mask is in the same folder and titled in similar format as the current dataset (e.g., if CT volume image is `P576_image.nrrd` , the CT volume mask should be `P576_mask.nrrd`)
        - Visualizations are saved under the `viz` folder
    - For training - train model
        - `MODE = "train"`
        - Change training parameters in `config.yml` and `maskrcnn/config/defaults.py`
    - For evaluation - evaluate model on the evaluation dataset
        - `MODE = "eval"`
4. Enter the command `python run.py` via the command line

### Configuring a new dataset

1. Dataset format - 
    - Data should be in the form of NRRD files
    - Place CT image volumes and corresponding mask volumes in the same folder
    - Name the NRRD files in similar format as the current dataset (e.g., if CT volume image is `P576_image.nrrd` , the CT volume mask should be `P576_mask.nrrd`)
2. Add new dataset path in `maskrcnn/config/paths_catalog.py` 
3. Reference the new dataset in `config.yml` - configures the train, evaluation and test datasets
4. Run the script `gen_lesion_slice_dict.py` to regenerate `maskrcnn/data/datasets/lesion_slice_dict.pkl` 
5. Configure the train-test-validation split in `BreastCT.py`

# Trained weights + metrics

|  | Classification metrics: |  |  | Detection metrics: |  | Segmentation metric: |
| --- | --- | --- | --- | --- | --- | --- |
| Weight | Average sensitivity  | Average sensitivity accounting for FPs* | Average confidence (over TP instances) | Average DICE coefficient (over TP instances) | Average IOU     (over TP instances) | Average DICE coefficient (over TP detections) |
| Pretrained | 0.6837 | 0.4476 | 0.7741 | 0.8412 | 0.7260 | - |
| Finetuned (6 epochs without segmentation) | 0.9457 | 0.8411 | 0.8960 | 0.8622 | 0.7579 | - |
| Finetuned (8 epochs with segmentation)  | 0.9531 | 0.8597 | 0.8923 | 0.8641 | 0.7607 | 0.9967 |

Weight names (for use)

- Pretrained: “MULAN trained on DeepLesion_epoch_08”
- Finetuned (6 epochs without segmentation): “MULAN trained on BreastCT_epoch_14”
- Finetuned (8 epochs with segmentation): "MULAN trained on BreastCT with Seg_epoch_16"

Note: all under the folder `checkpoints`

# Assumptions + Implications on code

1. Only one lesion per CT slice
    1. Generates a single bounding box for all ground-truth lesions
    2. Generates a single mask for all ground-truth lesions

# Key scripts and notes for future extensions

- More detailed code explanations can be found as comments within the code
    - Denoted with `# NOTE:`
    - Sections where future improvements can/should be made are denoted with `# TODO:`

### Configurations

Two main config scripts

1. `config.yml`
2. `maskrcnn/config/defaults.py`

Note: some overlapping constants/variables - `config.yml` overrides `defaults.py`

Dataset configurations

`maskrcnn/config/paths_catalog.py`

- Configures paths to dataset → add path to any new datasets here
- Referenced by `config.yml`

### General purpose

Overarching folder = `MULAN_universal_lesion_analysis`

| Script | Relative path | Capabilities | Extent of modifications |
| --- | --- | --- | --- |
| run.py | run.py | Main script to be run - will evoke other scripts downstream based on the mode variable set in config.yml | Only slight modifications |
| BreastCT.py | maskrcnn/data/datasets/BreastCT.py | Dataset object to allow for loading of the dataset with Breast CT scans (Configured specifically to the format of the current dataset under /Data/new data - i.e., directory arrangement + naming formats)  | Heavy  |
| load_ct_img.py | maskrcnn/data/datasets/load_ct_img.py | Helper functions to load and preprocess data (used in BreastCT.py and demo_process.py) | Only slight modifications |

### Demo

******************************`demo_process.py`******************************

- Required input: path to a patient’s image nrrd file
- Outputs: visualizations + logs of detections
    - Visualizations saved to the folder `viz` under a subfolder of the name of the weights used (check weights used in the script `config.yml`)
- Extent of modification: Mild

Note: `batch_demo_process.py` was left untouched - does not work on nrrd files with Breast CT directory structure

### Train

Mainly only modified to accommodate different format of **segmentation** ground-truths

- In DeepLesion, the full masks of the lesions were not available - polygons were generated from RECIST measurements

******************************`loss.py`******************************

- Relative path: maskrcnn/modeling/roi_heads/mask_head/loss.py
- Modified segmentation DICE loss calculation for Breast CT full ground truth format
- Extent of modification: Mild

******************************`mask_head.py`******************************

- Relative path: maskrcnn/modeling/roi_heads/mask_head/mask_head.py
- Modified segmentation branch final layer output vector to (512, 512) mask for calculation of DICE loss
- Extent of modification: Mild

### Evaluation

| Script | Relative path | Capabilities | Extent of modifications |
| --- | --- | --- | --- |
| detection_eval.py | maskrcnn/data/datasets/evaluation/ DeepLesion/detection_eval.py | Contains functions to calculate different detection metrics (i.e., average sensitivity over FPs, average sensitivity, IOU, confidences, dice score)  | Moderate |
| DL_eval.py | maskrcnn/data/datasets/evaluation/ DeepLesion/DL_eval.py | Calls functions in detection_eval.py to get detection metrics + evaluates segmentation metrics  | Moderate |

Note: Simple average of sensitivity was preferable over average sensitivities over FPs in the case of developing a selection tool to produce proposals using this model

Credit goes to the original authors of the MULAN code repository for building the original network.