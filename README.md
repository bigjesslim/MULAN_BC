MULAN
======

# MULAN_BC Documentation

GitHub code repository: [https://github.com/bigjesslim/MULAN_BC](https://github.com/bigjesslim/MULAN_BC)

Documentation for modifications done to the [original MULAN code repository](https://github.com/rsummers11/CADLab) to run on NCC (National Cancer Centre) Singapore’s NRRD files of Breast CT Scans. 

# How to run code

1. Navigate to the folder `MULAN_universal_lesion_analysis`
2. To setup: 
    1. `./setup.sh`
    2. `source venv/bin/activate`
3. Modify `config.yml` according to your requirements
    - For demo - produce visualizations for single CT volumes
        - mode = ‘demo’
        - Key in the path to the CT volume image in the command line
        - Ensure that the CT volume mask is in the same folder and titled in similar format as the current dataset (e.g., if CT volume image is `P576_image.nrrd` , the CT volume mask should be `P576_mask.nrrd`)
    - For training - train model
        - Change training parameters in `config.yml` and `maskrcnn/config/defaults.py`
        - mode = ‘train’
    - For evaluation - evaluate model on the evaluation dataset
        - mode = ‘evaluation’
4. Enter the command `python run.py` via the command line

# Key scripts and notes for future extensions

- More detailed code explanations can be found as comments within the code
    - Denoted with “NOTE”
    - Sections where future improvements can/should be made are denoted with “TODO”

### Configurations

Two main scripts:

1. `config.yml`
2. `maskrcnn/config/defaults.py`

Note: some overlapping constants/variables - `config.yml` overrides `defaults.py`

### General purpose

overarching folder = `MULAN_universal_lesion_analysis`

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

Note: `batch_demo_process.py` was left untouched - does not work on nrrd files with Breast CT directory structure

### Evaluation

| Script | Relative path | Capabilities | Extent of modifications |
| --- | --- | --- | --- |
| detection_eval.py | maskrcnn/data/datasets/evaluation/ DeepLesion/detection_eval.py | Contains functions to calculate different detection metrics (i.e., average sensitivity over FPs, average sensitivity, IOU, confidences, dice score)  | Moderate |
|  |  |  |  |
|  |  |  |  |

Note: Simple average of sensitivity 

# Trained weights + statistics

# Assumptions + Implications on code

1. Only one lesion per CT slice