# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Centralized catalog of paths."""
# Added DeepLesion dataset

import os


class DatasetCatalog(object):
    DATA_DIR = "/home/tester/jessica/MULAN_BC/MULAN_universal_lesion_analysis/maskrcnn/data/"
    DATASETS = {
        "DeepLesion_train": {
            "data_dir": "DeepLesion/Images_png",
            "split": "train",
            "ann_file": "DeepLesion/DL_info.csv",
        },
        "DeepLesion_val": {
            "data_dir": "DeepLesion/Images_png",
            "split": "val",
            "ann_file": "DeepLesion/DL_info.csv"
        },
        "DeepLesion_test": {
            "data_dir": "DeepLesion/Images_png",
            "split": "test",
            "ann_file": "DeepLesion/DL_info.csv"
        },
        "DeepLesion_small": {  # for debug
            "data_dir": "DeepLesion/Images_png",
            "split": "small",
            "ann_file": "DeepLesion/DL_info.csv"
        },
        "DeepLesion_mini": {  # for debug
            "data_dir": "DeepLesion/minideeplesion",
            "split": "small",
            "ann_file": "DeepLesion/DL_info.csv"
        },
        "BreastCT_train":{
            "data_dir": "Data/new data",
            "split": "train",
            "ann_file": None
        },
        "BreastCT_test":{
            "data_dir": "Data/new data",
            "split": "test",
            "ann_file": None
        },
        "BreastCT_val":{
            "data_dir": "Data/new data",
            "split": "val",
            "ann_file": None
        }
    }

    @staticmethod
    def get(name):
        if "DeepLesion" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="DeepLesionDataset",
                args=args,
            )
        if "BreastCT" in name:
            data_dir = "/home/tester/"
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                split=attrs["split"],
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                ann_file=None
            )
            return dict(
                factory="BreastCTDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://s3-us-west-2.amazonaws.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
