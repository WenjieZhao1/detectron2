#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    build_OE_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from config import config
import json, cv2
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import random

logger = logging.getLogger("detectron2")
import sys
sys.path.append('/people/cs/w/wxz220013/OOD/detectron2/detectron2/data/datasets')
from detectron2.data.datasets.cityscapes_coco import register_dataset
from data_loader import get_mix_loader
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
###############################################dataset
from detectron2.structures import BoxMode
import cv2
def get_OE_offline(img_dir):
    json_file = os.path.join(img_dir, "ood.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        filename = os.path.join(img_dir, "offline_dataset", v["file_name"])
        new_filename = os.path.splitext(filename)[0] + '.png'
        height, width = cv2.imread(new_filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        annos = v["annotations"]
        objs = []
        obj = {
            "bbox": [annos['bbox'][0], annos['bbox'][1], annos['bbox'][2], annos['bbox'][3]],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("OOD_" + d, lambda d=d: get_OE_offline("/people/cs/w/wxz220013/OOD/meta-ood/COCO/annotations"))
    MetadataCatalog.get("OOD_" + d).set(thing_classes=["OOD"])
OOD_metadata = MetadataCatalog.get("OOD_train")
###############################################

cfg = get_cfg()
cfg.merge_from_file('../configs/OE/OE.yaml')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_OE_offline("/people/cs/w/wxz220013/OOD/meta-ood/COCO/annotations")
i = 0
for d in random.sample(dataset_dicts, 5):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=OOD_metadata, 
                   scale=1
                #    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    bbox = outputs["instances"].pred_boxes.tensor.detach().to('cpu').numpy()
    print(bbox)
    print(bbox.sum)
    if bbox.sum==0 :
        continue
    out = v.draw_box(bbox[0],edge_color = 'lightblue',alpha = 1)
    out.save(str(i)+'out.jpg')
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    a=out.get_image()[:, :, ::-1]
    print(a.shape,type(a),)
    cv2.imwrite(str(i)+'test.jpg',out.get_image()[:, :, ::-1])
    i+=1