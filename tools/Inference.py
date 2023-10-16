from datasets.validation.fishyscapes import Fishyscapes
import logging
import os
from utils.img_utils import Compose, Normalize, ToTensor
from detectron2.config import get_cfg
from config import config
from tqdm import tqdm
from PIL import Image, ImageDraw
import json, cv2
import torch
import numpy
logger = logging.getLogger("detectron2")
import sys
sys.path.append('/people/cs/w/wxz220013/OOD/detectron2/detectron2/data/datasets')
from detectron2.data.datasets.cityscapes_coco import register_dataset
from data_loader import get_mix_loader
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import cv2
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
###############################################
cfg = get_cfg()
cfg.merge_from_file('../configs/OE/OE.yaml')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])
# fishyscapes_static = Fishyscapes(split='Static', root=config.fishy_root_path, transform=transform)
fishyscapes_static = Fishyscapes(split='Static', root=config.fishy_root_path)
test_set = fishyscapes_static
save_path=os.path.join(test_set.root,"inference")
stride = int(numpy.ceil(len(test_set)))
e_record = len(test_set)
shred_list = list(range(0, e_record))
curr_info = {}
#tbar = tqdm(shred_list[5:10], ncols=137, leave=True, miniters=1) if curr_rank <= 0 else shred_list #[5:10]：5，6，7，8，9
tbar = tqdm(shred_list, ncols=137, leave=True, miniters=1) #8,9
if not os.path.exists(save_path):
    os.makedirs(save_path)
with torch.no_grad():
    for idx in tbar:
        img, target = test_set[idx]
        im = numpy.array(img)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        bbox = outputs["instances"].pred_boxes.tensor.detach().to('cpu').numpy()
        for box in bbox:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 绘制矩形框
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (0, 255, 0) 是绿色，2 是线宽
        cv2.imwrite('img/'+str(idx)+'image_with_boxes.jpg', im)