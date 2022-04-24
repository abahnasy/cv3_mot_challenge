from yacs.config import CfgNode as CN

_C = CN()

# Experiment Name
_C.EXPERIMENT = CN()
_C.EXPERIMENT.UID = "" # overwrite with timestamp and job id in the main.py
_C.EXPERIMENT.PROJECT = "mot-challenge"
_C.EXPERIMENT.WORKSPACE = None
_C.EXPERIMENT.TAGS = []

# Datsets
_C.DATASETS = CN()
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.SEQ_NAME = "MOT16-reid"
_C.DATASETS.TEST = CN()

# Model
_C.MODEL = CN()
_C.MODEL.DETECTOR = CN()
_C.MODEL.DETECTOR.NAME = "FRCNN_FPN"
_C.MODEL.DETECTOR.CKPT = "/lustre/groups/imm01/datasets/ahmed.bahnasy/mot_challenge/cv3dst_exercise/models/faster_rcnn_fpn.model"
_C.MODEL.DETECTOR.BACKBONE = "resnet50"
_C.MODEL.DETECTOR.NMS_THRESH = 0.3
_C.MODEL.DETECTOR.NUM_CLASSES = 2

_C.MODEL.TRACKER = CN()
_C.MODEL.TRACKER.NAME = "TrackerIoUAssignment"

# Optimizer
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = ""
_C.OPTIMIZER.LR = 1e-4
_C.OPTIMIZER.WEIGHT_DECAY = 0.0
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = ""