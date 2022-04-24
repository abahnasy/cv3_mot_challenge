import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FRCNN_FPN(FasterRCNN):

    def __init__(self, backbone, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone(backbone, False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()


def build_detector(cfg):
    if(cfg.MODEL.DETECTOR.NAME == "FRCNN_FPN"):
        detector_fn = FRCNN_FPN
    else:
        raise ValueError("Unknown detector architecture")
    obj_detect =  detector_fn(
        backbone = cfg.MODEL.DETECTOR.BACKBONE,
        num_classes = cfg.MODEL.DETECTOR.NUM_CLASSES,
        nms_thresh = cfg.MODEL.DETECTOR.NMS_THRESH,
    )

    if cfg.MODEL.DETECTOR.CKPT:
        obj_detect_state_dict = torch.load(cfg.MODEL.DETECTOR.CKPT,
                                    map_location=lambda storage, loc: storage)
        obj_detect.load_state_dict(obj_detect_state_dict)
    
    return obj_detect