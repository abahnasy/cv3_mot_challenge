from src.modeling.detector import build_detector

from .tracker import (
    TrackerIoUAssignment,
)
def build_tracker(cfg):
    # build Detector
    obj_detect = build_detector(cfg)
    obj_detect.eval()
    obj_detect.cuda() #to(cfg.DEVICE)
    # build Tracking head
    if cfg.MODEL.TRACKER.NAME == "TrackerIoUAssignment":
        tracker_fn = TrackerIoUAssignment
    else:
        return ValueError("Undefined Tracking head !")
    
    return tracker_fn(
        obj_detect
    )

