from src.modeling.detector import build_detector

from .tracker import (
    TrackerIoUAssignment,
    HungarianIoUTracker,
    HungarianIoUTrackerB,
)
def build_tracker(cfg):
    # build Detector
    obj_detect = build_detector(cfg)
    obj_detect.eval()
    obj_detect.cuda() #to(cfg.DEVICE)
    # build Tracking head
    if cfg.MODEL.TRACKER.NAME == "TrackerIoUAssignment":
        tracker_fn = TrackerIoUAssignment
    elif cfg.MODEL.TRACKER.NAME == "HungarianIoUTracker":
        tracker_fn = HungarianIoUTracker
    elif cfg.MODEL.TRACKER.NAME == "HungarianIoUTrackerB":
        tracker_fn = HungarianIoUTrackerB
    else:
        raise ValueError("Undefined Tracking head !")
    
    return tracker_fn(
        obj_detect
    )

