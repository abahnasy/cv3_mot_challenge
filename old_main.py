import os
import time


from comet_ml import Experiment
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DataLoader
)
import dotenv

# load env variables and secret keys
from dotenv import load_dotenv
load_dotenv()

from tracker.track_data import MOT16Sequences
from tracker.object_detector import FRCNN_FPN
from tracker.utils import (
    evaluate_obj_detect,
    obj_detect_transforms
)
from tracker.data_object_detector import MOT16ObjDetect
from metrics.utils import (
    evaluate_mot_accums,
    get_mot_accum
)
from utils.timer import generate_time_stmp

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")

def best_deterministic_effort(seed = 12345):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_tracker(tracker, sequences, output_dir):
    time_total = 0
    mot_accums = []
    results_seq = {}
    for seq in sequences:
        tracker.reset()
        now = time.time()

        print(f"Tracking: {seq}")
        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        for frame in tqdm(data_loader):
            tracker.step(frame)
        results = tracker.get_results()
        results_seq[str(seq)] = results
        if seq.no_gt:
            print(f"No GT evaluation data available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        time_total += time.time() - now

        print(f"Tracks found: {len(results)}")
        print(f"Runtime for {seq}: {time.time() - now:.1f} s.")

        seq.write_results(results, os.path.join(output_dir))

    print(f"Runtime for all sequences: {time_total:.1f} s.")
    if mot_accums:
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in sequences if not s.no_gt],
                            generate_overall=True)


if __name__ == "__main__":

    best_deterministic_effort()
    
    experiment = Experiment(project_name="mot_challenge")
    experiment_name = "{}_{}_slurm-job-id-{}".format(generate_time_stmp, time_stamp, SLURM_JOB_ID)
    experiment.set_name(experiment_name)
    DATA_DIR = "/lustre/groups/imm01/datasets/ahmed.bahnasy/mot_challenge/cv3dst_exercise/data"
    MODEL_ZOO = "/lustre/groups/imm01/datasets/ahmed.bahnasy/mot_challenge/cv3dst_exercise/models"
    OUTPUT_DIR = "/lustre/groups/imm01/datasets/ahmed.bahnasy/mot_challenge/cv3dst_exercise/output" # fix, separte output folders according to timestamp generated folders

    # explore data
    # seq_name = 'MOT16-02'
    # data_dir = os.path.join(DATA_DIR, 'MOT16')
    # sequences = MOT16Sequences(seq_name, data_dir, load_seg=True)

    # for seq in sequences:
    #     for i, frame in enumerate(seq):
    #         img = frame['img']
            
    #         dpi = 96
    #         fig, ax = plt.subplots(1, dpi=dpi)

    #         img = img.mul(255).permute(1, 2, 0).byte().numpy()
    #         width, height, _ = img.shape
            
    #         ax.imshow(img, cmap='gray')
    #         fig.set_size_inches(width / dpi, height / dpi)

    #         if 'gt' in frame:
    #             gt = frame['gt']
    #             for gt_id, box in gt.items():
    #                 rect = plt.Rectangle(
    #                 (box[0], box[1]),
    #                 box[2] - box[0],
    #                 box[3] - box[1],
    #                 fill=False,
    #                 linewidth=1.0)
    #                 ax.add_patch(rect)

    #         plt.axis('off')
    #         # plt.show()
    #         experiment.log_figure(figure_name="GT Boxes", figure=fig, overwrite=False, step=None)

    #         if 'seg_img' in frame:
    #             seg_img = frame['seg_img']
    #             fig, ax = plt.subplots(1, dpi=dpi)
    #             fig.set_size_inches(width / dpi, height / dpi)
    #             ax.imshow(seg_img, cmap='gray')
    #             plt.axis('off')
    #             #plt.show()
    #             experiment.log_figure(figure_name="Seg_imgs", figure=fig, overwrite=False, step=None)
    #         break

    # Detector settings        
    obj_detect_model_file = os.path.join(MODEL_ZOO, 'faster_rcnn_fpn.model')
    obj_detect_nms_thresh = 0.3

    # object detector
    obj_detect = FRCNN_FPN(num_classes=2, nms_thresh=obj_detect_nms_thresh)
    obj_detect_state_dict = torch.load(obj_detect_model_file,
                                    map_location=lambda storage, loc: storage)
    obj_detect.load_state_dict(obj_detect_state_dict)
    obj_detect.eval()
    obj_detect.to(device)


    # dataset_test = MOT16ObjDetect(os.path.join(DATA_DIR, 'MOT16/train'),
                            #   obj_detect_transforms(train=False))
    
    # evalutate the detection model on the current dataset
    # def collate_fn(batch):
    #     return tuple(zip(*batch))
    # data_loader_test = DataLoader(
    #     dataset_test, batch_size=1, shuffle=False, num_workers=4,
    #     collate_fn=collate_fn)
    # evaluate_obj_detect(obj_detect, data_loader_test)

    # ======================================================== #
    # =============Test Naiive Tracker with IoU=============== #
    # ======================================================== #

    from tracker.tracker import TrackerIoUAssignment, HungarianIoUTracker, HungarianIoUTrackerB, HungarianIoUTrackerBInactiveCounter, ReIDHungarianIoUTracker

    seq_name = 'MOT16-reid'
    data_dir = os.path.join(DATA_DIR, 'MOT16')
    sequences = MOT16Sequences(seq_name, data_dir)

    # print("="*10, "IoU Tracker", "="*10)
    # iou_tracker = TrackerIoUAssignment(obj_detect)
    # run_tracker(iou_tracker, sequences, OUTPUT_DIR)
    # print("="*10, "Hungarian Tracker", "="*10)
    # hungarian_tracker = HungarianIoUTracker(obj_detect)
    # run_tracker(hungarian_tracker, sequences, OUTPUT_DIR)
    # print("="*10, "Hungarian TrackerB", "="*10)
    # hungarian_tracker = HungarianIoUTrackerB(obj_detect)
    # run_tracker(hungarian_tracker, sequences, OUTPUT_DIR)
    # print("="*10, "HungarianIoUTrackerBInactiveCounter", "="*10)
    # hungarian_tracker = HungarianIoUTrackerBInactiveCounter(obj_detect, inactive_patience=10)
    # run_tracker(hungarian_tracker, sequences, OUTPUT_DIR)
    
    print("="*10, "ReIDHungarianIoUTracker", "="*10)
    ReIDHungarianIoUTracker(obj_detect)
    run_tracker(hungarian_tracker, sequences, OUTPUT_DIR)
    

    

    