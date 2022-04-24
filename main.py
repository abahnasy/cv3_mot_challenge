import os, sys
import argparse
import random
import time
import logging

from comet_ml import Experiment
from dotenv import load_dotenv
from yacs.config import CfgNode
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import get_cfg
from src.tracker import build_tracker
from src.metrics.utils import (
    get_mot_accum,
    evaluate_mot_accums,
)
from src.utils.timer import generate_time_stmp
from src.data import MOT16Sequences
from src.utils.logger import (
    setup_loggers,
    get_logger
)
from src.utils.viz import plot_sequence

load_dotenv()

try:
    SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]
except:
    SLURM_JOB_ID = "INTERACTIVE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = os.environ["DATA_DIR"]


def seed_torch(seed=0):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup(opts):
    cfg = get_cfg()
    exp_time_stamp = generate_time_stmp()
    cfg.EXPERIMENT.UID = "{}_slurm_{}".format(exp_time_stamp, SLURM_JOB_ID)
    cfg.OUTPUT_DIR = "./outputs/{}".format(cfg.EXPERIMENT.UID)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id # not ready for multi-gpu training
    cfg.freeze()
    # configuration snapshot
    os.makedirs(cfg.OUTPUT_DIR)
    path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    if isinstance(cfg, CfgNode):
        with open(path, 'w') as f:
            f.write(cfg.dump())
    experiment = None
    if opts.enable_vis:
        experiment = Experiment(
            workspace = cfg.EXPERIMENT.WORKSPACE, 
            project_name=cfg.EXPERIMENT.PROJECT
        )
        experiment.set_name(cfg.EXPERIMENT.UID)
        experiment.log_parameters(cfg)
    return cfg, experiment



def main():
    
    opts = get_argparser().parse_args()
    seed_torch(opts.seed)
    cfg, experiment = setup(opts)
    setup_loggers(cfg)
    logger = get_logger()
    

    # prepare data
    sequences = MOT16Sequences(cfg.DATASETS.TRAIN.SEQ_NAME, DATA_DIR)
    tracker = build_tracker(cfg)

    # run tracker
    time_total = 0
    mot_accums = []
    results_seq = {}
    for seq in sequences:
        tracker.reset()
        now = time.time()

        logger.info(f"Tracking: {seq}")

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)

        for frame in tqdm(data_loader):
            tracker.step(frame)
        results = tracker.get_results()
        results_seq[str(seq)] = results

        if seq.no_gt:
            logger.info(f"No GT evaluation data available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        time_total += time.time() - now

        logger.info(f"Tracks found: {len(results)}")
        logger.info(f"Runtime for {seq}: {time.time() - now:.1f} s.")

        seq.write_results(results, os.path.join(cfg.OUTPUT_DIR))
        plot_sequence(results, seq, first_n_frames=10, output_dir = "{}/{}".format(cfg.OUTPUT_DIR, str(seq)))

    logger.info(f"Runtime for all sequences: {time_total:.1f} s.")
    if mot_accums:
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in sequences if not s.no_gt],
                            generate_overall=True)

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0',
                    help="GPU ID")
    parser.add_argument("--seed", type=int, default=12345,
                    help="consistent seed for repreduciblity")
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use CometML for visualization")
    return parser

if __name__ == "__main__":
    main()