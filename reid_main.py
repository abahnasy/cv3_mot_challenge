import os
import time, random

from comet_ml import Experiment
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.data.datasets.market.datamanager import ImageDataManager
from src.modeling import build_model
from src.metrics.distances import (
    euclidean_squared_distance,
    cosine_distance,
)
from src.utils.timer import generate_time_stmp
from src.utils.metrics import MetricMeter, AverageMeter, eval_market1501
from src.utils.misc import print_statistics, extract_features, compute_distance_matrix
from src.losses.triplet import CombinedLoss

load_dotenv()

try:
    SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]
except:
    SLURM_JOB_ID = "INTERACTIVE"

DATA_DIR = os.getenv("DATA_DIR")
DATA_DIR = os.path.join(DATA_DIR, '..')

def seed_torch(seed=0):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def evaluate(model, test_loader, metric_fn, ranks=[1, 5, 10, 20]):
    with torch.no_grad():
        model.eval()
        print('Extracting features from query set...')
        q_feat, q_pids, q_camids = extract_features(model, test_loader['query'])
        print('Done, obtained {}-by-{} matrix'.format(q_feat.size(0), q_feat.size(1)))

        print('Extracting features from gallery set ...')
        g_feat, g_pids, g_camids = extract_features(model, test_loader['gallery'])
        print('Done, obtained {}-by-{} matrix'.format(g_feat.size(0), g_feat.size(1)))
        
        distmat = compute_distance_matrix(q_feat, g_feat, metric_fn=metric_fn)
        distmat = distmat.numpy()

        print('Computing CMC and mAP ...')
        cmc, mAP = eval_market1501(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=50
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        return cmc[0], mAP


if __name__ == "__main__":
    # load data class
    # data loader
    # build model
    # build optimizer
    # training loop
    # comet ml experiment

    exp_time_stamp = generate_time_stmp()
    exp_name = "reid_model_{}_slurm_{}".format(exp_time_stamp, SLURM_JOB_ID)
    OUTPUT_DIR = "./outputs/{}".format(exp_name)
    experiment = Experiment(project_name="mot_challenge_reid")
    experiment.set_name(exp_name)
    # datamanager = ImageDataManager(root=DATA_DIR, height=256,width=128, batch_size_train=32, 
    #                            workers=2, transforms=['random_flip', 'random_crop'])
    datamanager = ImageDataManager(root=DATA_DIR, height=256,width=128, batch_size_train=32, 
                               workers=2, transforms=['random_flip', 'random_crop'],
                               train_sampler='RandomIdentitySampler')
    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader

    # model = build_model('resnet18', datamanager.num_train_pids, loss='softmax', pretrained=True)
    model = build_model('resnet34', datamanager.num_train_pids, loss='triplet', pretrained=True)
    model = model.cuda()

    trainable_params = model.parameters()
    
    optimizer = torch.optim.Adam(trainable_params, lr=0.0003, # <--- Feel free to play around with the lr parameter.
                             weight_decay=5e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    metric_fn = euclidean_squared_distance  # Needs to be set to one of the distance measurements..

    MAX_EPOCH = 30
    EPOCH_EVAL_FREQ = 5
    PRINT_FREQ = 10

    num_batches = len(train_loader)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = CombinedLoss(0.3, 1.0, 1.0)

    glob_step = 1
    for epoch in range(MAX_EPOCH):
        losses = MetricMeter()
        batch_time = AverageMeter()
        end = time.time()
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Predict output.
            imgs, pids = data['img'].cuda(), data['pid'].cuda()
            logits, features = model(imgs)
            # Compute loss.
            loss, loss_summary = criterion(logits, features, pids)
            experiment.log_metrics(loss_summary, step = glob_step)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            if (batch_idx + 1) % PRINT_FREQ == 0:
                print_statistics(batch_idx, num_batches, epoch, MAX_EPOCH, batch_time, losses)
            end = time.time()
            glob_step += 1
        
        scheduler.step()
        if (epoch + 1) % EPOCH_EVAL_FREQ == 0 or epoch == MAX_EPOCH - 1:
            rank1, mAP = evaluate(model, test_loader, metric_fn)
            experiment.log_metric("Rank1", rank1, step= glob_step )
            experiment.log_metric("mAP", mAP, step= glob_step)
            print('Epoch {0}/{1}: Rank1: {rank}, mAP: {map}'.format(
                        epoch + 1, MAX_EPOCH, rank=rank1, map=mAP))

    
    # for epoch in range(MAX_EPOCH):
    #     losses = MetricMeter()
    #     batch_time = AverageMeter()
    #     end = time.time()
    #     model.train()
    #     for batch_idx, data in enumerate(train_loader):
    #         # Predict output.
    #         imgs, pids = data['img'].cuda(), data['pid'].cuda()
    #         output = model(imgs)
    #         # Compute loss.
    #         # loss = criterion(output, pids)
    #          loss, loss_summary = criterion(logits, features, pids)
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         batch_time.update(time.time() - end)
    #         losses.update({'Loss': loss})
    #         if (batch_idx + 1) % PRINT_FREQ == 0:
    #             print_statistics(batch_idx, num_batches, epoch, MAX_EPOCH, batch_time, losses)
    #         end = time.time()
        
    #     scheduler.step()
    #     if (epoch + 1) % EPOCH_EVAL_FREQ == 0 or epoch == MAX_EPOCH - 1:
    #         rank1, mAP = evaluate(model, test_loader, metric_fn)
    #         print('Epoch {0}/{1}: Rank1: {rank}, mAP: {map}'.format(
    #                     epoch + 1, MAX_EPOCH, rank=rank1, map=mAP))