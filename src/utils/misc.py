import datetime

import numpy as np
import torch

def extract_features(model, data_loader):
    f_, pids_, camids_ = [], [], []
    for data in data_loader:
        imgs, pids, camids = data['img'], data['pid'], data['camid']
        imgs = imgs.cuda()
        features = model(imgs)
        features = features.cpu().clone()
        f_.append(features)
        pids_.extend(pids)
        camids_.extend(camids)
    f_ = torch.cat(f_, 0)
    pids_ = np.asarray(pids_)
    camids_ = np.asarray(camids_)
    return f_, pids_, camids_


def print_statistics(batch_idx, num_batches, epoch, max_epoch, batch_time, losses):
    batches_left = num_batches - (batch_idx + 1)
    future_batches_left = (max_epoch - (epoch + 1)) * num_batches
    eta_seconds = batch_time.avg * (batches_left + future_batches_left)
    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
    print(
        'epoch: [{0}/{1}][{2}/{3}]\t'
        'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'eta {eta}\t'
        '{losses}\t'.format(
            epoch + 1,
            max_epoch,
            batch_idx + 1,
            num_batches,
            batch_time=batch_time,
            eta=eta_str,
            losses=losses
        )
    )

def compute_distance_matrix(input1, input2, metric_fn):
    """A wrapper function for computing distance matrix.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric_fn (func): A function computing the pairwise distance 
            of input1 and input2.
    Returns:
        torch.Tensor: distance matrix.
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1), f'Input size 1 {input1.size(1)}; Input size 2 {input2.size(1)}'

    return metric_fn(input1, input2)