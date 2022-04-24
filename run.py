from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

MAX_PATIENCE = 20
MAX_EPOCHS = 15
EVAL_FREQ = 1


@torch.no_grad()
def compute_class_metric(pred, target, class_metrics = ('accuracy', 'recall', 'precision')):
    TP = ((target == 1) & (pred == 1)).sum().float()
    FP = ((target == 0) & (pred == 1)).sum().float()
    TN = ((target == 0) & (pred == 0)).sum().float()
    FN = ((target == 1) & (pred == 0)).sum().float()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0)
    precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0)

    class_metrics_dict =  {'accuracy': accuracy.item(), 'recall': recall.item(), 'precision': precision.item()}
    class_metrics_dict = {met_name: class_metrics_dict[met_name] for met_name in class_metrics}

    return class_metrics_dict


def train_one_epoch(model, data_loader, optimizer, accum_batches = 1, print_freq= 200):    
    model.train()
    device = next(model.parameters()).device
    metrics_accum = {'loss': 0., 'accuracy': 0., 'recall': 0., 'precision': 0.}
    #for i, batch in tqdm.tqdm(enumerate(data_loader)):
    for i, batch in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()

        # Since our model does not support automatic batching, we do manual
        # gradient accumulation        
        for sample in batch:
            past_frame, curr_frame = sample
            track_feats, track_coords, track_ids = past_frame['features'].to(device), past_frame['boxes'].to(device), past_frame['ids'].to(device)
            current_feats, current_coords, curr_ids = curr_frame['features'].to(device), curr_frame['boxes'].to(device), curr_frame['ids'].to(device)
            track_t, curr_t = past_frame['time'].to(device), curr_frame['time'].to(device)

            assign_sim =model.forward(track_app = track_feats, 
                                                             current_app = current_feats.cuda(), 
                                                             track_coords = track_coords.cuda(),
                                                             current_coords=current_coords.cuda(),
                                                             track_t = track_t,
                                                             curr_t = curr_t)

            same_id = (track_ids.view(-1, 1) == curr_ids.view(1, -1)).type(assign_sim.dtype)
            same_id = same_id.unsqueeze(0).expand(assign_sim.shape[0], -1, -1)

            loss = F.binary_cross_entropy_with_logits(assign_sim, same_id, pos_weight=torch.as_tensor(20.)) / float(len(batch))
            loss.backward()

            # Keep track of metrics
            with torch.no_grad():
                pred = (assign_sim[-1] > 0.5).view(-1).float()
                target = same_id[-1].view(-1)
                metrics = compute_class_metric(pred, target)

                for m_name, m_val in metrics.items():
                    metrics_accum[m_name] += m_val / float(len(batch))
                metrics_accum['loss'] += loss.item()

        if (i+1) %print_freq == 0 and i > 0:
            log_str = ". ".join([f"{m_name.capitalize()}: {m_val/ (print_freq if i !=0 else 1):.3f}" for m_name, m_val in metrics_accum.items()])
            print(f"Iter {i + 1}. " + log_str)
            metrics_accum = {'loss': 0., 'accuracy': 0., 'recall': 0., 'precision': 0.}
    
        optimizer.step()
    model.eval()

if __name__ == "__main__":
    # Define our model, and init 
    assign_net = AssignmentSimilarityNet(reid_network=None, # Not needed since we work with precomputed features
                                        node_dim=32, 
                                        edge_dim=64, 
                                        reid_dim=512, 
                                        edges_in_dim=6, 
                                        num_steps=10).cuda()

    # We only keep two sequences for validation. You can
    dataset = LongTrackTrainingDataset(dataset='MOT16-train_wo_val2', 
                                    db=train_db, 
                                    root_dir= osp.join(root_dir, 'data/MOT16'),
                                    max_past_frames = MAX_PATIENCE,
                                    vis_threshold=0.25)

    data_loader = DataLoader(dataset, batch_size=8, collate_fn = lambda x: x, 
                            shuffle=True, num_workers=2, drop_last=True)
    device = torch.device('cuda')
    optimizer = torch.optim.Adam(assign_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"-------- EPOCH {epoch:2d} --------")
        train_one_epoch(model = assign_net, data_loader=data_loader, optimizer=optimizer, print_freq=100)
        
        if epoch % EVAL_FREQ == 0:
            tracker =  MPNTracker(assign_net=assign_net.eval(), obj_detect=None, patience=MAX_PATIENCE)
            val_sequences = MOT16Sequences('MOT16-val2', osp.join(root_dir, 'data/MOT16'), vis_threshold=0.)
            run_tracker(val_sequences, db=train_db, tracker=tracker, output_dir=None)
            scheduler.step()

# understand the datatloader