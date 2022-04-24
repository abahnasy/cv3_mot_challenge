import torch.nn as nn
import torch.nn.functional as F

from .mpn import BipartiteNeuralMessagePassingLayer


class AssignmentSimilarityNet(nn.Module):
    def __init__(self, reid_network, node_dim, edge_dim, reid_dim, edges_in_dim, num_steps, dropout=0.):
        super().__init__()
        self.reid_network = reid_network
        self.graph_net = BipartiteNeuralMessagePassingLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
        self.num_steps = num_steps
        self.cnn_linear = nn.Linear(reid_dim, node_dim)
        self.edge_in_mlp = nn.Sequential(*[nn.Linear(edges_in_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(edge_dim, edge_dim), nn.ReLU(),nn.Dropout(dropout)])
        self.classifier = nn.Sequential(*[nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1)])
        
    
    def compute_edge_feats(self, track_coords, current_coords, track_t, curr_t):    
        """
        Computes initial edge feature tensor

        Args:
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)
                          
            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)        
            
        
        Returns:
            tensor with shape (num_trakcs, num_boxes, 5) containing pairwise
            position and time difference features 
        """

        ########################
        #### TODO starts
        ########################
        
        # NOTE 1: we recommend you to use box centers to compute distances
        # in the x and y coordinates.

        # NOTE 2: Check out the  code inside train_one_epoch function and 
        # LongTrackTrainingDataset class a few cells below to debug this
        
        track_centers = track_coords.view(-1, 2, 2).mean(-2)
        curr_centers = current_coords.view(-1, 2, 2).mean(-2)

        track_w = track_coords[..., 2] - track_coords[..., 0]
        track_h = track_coords[..., 3] - track_coords[..., 1]

        curr_w = current_coords[..., 2] - current_coords[..., 0]
        curr_h = current_coords[..., 3] - current_coords[..., 1]

        assert (track_w >0).all() and (track_h >0).all()
            
        denom = (track_w.unsqueeze(1) + curr_w.unsqueeze(0)) / 2
        dist_x = (track_centers[..., 0].unsqueeze(1) - curr_centers[..., 0].unsqueeze(0))
        dist_y = (track_centers[..., 1].unsqueeze(1) - curr_centers[..., 1].unsqueeze(0))
        
        dist_x = dist_x / denom
        dist_y = dist_y / denom

        dist_w = torch.log(track_w.unsqueeze(1) / curr_w.unsqueeze(0))
        dist_h = torch.log(track_h.unsqueeze(1) / curr_h.unsqueeze(0))

        dist_t = (track_t.unsqueeze(1) - curr_t.unsqueeze(0)).type(dist_h.dtype).to(dist_h.device)

        edge_feats = torch.stack([dist_x, dist_y, dist_w, dist_h, dist_t])
        edge_feats = edge_feats.permute(1, 2, 0)

        ########################
        #### TODO ends
        ########################

        return edge_feats # has shape (num_trakcs, num_boxes, 5)


    def forward(self, track_app, current_app, track_coords, current_coords, track_t, curr_t):
        """
        Args:
            track_app: track's reid embeddings, torch.Tensor with shape (num_tracks, 512)
            current_app: current frame detections' reid embeddings, torch.Tensor with shape (num_boxes, 512)
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)
                          
            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)
            
        Returns:
            classified edges: torch.Tensor with shape (num_steps, num_tracks, num_boxes),
                             containing at entry (step, i, j) the unnormalized probability that track i and 
                             detection j are a match, according to the classifier at the given neural message passing step
        """
        
        # Get initial edge embeddings to
        dist_reid = cosine_distance(track_app, current_app)
        pos_edge_feats = self.compute_edge_feats(track_coords, current_coords, track_t, curr_t)
        edge_feats = torch.cat((pos_edge_feats, dist_reid.unsqueeze(-1)), dim=-1)
        edge_embeds = self.edge_in_mlp(edge_feats)
        initial_edge_embeds = edge_embeds.clone()

        # Get initial node embeddings, reduce dimensionality from 512 to node_dim
        track_embeds = F.relu(self.cnn_linear(track_app))
        curr_embeds =F.relu(self.cnn_linear(current_app))

        classified_edges = []
        for _ in range(self.num_steps):
            edge_embeds = torch.cat((edge_embeds, initial_edge_embeds), dim=-1)            
            edge_embeds, track_embeds, curr_embeds = self.graph_net(edge_embeds=edge_embeds, 
                                                                    nodes_a_embeds=track_embeds, 
                                                                    nodes_b_embeds=curr_embeds)

            classified_edges.append(self.classifier(edge_embeds))

        return torch.stack(classified_edges).squeeze(-1)