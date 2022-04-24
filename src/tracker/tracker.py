import collections

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment

from src.metrics.distances import iou_matrix
from src.utils.misc import compute_distance_matrix

__all__ = [
    'Tracker',
    'TrackerIoUAssignment',
    'HungarianIoUTracker',
    'HungarianIoUTrackerB',
    'LongTermReIDHungarianTracker',
]

_UNMATCHED_COST = 255.0

class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id, feature=None, inactive=0):
		self.id = track_id
		self.box = box
		self.score = score
		self.feature = collections.deque([feature])
		self.inactive = inactive
		self.max_features_num = 10


	def add_feature(self, feature):
		"""Adds new appearance features to the object."""
		self.feature.append(feature)
		if len(self.feature) > self.max_features_num:
			self.feature.popleft()

	def get_feature(self):
		if len(self.feature) > 1:
			feature = torch.stack(list(self.feature), dim=0)
		else:
			feature = self.feature[0].unsqueeze(0)
		#return feature.mean(0, keepdim=False)
		return feature[-1]


class Tracker:
    """The main tracking file, here is where magic happens."""

    def __init__(self, obj_detect):
        self.obj_detect = obj_detect

        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        self.mot_accum = None

    def reset(self, hard=True):
        self.tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def add(self, new_boxes, new_scores):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(Track(
                new_boxes[i],
                new_scores[i],
                self.track_num + i
            ))
        self.track_num += num_new

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            box = self.tracks[0].box
        elif len(self.tracks) > 1:
            box = torch.stack([t.box for t in self.tracks], 0)
        else:
            box = torch.zeros(0).cuda()
        return box

    def data_association(self, boxes, scores):
        self.tracks = []
        self.add(boxes, scores)

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        boxes, scores = self.obj_detect.detect(frame['img'])

        self.data_association(boxes, scores)

        # results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1

    def get_results(self):
        return self.results


class TrackerIoUAssignment(Tracker):

    def data_association(self, boxes, scores):
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            
            distance = iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)

            # update existing tracks
            remove_track_ids = []
            for t, dist in zip(self.tracks, distance):
                if np.isnan(dist).all():
                    remove_track_ids.append(t.id)
                else:
                    match_id = np.nanargmin(dist)
                    t.box = boxes[match_id]
            self.tracks = [t for t in self.tracks
                           if t.id not in remove_track_ids]

            # add new tracks
            new_boxes = []
            new_scores = []
            for i, dist in enumerate(np.transpose(distance)):
                if np.isnan(dist).all():
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
            self.add(new_boxes, new_scores)

        else:
            self.add(boxes, scores)



"""
This solution (above) will only add new tracks when there is no overlap with any existing track. 
While this successfully reduces False Positives (FP), it leads to a lot of False Negatives (FN) in our setting. 
As an alternative, we present an approach that will just add all unmatched boxes as new tracks. Have a look at the results. 
How do the FPs and FNs change? What impact does this have on IDF1 and MOTA:
"""
class HungarianIoUTracker(Tracker):

    def data_association(self, boxes, scores):
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            
            # Build cost matrix.
            distance = iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)

            # Set all unmatched costs to _UNMATCHED_COST.
            distance = np.where(np.isnan(distance), _UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)

            # row_idx and col_idx are indices into track_boxes and boxes.
            # row_idx[i] and col_idx[i] define a match.
            # distance[row_idx[i], col_idx[i]] define the cost for that matching.

            # TODO: Update existing tracks and remove unmatched tracks.
            # Reminder: If the costs are equal to _UNMATCHED_COST, it's not a 
            # match.

            remove_track_ids = []
            seen_track_ids = []
            for track_idx, box_idx in zip(row_idx, col_idx):
                costs = distance[track_idx, box_idx] 
                internal_track_id = track_ids[track_idx]
                seen_track_ids.append(internal_track_id)
                if costs == _UNMATCHED_COST:
                    remove_track_ids.append(internal_track_id)
                else:
                    self.tracks[track_idx].box = boxes[box_idx]
            
            unseen_track_ids = set(track_ids) - set(seen_track_ids)
            remove_track_ids.extend(list(unseen_track_ids))
            self.tracks = [t for t in self.tracks
                           if t.id not in remove_track_ids]


            # TODO: Add new tracks.
            new_boxes = []  # <-- needs to be filled.
            new_scores = [] # <-- needs to be filled.

            for i, dist in enumerate(np.transpose(distance)):
                if np.all(dist == _UNMATCHED_COST):
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
            self.add(new_boxes, new_scores)
        else:
            # No tracks exist.
            self.add(boxes, scores)


class HungarianIoUTrackerB(Tracker):

    def data_association(self, boxes, scores):
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            
            # Build cost matrix.
            distance = iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)

            # Set all unmatched costs to _UNMATCHED_COST.
            distance = np.where(np.isnan(distance), _UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)

            # row_idx and col_idx are indices into track_boxes and boxes.
            # row_idx[i] and col_idx[i] define a match.
            # distance[row_idx[i], col_idx[i]] define the cost for that matching.

            # TODO: Update existing tracks and remove unmatched tracks.
            # Reminder: If the costs are equal to _UNMATCHED_COST, it's not a 
            # match.

            remove_track_ids = []
            seen_track_ids = []
            seen_box_idx = []
            for track_idx, box_idx in zip(row_idx, col_idx):
                costs = distance[track_idx, box_idx] 
                internal_track_id = track_ids[track_idx]
                seen_track_ids.append(internal_track_id)
                if costs == _UNMATCHED_COST:
                    remove_track_ids.append(internal_track_id)
                else:
                    self.tracks[track_idx].box = boxes[box_idx]
                    seen_box_idx.append(box_idx)
            
            unseen_track_ids = set(track_ids) - set(seen_track_ids)
            remove_track_ids.extend(list(unseen_track_ids))
            self.tracks = [t for t in self.tracks
                           if t.id not in remove_track_ids]


            # TODO: Add new tracks.
            new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
            new_boxes = [boxes[i] for i in new_boxes_idx]
            new_scores = [scores[i] for i in new_boxes_idx]

            self.add(new_boxes, new_scores)
        else:
            # No tracks exist.
            self.add(boxes, scores)





class ReIDTrackerGNN(Tracker):
	def add(self, new_boxes, new_scores, new_features):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
				new_scores[i],
				self.track_num + i,
				new_features[i]
			))
		self.track_num += num_new

	def reset(self, hard=True):
		self.tracks = []
		#self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def data_association(self, boxes, scores, features):
		raise NotImplementedError

	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		boxes = frame['det']['boxes']
		scores = frame['det']['scores']
		reid_feats= frame['det']['reid'].cpu()
		self.data_association(boxes, scores, reid_feats)

		# results
		self.update_results()


	def compute_distance_matrix(self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0):

		# Build cost matrix.
		distance = mm.distances.iou_matrix(track_boxes.numpy(), boxes.numpy(), max_iou=0.5)

		appearance_distance = compute_distance_matrix(track_features, pred_features, metric_fn=metric_fn)
		appearance_distance = appearance_distance.numpy() * 0.5
		# return appearance_distance

		assert np.alltrue(appearance_distance >= -0.1)
		assert np.alltrue(appearance_distance <= 1.1)

		combined_costs = alpha * distance + (1-alpha) * appearance_distance

		# Set all unmatched costs to _UNMATCHED_COST.
		distance = np.where(np.isnan(distance), _UNMATCHED_COST, combined_costs)

		distance = np.where(appearance_distance > 0.1, _UNMATCHED_COST, distance)

		return distance

        
class ReIDHungarianTrackerGNN(ReIDTracker):
    def data_association(self, boxes, scores, pred_features):  
        """Refactored from previous implementation to split it onto distance computation and track management"""
        if self.tracks:
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0)
            track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)
            
            distance = self.compute_distance_matrix(track_features, pred_features,
                                                    track_boxes, boxes, metric_fn=cosine_distance)
            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)            
            self.update_tracks(row_idx, col_idx,distance, boxes, scores, pred_features)
        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)
        
    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        """Updates existing tracks and removes unmatched tracks.
           Reminder: If the costs are equal to _UNMATCHED_COST, it's not a 
           match. 
        """
        track_ids = [t.id for t in self.tracks]

        unmatched_track_ids = []
        seen_track_ids = []
        seen_box_idx = []
        for track_idx, box_idx in zip(row_idx, col_idx):
            costs = distance[track_idx, box_idx] 
            internal_track_id = track_ids[track_idx]
            seen_track_ids.append(internal_track_id)
            if costs == _UNMATCHED_COST:
                unmatched_track_ids.append(internal_track_id)
            else:
                self.tracks[track_idx].box = boxes[box_idx]
                self.tracks[track_idx].add_feature(pred_features[box_idx])
                seen_box_idx.append(box_idx)

        unseen_track_ids = set(track_ids) - set(seen_track_ids)
        unmatched_track_ids.extend(list(unseen_track_ids))
        self.tracks = [t for t in self.tracks
                       if t.id not in unmatched_track_ids]


        # Add new tracks.
        new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)



class LongTermReIDHungarianTracker(ReIDHungarianTracker):
    def __init__(self, patience, *args, **kwargs):
        """ Add a patience parameter"""
        self.patience=patience
        super().__init__(*args, **kwargs)

    def update_results(self):
        """Only store boxes for tracks that are active"""
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            if t.inactive == 0: # Only change
                self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1        
        
    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        track_ids = [t.id for t in self.tracks]

        unmatched_track_ids = []
        seen_track_ids = []
        seen_box_idx = []
        for track_idx, box_idx in zip(row_idx, col_idx):
            costs = distance[track_idx, box_idx] 
            internal_track_id = track_ids[track_idx]
            seen_track_ids.append(internal_track_id)
            if costs == _UNMATCHED_COST:
                unmatched_track_ids.append(internal_track_id)

            else:
                self.tracks[track_idx].box = boxes[box_idx]
                self.tracks[track_idx].add_feature(pred_features[box_idx])
                
                # Note: the track is matched, therefore, inactive is set to 0
                self.tracks[track_idx].inactive=0
                seen_box_idx.append(box_idx)
                

        unseen_track_ids = set(track_ids) - set(seen_track_ids)
        unmatched_track_ids.extend(list(unseen_track_ids))
        ##################
        ### TODO starts
        ##################
        
        # Update the `inactive` attribute for those tracks that have been 
        # not been matched. kill those for which the inactive parameter 
        # is > self.patience

        kill_tracks = []
        for track_idx, t in enumerate(self.tracks):
            if t.id in unmatched_track_ids:
                self.tracks[track_idx].inactive += 1

            if t.inactive > self.patience:
                kill_tracks.append(t.id)

        self.tracks = [t for t in self.tracks
                       if t.id not in kill_tracks]
        
        ##################
        ### TODO ends
        ##################        
        
        new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)


class MPNTracker(LongTermReIDHungarianTracker):
    def __init__(self, assign_net, *args, **kwargs):
        self.assign_net = assign_net
        super().__init__(*args, **kwargs)
        
    def data_association(self, boxes, scores, pred_features):  
        if self.tracks:  
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0).cuda()
            track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)
            
            # Hacky way to recover the timestamps of boxes and tracks
            curr_t = self.im_index * torch.ones((pred_features.shape[0],)).cuda()
            track_t = torch.as_tensor([self.im_index - t.inactive - 1 for t in self.tracks]).cuda()

            ########################
            #### TODO starts
            ########################
            
            # Do a forward pass through self.assign_net to obtain our costs.
            # Note: self.assign_net will return unnormalized probabilities. 
            # Make sure to apply the sigmoid function to them!
            

            # TODO: Forward pass through self.assign_net
            # TODO: Document this a bit better
            
            pred_sim = torch.sigmoid(self.assign_net.forward(track_app = track_features.cuda(), 
                                                             current_app = pred_features.cuda(), 
                                                             track_coords = track_boxes.cuda(),
                                                             current_coords=boxes.cuda(),
                                                             track_t = track_t,
                                                             curr_t = curr_t)
                                    ).detach().cpu().numpy()
            
            ########################
            #### TODO ends
            ########################

            pred_sim = pred_sim[-1]  # Use predictions at last message passing step
            distance = (1- pred_sim) 
            
            # Do not allow mataches when sim < 0.5, to avoid low-confident associations
            distance = np.where(pred_sim < 0.5, _UNMATCHED_COST, distance) 

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)            
            self.update_tracks(row_idx, col_idx,distance, boxes, scores, pred_features)

            
        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)