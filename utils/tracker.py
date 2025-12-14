import itertools
import matplotlib.pyplot as plt
import numpy as np

def mask_iou(mask1, mask2):
    """
    mask1, mask2: (H, W) bool or 0/1 arrays
    returns IoU in [0, 1]
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0
    return intersection / union

def pointcloud_iou(pc1, pc2, voxel_size=0.2):
    """
    pc1, pc2: (N1,3) and (N2,3) point clouds
    voxel_size: size of cubic voxels in meters
    returns IoU in [0, 1]
    """
    if pc1.size == 0 or pc2.size == 0:
        return 0.0

    # Discretize / voxelize
    coords1 = np.floor(pc1 / voxel_size).astype(np.int32)
    coords2 = np.floor(pc2 / voxel_size).astype(np.int32)

    set1 = {tuple(p) for p in coords1}
    set2 = {tuple(p) for p in coords2}

    inter = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0
    return inter / union


class Tracker:
    def __init__(self):
        self.id = None
        self.mask = None
        self.color = None
        self.pointcloud = None
        self.tracker_name = None
        self.missed_frames = 0
        self.killed = False

    def update(self, mask, pointcloud, color=None, tracker_name=None):
        self.mask = mask
        self.pointcloud = pointcloud
        if color is not None:
            self.color = color
        if tracker_name is not None:
            self.tracker_name = tracker_name
            
class TrackerManager:
    def __init__(self):
        self.trackers = {}
        self.color_map = plt.get_cmap('rainbow')
        self.found_this_iter = []
        self._next_id = itertools.count(1)
        self.count = {"car": 0, "pedestrian": 0, "cyclist": 0, "traffic light": 0, "train": 0}

    def update_tracker(self, tracker_id, mask, pointcloud, name=None):
        if tracker_id in self.trackers:
            self.trackers[tracker_id].update(mask, pointcloud)
        else:
            tracker = Tracker()
            tracker.id = tracker_id
            tracker_name = f"{name}_{self.count[name]}"
            self.count[name] += 1
            tracker.update(mask, pointcloud, color=self.assign_color(tracker.id), tracker_name=tracker_name)
            self.trackers[tracker.id] = tracker

    def get_new_tracker_id(self):
        return next(self._next_id)
    
    def get_tracker(self, tracker_id):
        return self.trackers.get(tracker_id, None)
    
    def find_closest_tracker(self, mask, iou_threshold=0.3):
        best_iou = 0.0
        best_tracker_id = None
        best_tracker_name = None
        for tracker_id, tracker in self.trackers.items():
            if tracker.killed:
                continue
            if tracker_id in self.found_this_iter:
                continue  # already assigned this frame
            if tracker.mask is not None:
                iou = mask_iou(mask, tracker.mask)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_tracker_id = tracker_id
                    best_tracker_name = tracker.tracker_name
        if best_tracker_id is not None:
            self.found_this_iter.append(best_tracker_id)
        return best_tracker_id, best_tracker_name
    
    def assign_color(self, tracker_id):
        colors = [
            tuple(int(c * 255) for c in self.color_map(i)[:3])
            for i in range(255)
        ]
        return colors[tracker_id * 10 % 255]
    
    def get_color(self, tracker_id):
        return self.trackers[tracker_id].color

    def length(self):
        return len(self.trackers)
    
    def step(self):
        total_trackers = self.trackers.keys()
        for tracker_id in total_trackers:
            tracker = self.trackers[tracker_id]
            if not tracker.killed and tracker_id not in self.found_this_iter:
                tracker.missed_frames += 1
            else:
                tracker.missed_frames = 0
            
            if tracker.missed_frames >= 5:
                tracker.killed = True
        self.found_this_iter = []