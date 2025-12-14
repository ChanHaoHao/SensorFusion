import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from openpcdet.pcdet.config import cfg, cfg_from_yaml_file
from openpcdet.pcdet.datasets import DatasetTemplate
from openpcdet.pcdet.models import build_network, load_data_to_gpu
from openpcdet.pcdet.utils import common_utils

import plotly.graph_objects as go
from PIL import Image
from scipy.ndimage import binary_erosion

import utils.projections as proj_utils
import utils.load_calib as calib_utils
import utils.visualization as vis_utils
import utils.pc_matching as pc_matching_utils

logger = common_utils.create_logger()

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='openpcdet/cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def read_pc_and_img_files(pc_root_path='2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/', image_root_path='2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'):
    point_cloud_files = sorted(os.listdir(pc_root_path))
    image_files = sorted(os.listdir(image_root_path))
    
    return point_cloud_files, image_files

def load_transformation_matrix(vel_to_cam_path='2011_09_26/calib_velo_to_cam.txt', cam_to_cam_path='2011_09_26/calib_cam_to_cam.txt'):
    # load lidar to camera transformation matrix
    vel_to_cam = calib_utils.read_vel_to_cam(vel_to_cam_path)

    # load camera intrinsics and rectification matrix
    P_rect_02, R_rect_02 = calib_utils.read_cam_to_cam(cam_to_cam_path)
    
    return vel_to_cam, R_rect_02, P_rect_02

def main():
    args, cfg = parse_config()
    output_root_path = 'results'
    os.makedirs(output_root_path, exist_ok=True)
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    # pc_root_path = '2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/'
    image_root_path = '2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
    
    vel_to_cam, R_rect_02, P_rect_02 = load_transformation_matrix()
    _, image_files = read_pc_and_img_files()
    
    masks_path = "masks/0001/"
    masks_files = sorted(os.listdir(masks_path))

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'======================== Processing sample index: \t{idx + 1}/{len(demo_dataset)} ========================')
            masks_file_paths = [mask for mask in masks_files if f"img_{idx}_" in mask]
            
            image_path = os.path.join(image_root_path, image_files[idx])
            image = Image.open(image_path).convert("RGB")
            
            rgb_fig = vis_utils.create_rgb_image(image)
            
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            
            points = data_dict['points'].cpu().numpy()  # (N,5)
            points = points[:, 1:4]
            u, v, d = proj_utils.project_velo_to_image(
                points_velo=points,
                T_velo_to_cam_4x4=vel_to_cam,
                R_rect_02_4x4=R_rect_02,
                P_rect_02=P_rect_02,
                image_shape=(1242, 375)
            )
            cam_view_points = proj_utils.project_uv_to_lidar(
                u=u,
                v=v,
                depth=d,
                T_velo_to_cam_4x4=vel_to_cam,
                R_rect_02_4x4=R_rect_02,
                P_rect_02=P_rect_02
            )
            
            cam_view_points = torch.tensor(cam_view_points, dtype=torch.float32).cuda()
            zeros_front = torch.zeros((cam_view_points.shape[0], 1), dtype=cam_view_points.dtype, device=cam_view_points.device)
            zeros_back  = torch.zeros((cam_view_points.shape[0], 1), dtype=cam_view_points.dtype, device=cam_view_points.device)
            data_dict['points'] = torch.cat((zeros_front, cam_view_points, zeros_back), dim=1)

            masks_points_list = []
            for mask in masks_file_paths:
                mask_array = np.load(os.path.join(masks_path, mask))  # H x W, bool
                ones = np.sum(mask_array)
                if ones > 500:
                    mask_array = binary_erosion(mask_array, structure=np.ones((5,5))).astype(bool)
                                        
                mask_u = []
                mask_v = []
                mask_depth = []
                for i in range(u.shape[0]):
                    if mask_array[v.astype(int)[i], u.astype(int)[i]]:
                        mask_u.append(u[i])
                        mask_v.append(v[i])
                        mask_depth.append(d[i])
                
                rgb_fig = vis_utils.rgb_add_2dtrace(
                    fig=rgb_fig,
                    x=np.array(mask_u),
                    y=np.array(mask_v),
                    color="yellow",
                    mode='markers',
                    marker_size=1,
                )
                
                mask_points_velo = proj_utils.project_uv_to_lidar(
                    u=np.array(mask_u),
                    v=np.array(mask_v),
                    depth=np.array(mask_depth),
                    T_velo_to_cam_4x4=vel_to_cam,
                    R_rect_02_4x4=R_rect_02,
                    P_rect_02=P_rect_02
                )  # (N,3)
                
                masks_points_list.append(mask_points_velo)
            
            
            pred_dicts, _ = model.forward(data_dict)
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            
            pc_fig = vis_utils.create_pc_figure(points)
            
            # Box corners
            matched, un_boxes, un_masks = pc_matching_utils.match_masks_to_boxes(
                mask_points_list=masks_points_list,
                boxes7=pred_boxes,          # (M,7)
                score_thresh=0.3,            # tune
                min_inside_points=0         # tune
            )
            
            for match in matched:
                box = match['box'].cpu().numpy().reshape(1,7)
                mask_points = match['mask_points'].cpu().numpy()  # (N,3)
                
                pc_fig = vis_utils.pc_add_3dtrace(
                    pc_fig,
                    mask_points,
                    color="green",
                )
                
                rgb_u, rgb_v, _ = proj_utils.project_velo_to_image(
                    points_velo=mask_points,
                    T_velo_to_cam_4x4=vel_to_cam,
                    R_rect_02_4x4=R_rect_02,
                    P_rect_02=P_rect_02,
                )
                rgb_fig = vis_utils.rgb_add_2dtrace(
                    fig=rgb_fig,
                    x=rgb_u,
                    y=rgb_v,
                    color="green",
                    marker_size=1,
                )
                
                corners = proj_utils.boxes_to_corners_3d_lidar(box)  # (1,8,3)
                pc_fig = vis_utils.plot_boxes_plotly(pc_fig, corners, color="lime", name="matched_box")

                u, v, _ = proj_utils.project_velo_to_image(
                    points_velo=corners[0],
                    T_velo_to_cam_4x4=vel_to_cam,
                    R_rect_02_4x4=R_rect_02,
                    P_rect_02=P_rect_02,
                )
                corners_uv = np.vstack((u, v)).T  # (8,2)
                rgb_fig = vis_utils.plot_boxes_on_image_plotly(rgb_fig, [corners_uv], color="lime", name="matched_box")
                    
            for unmatched_box in un_boxes:
                box = unmatched_box['box'].cpu().numpy().reshape(1,7)
                corners = proj_utils.boxes_to_corners_3d_lidar(box)  # (1,8,3)
                pc_fig = vis_utils.plot_boxes_plotly(pc_fig, corners, color="red", name="unmatched_box")
                
            for unmatched_mask in un_masks:
                mask_points = unmatched_mask["mask_points"].cpu().numpy()  # (N,3)
                unmatched_idx = unmatched_mask["mask_idx"]
                mask_array = np.load(os.path.join(masks_path, masks_file_paths[unmatched_idx]))  # H x W, bool
                u, v = np.where(mask_array>0)
                
                rgb_fig = vis_utils.rgb_add_2dtrace(
                    fig=rgb_fig,
                    x=v,
                    y=u,
                    color="blue",
                    marker_size=1,
                )

                pc_fig = vis_utils.pc_add_3dtrace(
                    pc_fig,
                    mask_points,
                    color="blue",
                )
            
            logger.info(f'Saving results to {output_root_path}/image_boxes_{idx}.png and {output_root_path}/demo_output_{idx}.png')
            rgb_fig.write_image(f"{output_root_path}/image_boxes_{idx}.png")
            pc_fig.write_image(f"{output_root_path}/demo_output_{idx}.png")

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
