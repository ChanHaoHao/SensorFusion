import torch
import os
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
from scipy.ndimage import binary_erosion
from utils.tracker import TrackerManager

import utils.load_calib as calib_utils
import utils.projections as proj_utils
import utils.visualization as vis_utils

print(torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run on a machine with a CUDA-capable GPU.")

track = TrackerManager()

def load_transformation_matrix(vel_to_cam_path='2011_09_26/calib_velo_to_cam.txt', cam_to_cam_path='2011_09_26/calib_cam_to_cam.txt'):
    # load lidar to camera transformation matrix
    vel_to_cam = calib_utils.read_vel_to_cam(vel_to_cam_path)

    # load camera intrinsics and rectification matrix
    P_rect_02, R_rect_02 = calib_utils.read_cam_to_cam(cam_to_cam_path)
    
    return vel_to_cam, R_rect_02, P_rect_02

def load_model():
    # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    return model, processor

def read_pc_and_img_files(pc_root_path, image_root_path):
    point_cloud_files = sorted(os.listdir(pc_root_path))
    image_files = sorted(os.listdir(image_root_path))
    
    return point_cloud_files, image_files

def main():
    pc_root_path = '2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/'
    image_root_path = '2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
    output_root_path = 'results'
    os.makedirs(output_root_path, exist_ok=True)
    
    vel_to_cam, R_rect_02, P_rect_02 = load_transformation_matrix()
    model, processor = load_model()
    point_cloud_files, image_files = read_pc_and_img_files(pc_root_path, image_root_path)
    
    # Prompt for SAM3 inference
    text_prompts = ["car", "pedestrian", "cyclist", "traffic light", "train"]
    color_prompts = {"car": "red", "pedestrian": "blue", "cyclist": "green", "traffic light": "yellow", "train": "purple"}

    for index in range(len(point_cloud_files)):
        file_name = point_cloud_files[index]
        if not file_name.endswith('.bin'):
            continue
        path = os.path.join(pc_root_path, file_name)
        points = calib_utils.read_kitti_bin(path)
        u, v, depth = proj_utils.project_velo_to_image(points, vel_to_cam, R_rect_02, P_rect_02, (1242, 375))
        cam_view_points = proj_utils.project_uv_to_lidar(u, v, depth, vel_to_cam, R_rect_02, P_rect_02)

        # Load an image
        image_path = os.path.join(image_root_path, image_files[index])
        image = Image.open(image_path).convert("RGB")
        inference_state = processor.set_image(image)

        rgb_fig = vis_utils.create_rgb_image(image)

        detected_points = {}
        assigned_colors = []

        # Perform SAM3 inference for each text prompt
        for prompt in text_prompts:
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

            # Get the masks, bounding boxes, and scores
            masks = output["masks"]

            n_masks = masks.shape[0]
            if n_masks == 0:
                continue

            # image = np.array(image) # H x W x 3
            for instance_id in range(n_masks):
                mask = masks[instance_id].detach().cpu().numpy()  # 1 x H x W, bool
                mask = mask[0]  # H x W
                print(np.sum(mask), "pixels in mask for instance", instance_id, "of prompt", prompt)
                
                if np.sum(mask) < 100:
                    print("Too few pixels don't erode.")
                else:
                    print(f"Mask has {np.sum(mask)} pixels before erosion.")
                    mask = binary_erosion(mask, structure=np.ones((5,5))).astype(bool)
                    print(f"Mask has {np.sum(mask)} pixels after erosion.")
                
                # Find LiDAR points that project into the mask
                target_points_u = []
                target_points_v = []
                target_points_z = []
                for i in range(u.shape[0]):
                    if mask[v.astype(int)[i], u.astype(int)[i]]:
                        target_points_u.append(u[i])
                        target_points_v.append(v[i])
                        target_points_z.append(depth[i])
                
                # Project back to LiDAR coordinates
                target_lidar_points = proj_utils.project_uv_to_lidar(
                    np.array(target_points_u), 
                    np.array(target_points_v), 
                    np.array(target_points_z),
                    vel_to_cam,
                    R_rect_02,
                    P_rect_02
                )
                
                if len(target_lidar_points) == 0:
                    print("No LiDAR points found for this instance, skipping.")
                    continue
                
                tracker_id, tracker_name = track.find_closest_tracker(mask)
                if tracker_id is not None:
                    print(f"Found matching tracker ID {tracker_id} for prompt '{prompt}' instance {instance_id}.")
                else:
                    tracker_id = track.get_new_tracker_id()
                    print(f"Creating new tracker ID {tracker_id} for prompt '{prompt}' instance {instance_id}.")
                track.update_tracker(tracker_id, mask, target_lidar_points, name=prompt)
                if tracker_name is None:
                    tracker_name = track.get_tracker(tracker_id).tracker_name

                detected_points[tracker_name] = target_lidar_points
                assigned_colors.append(track.get_color(tracker_id))
                
                rgb_fig = vis_utils.rgb_add_2dtrace(
                    rgb_fig,
                    np.array(target_points_u).astype(float),
                    np.array(target_points_v).astype(float),
                    color_prompts[prompt],
                )
                
            track.step()
        print("===================================================================================================")
        

        pc_fig = vis_utils.create_pc_figure(cam_view_points)

        for i, (key, pts) in enumerate(detected_points.items()):
            color = f"rgb{assigned_colors[i]}"
            pc_fig = vis_utils.pc_add_3dtrace(
                pc_fig,
                pts,
                color=color,
                name=key,
            )

        pc_fig.write_image(f"{output_root_path}/pointcloud_view_{index}.png")
        rgb_fig.write_image(f"{output_root_path}/rgb_with_boxes_{index}.png")
        
if __name__ == "__main__":
    main()