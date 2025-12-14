# Early vs. Late Camera-LiDAR Fusion in 3D Object Detection and Tracking

## Introduction
Autonomous vehicles depend on multimodal sensing to achieve accurate environmental perception. Among these modalities, cameras provide detailed appearance information, whereas LiDAR delivers precise depth and geometric structure. Fusing data from these sensors enables a more reliable and comprehensive understanding of the surrounding environment by leveraging their complementary capabilities.

### Video for SAM3 detection and PV-RCNN only
![SAM3 and PV-RCNN only](./videos/merged_sam3_openpcdet.gif)

Camera-based perception degrades significantly under challenging conditions such as low-light or nighttime environments, glare, shadows, snow, and dust, which directly reduces segmentation accuracy and reliability. Moreover, when extracting LiDAR points corresponding to segmented vehicles, the sparsity of the point cloud makes it difficult to reliably infer vehicle orientation using LiDAR alone. In 3D point cloud–based detection, false positives occur frequently; while the method can often estimate object orientation accurately, it tends to misclassify or merge targets when multiple objects are spatially close.

## Why fuse RGB and LiDAR info?
In this project, we compare two sensor fusion strategies -- early fusion and late fusion -- for 3D object detection. Using KITTI dataset, SAM3 (2D image segmentation model) and PV-RCNN (3D object detection model) to evaluate each approach.

### Early Fusion
Early fusion combines raw or low-level features from different sensors at the begin of the detection pipeline. In this project, the pipeline for early fusion is:

1. Get the masks of each target features by passing the prompts and images to SAM3
2. Project the LiDAR points into the image frame
3. Find all the LiDAR points in each masks
4. Try to use these points to do object tracking by comparing the IoU of objects at different timestep, and estimate the pose of the object.

### Late Fusion
Late fusion keeps sensor processing streams seperate and combines their high-level outputs. In this project, the pipeline for late fusion is:

1. Get the masks of each target features by passing the prompts and images to SAM3
2. Use PV-RCNN to get 3D detections from LiDAR pointcloud
3. Combine detections using IoU matching

### Video for Early Fusion and Late Fusion
![Video for Early Fusion and Late Fusion](./videos/merged_early_late_fusion.gif)

In early fusion, failures in camera-based perception propagate directly to the fused representation: if the camera does not detect distant pedestrians or cyclists, these objects are not marked even when corresponding LiDAR points are present. In contrast, late fusion combines high-level outputs from each modality, enabling the suppression of false detections by discarding LiDAR-based detections with low confidence scores and no associated segmentation masks, thereby improving overall detection accuracy. Additionally, because PV-RCNN is not trained to recognize certain object categories such as trains, it fails to detect them using LiDAR alone. However, by integrating camera-based segmentation with LiDAR point clouds in a late-fusion framework, such previously unseen objects can still be successfully identified.

### Next steps
Dive into BEVFusion and see how encoding LiDAR points into voxel space and embedded features from image encoders work can merge into a joint feature space and generate semantic masks, 3D pointcloud detection, and even end-to-end trajectory planning simultaneously.

### References
* [Early vs. Late Camera-LiDAR Fusion in 3D Object Detection: A Performance Study](https://medium.com/@az.tayyebi/early-vs-late-camera-lidar-fusion-in-3d-object-detection-a-performance-study-5fb1688426f9)
* [SAM3](https://github.com/facebookresearch/sam3)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/)
