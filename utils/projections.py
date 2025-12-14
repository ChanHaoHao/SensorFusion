import numpy as np

def project_velo_to_image(points_velo, 
                          T_velo_to_cam_4x4, 
                          R_rect_02_4x4, 
                          P_rect_02, 
                          image_shape=None):
    """
    points_velo: (N,3) or (N,4) LiDAR points in Velodyne frame
    image_shape: (H, W) if you want to filter to image bounds
    returns: u, v, depth
    """
    # Ensure (N,3)
    if points_velo.shape[1] == 4:
        points_velo = points_velo[:, :3]

    N = points_velo.shape[0]
    ones = np.ones((N, 1))
    pts_velo_h = np.hstack([points_velo, ones])  # (N,4)

    # 1) Velodyne -> cam (unrectified)
    pts_cam = (T_velo_to_cam_4x4 @ pts_velo_h.T).T  # (N,4)

    # 2) Rectification
    pts_rect = (R_rect_02_4x4 @ pts_cam.T).T        # (N,4)

    # 3) Keep only points in front of camera (Z > 0)
    z = pts_rect[:, 2]
    in_front = z > 0
    pts_rect = pts_rect[in_front]
    z = z[in_front]

    # 4) Project to image
    pts_img_h = (P_rect_02 @ pts_rect.T).T  # (N',3)

    u = pts_img_h[:, 0] / pts_img_h[:, 2]
    v = pts_img_h[:, 1] / pts_img_h[:, 2]

    # 5) Optionally filter to image size
    if image_shape is not None:
        W, H = image_shape
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid]
        v = v[valid]
        z = z[valid]

    return u, v, z

def project_uv_to_lidar(u, v, depth, 
                        T_velo_to_cam_4x4, 
                        R_rect_02_4x4, 
                        P_rect_02):
    """
    u, v: pixel coordinates
    depth: depth values corresponding to (u,v)
    returns: points_velo (N,3)
    """
    N = u.shape[0]
    ones = np.ones((N, 1))
    pts_img_h = np.hstack([u.reshape(-1,1), 
                           v.reshape(-1,1), 
                           ones])  # (N,3)

    # Inverse projection
    K = P_rect_02[:, :3]
    K_inv = np.linalg.pinv(K)

    X_cam = (K_inv @ pts_img_h.T).T  * depth.reshape(-1,1)  # (N,3)

    pts_rect_h = np.hstack([X_cam, ones])  # (N,4)

    # Inverse rectification
    R_rect_inv = np.linalg.pinv(R_rect_02_4x4)
    pts_cam = (R_rect_inv @ pts_rect_h.T).T  # (N,4)

    # Inverse transform to Velodyne
    T_cam_to_velo_4x4 = np.linalg.pinv(T_velo_to_cam_4x4)
    pts_velo_h = (T_cam_to_velo_4x4 @ pts_cam.T).T  # (N,4)

    points_velo = pts_velo_h[:, :3] / pts_velo_h[:, 3].reshape(-1,1)

    return points_velo

def boxes_to_corners_3d_lidar(boxes3d):
    """
    boxes3d: (N, 7) [x, y, z, dx, dy, dz, yaw]
    Output: corners: (N, 8, 3)
    """

    boxes3d = np.asarray(boxes3d)
    N = boxes3d.shape[0]

    # (8 corners)
    # in this order:
    # 0: x+ y+
    # 1: x+ y-
    # 2: x- y-
    # 3: x- y+
    # 4: top same order
    x_corners_norm = np.array([ 0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5, -0.5])
    y_corners_norm = np.array([ 0.5, -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5])
    z_corners_norm = np.array([-0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5])

    corners = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        x, y, z, dx, dy, dz, yaw = boxes3d[i]

        # scale normalized corners
        cx = x_corners_norm * dx
        cy = y_corners_norm * dy
        cz = z_corners_norm * dz

        # rotation about Z
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])

        corner_pts = np.vstack((cx, cy, cz))  # (3,8)
        corner_pts = R @ corner_pts           # (3,8)

        # translate to center
        corner_pts[0, :] += x
        corner_pts[1, :] += y
        corner_pts[2, :] += z

        corners[i] = corner_pts.T  # (8,3)

    return corners