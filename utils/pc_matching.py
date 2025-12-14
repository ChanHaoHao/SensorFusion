import torch

def points_in_box_torch(points_xyz: torch.Tensor, box7: torch.Tensor) -> torch.Tensor:
    """
    points_xyz: (N,3)
    box7: (7,) [cx,cy,cz, dx,dy,dz, yaw] in same frame
    returns: (N,) bool tensor
    """
    if points_xyz.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=points_xyz.device)

    if not torch.is_tensor(box7):
        box7 = torch.tensor(box7, device=points_xyz.device, dtype=points_xyz.dtype)
    else:
        box7 = box7.to(device=points_xyz.device, dtype=points_xyz.dtype)

    cx, cy, cz, dx, dy, dz, yaw = box7
    p = points_xyz - box7[:3]

    c = torch.cos(-yaw)
    s = torch.sin(-yaw)

    # rotate into box local frame
    x = p[:, 0] * c - p[:, 1] * s
    y = p[:, 0] * s + p[:, 1] * c
    z = p[:, 2]

    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
    return (x.abs() <= hx) & (y.abs() <= hy) & (z.abs() <= hz)

def build_match_scores(mask_points_list, boxes7, min_inside_points=20):
    """
    mask_points_list: list of (Ni,3) torch tensors
    boxes7: (M,7) torch tensor
    returns:
      scores: (K,M) float tensor, where K=len(mask_points_list)
      inside_counts: (K,M) long tensor
      mask_sizes: (K,) long tensor
    """
    device = boxes7.device
    dtype = boxes7.dtype
    K = len(mask_points_list)
    M = boxes7.shape[0]

    scores = torch.zeros((K, M), device=device, dtype=torch.float32)
    inside_counts = torch.zeros((K, M), device=device, dtype=torch.long)
    mask_sizes = torch.zeros((K,), device=device, dtype=torch.long)

    for i, pts in enumerate(mask_points_list):
        if not torch.is_tensor(pts):
            pts = torch.tensor(pts, device=device, dtype=dtype)
        else:
            pts = pts.to(device=device, dtype=dtype)

        mask_sizes[i] = pts.shape[0]
        if pts.shape[0] == 0:
            continue

        for j in range(M):
            inside = points_in_box_torch(pts, boxes7[j])
            cnt = int(inside.sum().item())
            inside_counts[i, j] = cnt

            # coverage score
            if cnt >= min_inside_points:
                scores[i, j] = cnt / max(int(pts.shape[0]), 1)
            else:
                scores[i, j] = 0.0

    return scores, inside_counts, mask_sizes

def match_masks_to_boxes(
    mask_points_list,
    boxes7,
    score_thresh=0.3,
    min_inside_points=20
):
    """
    Returns:
      matched: list of dicts with keys:
        - mask_idx, box_idx, score, inside_count
        - box (7,), mask_points (Ni,3)
      unmatched_boxes: list of dicts: box_idx, box
      unmatched_masks: list of dicts: mask_idx, mask_points
    """
    if not torch.is_tensor(boxes7):
        boxes7 = torch.tensor(boxes7, dtype=torch.float32)
    boxes7 = boxes7.contiguous()
    device = boxes7.device
    dtype = boxes7.dtype

    scores, inside_counts, mask_sizes = build_match_scores(
        mask_points_list, boxes7, min_inside_points=min_inside_points
    )

    K, M = scores.shape
    matched = []

    # greedy: repeatedly take best remaining (mask, box)
    scores_work = scores.clone()
    # logger.info(f"Matching Scores: {scores_work}")
    used_masks = set()
    used_boxes = set()

    while True:
        best = torch.max(scores_work)
        best_score = float(best.item())
        if best_score < score_thresh:
            idx = torch.argmax(scores_work).item()
            mi = idx // M
            pts = mask_points_list[mi]
            # logger.info(f"Stopping matching with best score {best_score:.3f} < threshold {score_thresh:.3f}, points {pts.shape[0]}")
            break

        idx = torch.argmax(scores_work).item()
        mi = idx // M
        bj = idx % M

        # accept
        used_masks.add(mi)
        used_boxes.add(bj)

        # package outputs
        pts = mask_points_list[mi]
        if not torch.is_tensor(pts):
            pts = torch.tensor(pts, device=device, dtype=dtype)
        else:
            pts = pts.to(device=device, dtype=dtype)

        matched.append({
            "mask_idx": mi,
            "box_idx": bj,
            "score": best_score,
            "inside_count": int(inside_counts[mi, bj].item()),
            "mask_size": int(mask_sizes[mi].item()),
            "box": boxes7[bj].detach().clone(),
            "mask_points": pts.detach().clone(),
        })

        # remove row/col
        scores_work[mi, :] = -1.0
        scores_work[:, bj] = -1.0

    # unmatched
    unmatched_boxes = []
    for bj in range(M):
        if bj not in used_boxes:
            unmatched_boxes.append({"box_idx": bj, "box": boxes7[bj].detach().clone()})

    unmatched_masks = []
    for mi in range(K):
        if mi not in used_masks:
            pts = mask_points_list[mi]
            if not torch.is_tensor(pts):
                pts = torch.tensor(pts, device=device, dtype=dtype)
            else:
                pts = pts.to(device=device, dtype=dtype)
            unmatched_masks.append({"mask_idx": mi, "mask_points": pts.detach().clone()})

    return matched, unmatched_boxes, unmatched_masks