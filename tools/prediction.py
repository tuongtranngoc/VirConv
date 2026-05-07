"""
Predict and visualize 3D object detection results on camera images.

Usage (run from tools/):
    python prediction.py --cfg_file cfgs/models/kitti/VirConv-S.yaml \\
                         --ckpt VirConv-S.pth \\
                         --split val \\
                         --score_thresh 0.5 \\
                         --nms_method nms \\
                         --nms_iou_thresh 0.5 \\
                         --output_dir ../output/predictions

    # Specific frames + WBF:
    python prediction.py ... --sample_idx 000008 000010 000050 --nms_method wbf

    # With LiDAR overlay + Soft-NMS:
    python prediction.py ... --show_lidar --nms_method soft_nms

Post-processing methods (applied AFTER the model's built-in NMS):
    none     - use model output as-is
    nms      - per-class rotated BEV NMS (GPU)            [default]
    soft_nms - per-class Soft-NMS with Gaussian score decay
    wbf      - per-class Weighted Box Fusion (merges overlapping boxes)
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.model_utils.model_nms_utils import compute_WBF
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils
from pcdet.utils.box_utils import boxes_to_corners_3d

# Predefined BGR colors for all standard KITTI classes
KNOWN_CLASS_COLORS = {
    'Car':             (0,   255,   0),   # green
    'Pedestrian':      (255,  80,   0),   # blue
    'Cyclist':         (0,   165, 255),   # orange
    'Van':             (0,   200, 130),   # teal
    'Truck':           (180, 255,   0),   # lime
    'Tram':            (255,   0, 200),   # magenta
    'Misc':            (180, 180,   0),   # olive
    'Person_sitting':  (0,   100, 255),   # amber
}
# Fallback palette for classes not in KNOWN_CLASS_COLORS
_AUTO_PALETTE = [
    (0, 255, 255), (255, 255, 0), (128, 0, 255),
    (255, 0, 128), (0, 128, 255), (255, 128, 0),
]


def build_color_map(class_names: List[str]) -> dict:
    """Return {class_name: BGR_color} for every class in *class_names*."""
    color_map = {}
    auto_idx = 0
    for name in class_names:
        if name in KNOWN_CLASS_COLORS:
            color_map[name] = KNOWN_CLASS_COLORS[name]
        else:
            color_map[name] = _AUTO_PALETTE[auto_idx % len(_AUTO_PALETTE)]
            auto_idx += 1
    return color_map

# 3D box edge pairs — indices into the 8-corner array from boxes_to_corners_3d
#        7 -------- 4
#       /|         /|
#      6 -------- 5 .
#      | |        | |
#      . 3 -------- 0
#      |/         |/
#      2 -------- 1
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical pillars
]
# X drawn on the front face (corners 0,1,4,5) to indicate heading
FRONT_CROSS = [(0, 5), (1, 4)]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def corners_to_image(corners_lidar: np.ndarray, calib) -> Optional[np.ndarray]:
    """
    Project 8 LiDAR-frame 3D corners onto the camera image plane.

    Returns (8, 2) int32 pixel coords, or None if any corner is behind camera.
    """
    corners_rect = calib.lidar_to_rect(corners_lidar)      # (8, 3) in camera rect
    if (corners_rect[:, 2] <= 0).any():
        return None
    pts_img, _ = calib.rect_to_img(corners_rect)           # (8, 2) float
    return pts_img.astype(np.int32)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_box_3d(img: np.ndarray, corners_img: np.ndarray,
                color: tuple, label: Optional[str] = None,
                score: Optional[float] = None, thickness: int = 2) -> None:
    """Draw a projected 3D bounding box wireframe on *img* in-place."""
    for i, j in BOX_EDGES:
        cv2.line(img, tuple(corners_img[i]), tuple(corners_img[j]),
                 color, thickness, cv2.LINE_AA)

    for i, j in FRONT_CROSS:
        cv2.line(img, tuple(corners_img[i]), tuple(corners_img[j]),
                 color, max(1, thickness - 1), cv2.LINE_AA)

    if label is not None or score is not None:
        parts = []
        if label:
            parts.append(label)
        if score is not None:
            parts.append(f'{score:.2f}')
        text = ' '.join(parts)
        anchor = corners_img[np.argmin(corners_img[:, 1])]  # topmost projected corner
        # dark background for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x0, y0 = int(anchor[0]), int(anchor[1]) - 2
        cv2.rectangle(img, (x0, y0 - th - 2), (x0 + tw, y0 + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_legend(img: np.ndarray, color_map: dict) -> None:
    """Draw a class-colour legend in the top-right corner of *img* in-place."""
    pad, swatch, gap, font_scale = 6, 12, 4, 0.4
    line_h = swatch + gap
    total_h = pad + len(color_map) * line_h + pad

    # measure widest label
    max_w = max(
        cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for name in color_map
    )
    box_w = pad + swatch + gap + max_w + pad
    x0 = img.shape[1] - box_w - 6
    y0 = 6
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + total_h), (30, 30, 30), -1)

    for k, (name, color) in enumerate(color_map.items()):
        ys = y0 + pad + k * line_h
        cv2.rectangle(img, (x0 + pad, ys), (x0 + pad + swatch, ys + swatch), color, -1)
        cv2.putText(img, name,
                    (x0 + pad + swatch + gap, ys + swatch - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)


def overlay_lidar(img: np.ndarray, points: np.ndarray, calib,
                  max_depth: float = 60.0) -> None:
    """Project LiDAR points onto *img* in-place, coloured by depth (blue→red)."""
    pts_rect = calib.lidar_to_rect(points[:, :3])
    valid = pts_rect[:, 2] > 0
    pts_rect = pts_rect[valid]
    depth = pts_rect[:, 2]

    pts_img, _ = calib.rect_to_img(pts_rect)
    h, w = img.shape[:2]
    in_fov = (
        (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) &
        (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
    )
    pts_img = pts_img[in_fov].astype(np.int32)
    depth_n = np.clip(depth[in_fov] / max_depth, 0.0, 1.0)

    for (u, v), d in zip(pts_img, depth_n):
        color = (int(255 * d), 0, int(255 * (1 - d)))   # far=blue, close=red
        cv2.circle(img, (int(u), int(v)), 1, color, -1)


# ---------------------------------------------------------------------------
# Per-sample visualization
# ---------------------------------------------------------------------------

def visualize_predictions(image_rgb: np.ndarray,
                          pred_boxes: np.ndarray,
                          pred_scores: np.ndarray,
                          pred_labels: np.ndarray,
                          calib,
                          class_names: List[str],
                          color_map: dict,
                          score_thresh: float = 0.5,
                          gt_boxes: Optional[np.ndarray] = None,
                          lidar_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Render detections on the camera image.

    Args:
        image_rgb:   H×W×3 uint8 RGB image
        pred_boxes:  (N, 7) predicted boxes in LiDAR coords
        pred_scores: (N,)   confidence scores
        pred_labels: (N,)   1-indexed class labels (matching *class_names*)
        calib:       Calibration object
        class_names: ordered list of class name strings (1-indexed by pred_labels)
        color_map:   {class_name: BGR_color} built from build_color_map()
        score_thresh: minimum score to display
        gt_boxes:    (M, 7) optional ground-truth boxes in LiDAR coords
        lidar_points:(P, 4) optional raw LiDAR point cloud for overlay

    Returns:
        BGR image with annotations
    """
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if lidar_points is not None:
        overlay_lidar(vis, lidar_points, calib)

    # Ground-truth boxes in white/thin outline
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_corners_all = boxes_to_corners_3d(gt_boxes)    # (M, 8, 3)
        for corners in gt_corners_all:
            pts = corners_to_image(corners, calib)
            if pts is None:
                continue
            draw_box_3d(vis, pts, color=(200, 200, 200), label='GT', thickness=1)

    # Predicted boxes — one colour per class
    keep = pred_scores >= score_thresh
    for box, score, label_idx in zip(pred_boxes[keep], pred_scores[keep], pred_labels[keep]):
        cls = class_names[int(label_idx) - 1]
        color = color_map.get(cls, (0, 255, 255))   # cyan fallback
        corners = boxes_to_corners_3d(box[np.newaxis])[0]  # (8, 3)
        pts = corners_to_image(corners, calib)
        if pts is None:
            continue
        draw_box_3d(vis, pts, color=color, label=cls, score=float(score))

    # Legend for all active classes
    draw_legend(vis, color_map)

    return vis


# ---------------------------------------------------------------------------
# Post-processing: choose best prediction per object
# ---------------------------------------------------------------------------

def _per_class_indices(pred_labels: np.ndarray):
    """Yield (class_id, boolean_mask) for each unique label."""
    for cls_id in np.unique(pred_labels):
        yield cls_id, pred_labels == cls_id


def apply_nms(pred_boxes: np.ndarray, pred_scores: np.ndarray,
              pred_labels: np.ndarray, iou_thresh: float = 0.5):
    """
    Per-class rotated BEV NMS using the CUDA kernel already in the repo.
    Boxes sorted by score descending; lower-scoring overlapping boxes removed.
    """
    keep = []
    for _, mask in _per_class_indices(pred_labels):
        orig_idxs = np.where(mask)[0]
        cls_boxes  = torch.from_numpy(pred_boxes[mask]).float().cuda()
        cls_scores = torch.from_numpy(pred_scores[mask]).float().cuda()
        keep_local, _ = iou3d_nms_utils.nms_gpu(cls_boxes, cls_scores, iou_thresh)
        keep.extend(orig_idxs[keep_local.cpu().numpy()].tolist())
    keep = sorted(keep)
    return pred_boxes[keep], pred_scores[keep], pred_labels[keep]


def apply_soft_nms(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                   pred_labels: np.ndarray, iou_thresh: float = 0.5,
                   sigma: float = 0.5, min_score: float = 0.001):
    """
    Per-class Soft-NMS with Gaussian score decay (Bodla et al., 2017).
    Instead of hard removal, overlapping boxes have their scores reduced by
      s_j <- s_j * exp(-(iou(b_i, b_j)^2) / sigma)
    Boxes falling below *min_score* after decay are discarded.
    """
    out_boxes, out_scores, out_labels = [], [], []
    for cls_id, mask in _per_class_indices(pred_labels):
        b = pred_boxes[mask].copy()   # (M, 7)
        s = pred_scores[mask].copy()  # (M,)
        N = len(s)
        for i in range(N):
            # swap current position with highest-scored remaining box
            best = i + int(np.argmax(s[i:]))
            b[[i, best]] = b[[best, i]]
            s[[i, best]] = s[[best, i]]
            # BEV IoU of box i against all remaining boxes
            ious = iou3d_nms_utils.boxes_bev_iou_cpu(b[i:i+1], b[i+1:])[0]  # (N-i-1,)
            # Gaussian decay — only suppress boxes above the IoU threshold
            decay = np.where(ious > iou_thresh,
                             np.exp(-(ious ** 2) / sigma),
                             np.ones_like(ious))
            s[i+1:] *= decay
        valid = s >= min_score
        out_boxes.append(b[valid])
        out_scores.append(s[valid])
        out_labels.append(np.full(valid.sum(), cls_id, dtype=pred_labels.dtype))

    if not out_boxes:
        return pred_boxes[:0], pred_scores[:0], pred_labels[:0]
    out_boxes   = np.concatenate(out_boxes)
    out_scores  = np.concatenate(out_scores)
    out_labels  = np.concatenate(out_labels)
    order = np.argsort(out_scores)[::-1]
    return out_boxes[order], out_scores[order], out_labels[order]


def apply_wbf(pred_boxes: np.ndarray, pred_scores: np.ndarray,
              pred_labels: np.ndarray, class_names: List[str],
              iou_thresh: float = 0.85, score_thresh: float = 0.001):
    """
    Per-class Weighted Box Fusion (Solovyev et al., 2021).
    Overlapping boxes are clustered by BEV IoU and merged into a single box
    whose position/size is the score-weighted mean of all members in the cluster.
    """
    out_boxes, out_scores, out_labels = [], [], []
    for cls_id, mask in _per_class_indices(pred_labels):
        cls_name = class_names[int(cls_id) - 1]
        names  = np.array([cls_name] * mask.sum())
        merged_names, merged_scores, merged_boxes = compute_WBF(
            names,
            pred_scores[mask].copy(),
            pred_boxes[mask].copy(),
            iou_thresh=iou_thresh,
            score_thresh=score_thresh,
        )
        if len(merged_boxes) == 0:
            continue
        out_boxes.append(merged_boxes)
        out_scores.append(merged_scores)
        out_labels.append(np.full(len(merged_scores), cls_id, dtype=pred_labels.dtype))

    if not out_boxes:
        return pred_boxes[:0], pred_scores[:0], pred_labels[:0]
    out_boxes   = np.concatenate(out_boxes)
    out_scores  = np.concatenate(out_scores)
    out_labels  = np.concatenate(out_labels)
    order = np.argsort(out_scores)[::-1]
    return out_boxes[order], out_scores[order], out_labels[order]


def postprocess(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                pred_labels: np.ndarray, class_names: List[str],
                method: str, iou_thresh: float):
    """Dispatch to the requested post-processing algorithm."""
    if len(pred_boxes) == 0 or method == 'none':
        return pred_boxes, pred_scores, pred_labels
    if method == 'nms':
        return apply_nms(pred_boxes, pred_scores, pred_labels, iou_thresh)
    if method == 'soft_nms':
        return apply_soft_nms(pred_boxes, pred_scores, pred_labels, iou_thresh)
    if method == 'wbf':
        return apply_wbf(pred_boxes, pred_scores, pred_labels, class_names, iou_thresh)
    raise ValueError(f'Unknown nms_method: {method!r}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='VirConv prediction & image visualisation')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='model config YAML (e.g. cfgs/models/kitti/VirConv-S.yaml)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='checkpoint path (.pth)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='KITTI split to run on (default: val)')
    parser.add_argument('--sample_idx', type=str, nargs='+', default=None,
                        help='frame IDs to visualize, e.g. 000008 000010')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='stop after this many samples')
    parser.add_argument('--score_thresh', type=float, default=0.5,
                        help='confidence threshold for display (default: 0.5)')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                        help='show only these classes, e.g. --class_names Car Pedestrian '
                             '(default: all classes from the config)')
    parser.add_argument('--output_dir', type=str, default='../output/predictions',
                        help='where to save annotated images')
    parser.add_argument('--nms_method', type=str, default='nms',
                        choices=['none', 'nms', 'soft_nms', 'wbf'],
                        help='post-processing to select best box per object '
                             '(none/nms/soft_nms/wbf, default: nms)')
    parser.add_argument('--nms_iou_thresh', type=float, default=0.5,
                        help='IoU threshold used by nms / soft_nms / wbf (default: 0.5)')
    parser.add_argument('--show_gt', action='store_true',
                        help='overlay ground-truth boxes (blue) when available')
    parser.add_argument('--show_lidar', action='store_true',
                        help='overlay projected LiDAR points coloured by depth')
    parser.add_argument('--set', dest='set_cfgs', default=None,
                        nargs=argparse.REMAINDER,
                        help='override config keys, e.g. --set MODEL.POST_PROCESSING.SCORE_THRESH 0.3')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs, cfg)

    # Point the dataset at the requested split
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = args.split

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = common_utils.create_logger(output_dir / 'prediction.log')
    # Resolve which classes to display
    all_classes = cfg.CLASS_NAMES                         # from config
    show_classes = args.class_names if args.class_names else all_classes
    unknown = [c for c in show_classes if c not in all_classes]
    if unknown:
        raise ValueError(f'--class_names {unknown} not in model classes {all_classes}')
    # set of 1-based label indices to display
    display_label_ids = {i + 1 for i, name in enumerate(all_classes) if name in show_classes}
    color_map = build_color_map(show_classes)

    logger.info(f'Config  : {args.cfg_file}')
    logger.info(f'Ckpt    : {args.ckpt}')
    logger.info(f'Split   : {args.split}')
    logger.info(f'Classes : {show_classes}')
    logger.info(f'Thresh  : {args.score_thresh}')
    logger.info(f'Method  : {args.nms_method}  IoU={args.nms_iou_thresh}')
    logger.info(f'Output  : {output_dir}')

    # -----------------------------------------------------------------------
    # Dataset & model
    # -----------------------------------------------------------------------
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )

    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # When specific frame IDs are requested, build a direct index map so we
    # don't have to scan the entire dataset to find them.
    if args.sample_idx:
        frame_to_idx = {
            info['point_cloud']['lidar_idx']: i
            for i, info in enumerate(dataset.kitti_infos)
        }
        missing = [fid for fid in args.sample_idx if fid not in frame_to_idx]
        if missing:
            logger.warning(f'Frame IDs not found in split: {missing}')
        subset_indices = [frame_to_idx[fid] for fid in args.sample_idx if fid in frame_to_idx]
        from torch.utils.data import DataLoader, Subset
        loader = DataLoader(
            Subset(dataset, subset_indices),
            batch_size=1,
            num_workers=0,
            collate_fn=dataset.collate_batch,
        )
        logger.info(f'Processing {len(subset_indices)} specific frame(s): {args.sample_idx}')

    count = 0

    with torch.no_grad():
        for batch_dict in loader:
            if args.max_samples and count >= args.max_samples:
                break

            frame_id = batch_dict['frame_id'][0]
            load_data_to_gpu(batch_dict)
            pred_dicts, _, batch_dict = model(batch_dict)

            pred_boxes  = pred_dicts[0]['pred_boxes'].cpu().numpy()   # (N, 7)
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()  # (N,)
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()  # (N,)

            # ------------------------------------------------------------------
            # Filter to requested classes only
            # ------------------------------------------------------------------
            cls_mask = np.isin(pred_labels, list(display_label_ids))
            pred_boxes  = pred_boxes[cls_mask]
            pred_scores = pred_scores[cls_mask]
            pred_labels = pred_labels[cls_mask]

            # ------------------------------------------------------------------
            # Post-processing: select best box per object
            # ------------------------------------------------------------------
            n_raw = len(pred_boxes)
            pred_boxes, pred_scores, pred_labels = postprocess(
                pred_boxes, pred_scores, pred_labels,
                all_classes, args.nms_method, args.nms_iou_thresh,
            )
            calib = batch_dict['calib'][0]

            # ------------------------------------------------------------------
            # Load raw camera image from disk
            # ------------------------------------------------------------------
            img_path = dataset.root_split_path / 'image_2' / f'{frame_id}.png'
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                logger.warning(f'Image not found: {img_path}')
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # ------------------------------------------------------------------
            # Optional: raw LiDAR points for depth overlay
            # ------------------------------------------------------------------
            lidar_pts = None
            if args.show_lidar:
                lidar_path = dataset.root_split_path / 'velodyne' / f'{frame_id}.bin'
                if lidar_path.exists():
                    lidar_pts = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

            # ------------------------------------------------------------------
            # Optional: ground-truth boxes (val/train split only)
            # ------------------------------------------------------------------
            gt_boxes = None
            if args.show_gt and 'gt_boxes' in batch_dict:
                raw_gt = batch_dict['gt_boxes'][0]   # (max_gt, 8)
                if isinstance(raw_gt, torch.Tensor):
                    raw_gt = raw_gt.cpu().numpy()
                valid = raw_gt[:, -1] > 0            # last col = class index; 0 = padding
                gt_boxes = raw_gt[valid, :7]

            # ------------------------------------------------------------------
            # Visualize
            # ------------------------------------------------------------------
            vis = visualize_predictions(
                image_rgb=image_rgb,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                calib=calib,
                class_names=all_classes,
                color_map=color_map,
                score_thresh=args.score_thresh,
                gt_boxes=gt_boxes,
                lidar_points=lidar_pts,
            )

            n_det = int((pred_scores >= args.score_thresh).sum())
            info_str = f'Frame {frame_id}  |  {n_det} det(s) >= {args.score_thresh}'
            cv2.putText(vis, info_str, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, info_str, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            out_path = output_dir / f'{frame_id}.png'
            cv2.imwrite(str(out_path), vis)
            logger.info(
                f'[{count + 1:04d}] {out_path}  —  '
                f'{n_det} det(s) >= {args.score_thresh}  '
                f'[{args.nms_method}: {n_raw} → {len(pred_boxes)} boxes]'
            )

            count += 1

    logger.info(f'Finished. {count} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
