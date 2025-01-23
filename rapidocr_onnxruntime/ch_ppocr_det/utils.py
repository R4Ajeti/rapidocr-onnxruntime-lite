# -*- encoding: utf-8 -*-
# @Author: MTF / rinor4ever
# @Contact: r4ajeti@gmail.com
from typing import List, Optional, Tuple

import pyclipper
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from skimage.measure import find_contours
from numpy.lib.stride_tricks import sliding_window_view


def binaryDilation(inputArray, structure=np.ones((3, 3), dtype=bool)):
    padWidth = (structure.shape[0] // 2, structure.shape[1] // 2)
    paddedInput = np.pad(inputArray.astype(bool), padWidth, mode='constant', constant_values=0)
    outputArray = np.zeros_like(inputArray, dtype=bool)
    
    for i in range(structure.shape[0]):
        for j in range(structure.shape[1]):
            if structure[i, j]:
                outputArray |= paddedInput[i:i+inputArray.shape[0], j:j+inputArray.shape[1]]
    
    return outputArray.astype(inputArray.dtype)

def binaryDilationFast(inputArray, structure=np.ones((3, 3))):
    """
    Optimized binary dilation using vectorized sliding window operations
    Works for 2D arrays and arbitrary structuring elements
    """
    inputArray = inputArray.astype(bool)
    structure = structure.astype(bool)
    
    padH, padW = [(dim // 2, dim // 2) for dim in structure.shape]
    padded = np.pad(inputArray, (padH, padW), mode='constant')
    
    windows = sliding_window_view(padded, structure.shape)
    dilated = np.any(windows & structure, axis=(-2, -1))
    
    return dilated.astype(inputArray.dtype)

class DetPreProcess:
    def __init__(
        self, limit_side_len: int = 736, limit_type: str = "min", mean=None, std=None
    ):
        if mean is None:
            mean = [0.5, 0.5, 0.5]

        if std is None:
            std = [0.5, 0.5, 0.5]

        self.mean = np.array(mean)
        self.std = np.array(std)
        self.scale = 1 / 255.0

        self.limit_side_len = limit_side_len
        self.limit_type = limit_type

    def __call__(self, img: np.ndarray) -> Optional[np.ndarray]:
        resized_img = self.resize(img)
        if resized_img is None:
            return None

        img = self.normalize(resized_img)
        img = self.permute(img)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def normalize(self, img: np.ndarray) -> np.ndarray:
        return (img.astype("float32") * self.scale - self.mean) / self.std

    def permute(self, img: np.ndarray) -> np.ndarray:
        return img.transpose((2, 0, 1))

    def resize(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Resize image to a size multiple of 32 using PIL"""
        h, w = img.shape[:2]
        ratio = 1.0

        if self.limit_type == "max":
            if max(h, w) > self.limit_side_len:
                ratio = self.limit_side_len / max(h, w)
        else:
            if min(h, w) < self.limit_side_len:
                ratio = self.limit_side_len / min(h, w)

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # Round to nearest multiple of 32
        resize_h = (resize_h + 16) // 32 * 32
        resize_w = (resize_w + 16) // 32 * 32

        if resize_w <= 0 or resize_h <= 0:
            return None

        try:
            mode = 'RGB' if img.ndim == 3 and img.shape[2] == 3 else 'L'
            pil_img = Image.fromarray(img.astype(np.uint8), mode=mode)
            resized_img = pil_img.resize((resize_w, resize_h), Image.BILINEAR)
            return np.array(resized_img)
        except Exception as e:
            raise RuntimeError(f"Error resizing image: {str(e)}") from e


class DBPostProcess:
    """The post process for Differentiable Binarization (DB)."""

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        unclip_ratio: float = 2.0,
        score_mode: str = "fast",
        use_dilation: bool = False,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.dilation_kernel = np.ones((2, 2), dtype=bool) if use_dilation else None

    def __call__(
        self, pred: np.ndarray, ori_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[float]]:
        src_h, src_w = ori_shape
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        mask = segmentation[0]
        if self.dilation_kernel is not None:
            mask = binaryDilation(mask.astype(np.uint8), self.dilation_kernel)
            
        boxes, scores = self.boxes_from_bitmap(pred[0], mask, src_w, src_h)
        return boxes, scores
    
    def min_area_rect(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        # Replace with custom convex hull implementation
        hull = self.convex_hull(points)
        min_rect = self.rotating_calipers(hull)
        return min_rect

    def boxes_from_bitmap(
        self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int
    ) -> Tuple[np.ndarray, List[float]]:
        height, width = bitmap.shape
        contours = find_contours(bitmap.astype(np.uint8), 0.5)

        boxes, scores = [], []
        for contour in contours[:self.max_candidates]:
            points = contour[:, ::-1]  # Convert to (x,y) format
            points = np.expand_dims(points.astype(np.float32), 1)
            
            box, sside = self.get_mini_boxes(points)
            if sside < self.min_size:
                continue

            score = self.box_score_fast(pred, box.reshape(-1, 2))
            if score < self.box_thresh:
                continue

            unclipped_box = self.unclip(box)
            unclipped_box, sside = self.get_mini_boxes(unclipped_box)
            if sside < self.min_size + 2:
                continue

            unclipped_box[:, 0] = np.clip(unclipped_box[:, 0] / width * dest_width, 0, dest_width)
            unclipped_box[:, 1] = np.clip(unclipped_box[:, 1] / height * dest_height, 0, dest_height)
            boxes.append(unclipped_box.astype(np.int32))
            scores.append(score)
        return np.array(boxes, dtype=np.int32), scores
    
    def get_mini_boxes(self, contour: np.ndarray) -> Tuple[np.ndarray, float]:
        # Handle contour shape properly
        points = contour.reshape(-1, 2)  # Convert to (N, 2) shape
        hull = self.convex_hull(points)
        
        # Rest of the code remains the same
        rect_points, (width, height) = self.rotating_calipers(hull)
        
        # Sort points by x-coordinate
        sorted_points = sorted(rect_points.tolist(), key=lambda x: x[0])
        
        # Original point ordering logic
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if sorted_points[1][1] > sorted_points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0

        if sorted_points[3][1] > sorted_points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array([
            sorted_points[index_1],
            sorted_points[index_2],
            sorted_points[index_3],
            sorted_points[index_4]
        ])
        return box, min(width, height)

    def convex_hull(self, points: np.ndarray) -> np.ndarray:
        """Andrew's monotone chain convex hull algorithm"""
        points = np.unique(points, axis=0)
        if len(points) <= 1:
            return points
        
        # Sort points lexicographically (first by x, then by y)
        sorted_points = points[np.lexsort((points[:, 1], points[:, 0]))]
        
        # Build lower hull
        lower = []
        for p in sorted_points:
            while len(lower) >= 2 and self.cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
            
        # Build upper hull
        upper = []
        for p in reversed(sorted_points):
            while len(upper) >= 2 and self.cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
            
        return np.array(lower[:-1] + upper[:-1])

    def rotating_calipers(self, points: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Minimum area rectangle using rotating calipers algorithm"""
        min_area = float('inf')
        best_rect = None
        best_dims = (0, 0)
        
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i+1) % len(points)]
            edge = p2 - p1
            edge_dir = edge / np.linalg.norm(edge)
            normal_dir = np.array([-edge_dir[1], edge_dir[0]])
            
            # Project all points onto edge and normal directions
            proj_edge = points @ edge_dir
            proj_norm = points @ normal_dir
            
            # Calculate rectangle dimensions
            min_e, max_e = np.min(proj_edge), np.max(proj_edge)
            min_n, max_n = np.min(proj_norm), np.max(proj_norm)
            width = max_e - min_e
            height = max_n - min_n
            area = width * height
            
            if area < min_area:
                min_area = area
                # Calculate rectangle corners
                best_rect = np.array([
                    [min_e, min_n],
                    [max_e, min_n],
                    [max_e, max_n],
                    [min_e, max_n]
                ]) @ np.array([edge_dir, normal_dir])
                best_dims = (width, height)
        
        # Order points consistently (top-left, top-right, bottom-right, bottom-left)
        mean = np.mean(best_rect, axis=0)
        angles = np.arctan2(best_rect[:,1]-mean[1], best_rect[:,0]-mean[0])
        ordered_indices = np.argsort(angles)
        ordered_rect = best_rect[ordered_indices]
        
        return ordered_rect, best_dims

    @staticmethod
    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """2D cross product of OA and OB vectors"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def box_score_fast(self, bitmap: np.ndarray, box: np.ndarray) -> float:
        h, w = bitmap.shape
        xmin = np.clip(np.floor(box[:,0].min()), 0, w-1).astype(int)
        xmax = np.clip(np.ceil(box[:,0].max()), 0, w-1).astype(int)
        ymin = np.clip(np.floor(box[:,1].min()), 0, h-1).astype(int)
        ymax = np.clip(np.ceil(box[:,1].max()), 0, h-1).astype(int)
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box_local = box - [xmin, ymin]  # Convert to local coordinates
        
        # Scanline fill algorithm
        for y in range(mask.shape[0]):
            intersections = []
            for i in range(4):
                p1 = box_local[i]
                p2 = box_local[(i+1) % 4]
                
                y1, y2 = sorted([p1[1], p2[1]])
                if not (y1 <= y <= y2):
                    continue
                    
                if p1[1] == p2[1]:
                    continue  # Horizontal edge
                    
                # Calculate intersection x coordinate
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                x = p1[0] + dx * (y - p1[1]) / dy
                intersections.append(x)
            
            # Sort and fill between pairs
            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i+1 >= len(intersections):
                    break
                start = int(np.clip(intersections[i], 0, mask.shape[1]-1))
                end = int(np.clip(intersections[i+1], 0, mask.shape[1]-1))
                mask[y, start:end+1] = 1

        return bitmap[ymin:ymax+1, xmin:xmax+1][mask.astype(bool)].mean()

    def unclip(self, box: np.ndarray) -> np.ndarray:
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance)).reshape(-1, 2)
        return expanded
