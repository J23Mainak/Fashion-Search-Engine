import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Dict, Optional
import colorsys

def load_image(image_path: str, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    img = Image.open(image_path).convert('RGB')
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img

def _to_numpy_image(image) -> np.ndarray:
    # If torch tensor, move to cpu and convert
    if isinstance(image, torch.Tensor):
        # Expect CHW or HWC; try to convert to HWC uint8
        arr = image.detach().cpu().numpy()
        # If CHW, transpose to HWC
        if arr.ndim == 3 and arr.shape[0] in (1,3):
            arr = np.transpose(arr, (1, 2, 0))
        # If values appear in [0,1], scale to [0,255]
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            if arr.max() <= 1.01:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return arr
    # If PIL Image
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        return arr
    # If already numpy
    if isinstance(image, np.ndarray):
        # If grayscale expand dims
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        # Ensure dtype uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.01:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image
    # Fallback: try to convert
    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_numpy_mask(mask, target_shape: Optional[Tuple[int,int]] = None) -> np.ndarray:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.array(mask)
    # If mask has channel dim, squeeze
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[:,:,0]
    # Resize if target shape provided and mismatch
    if target_shape is not None and (m.shape[0] != target_shape[0] or m.shape[1] != target_shape[1]):
        m = cv2.resize(m.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    # Normalize to 0/255 uint8
    if m.dtype != np.uint8:
        # If boolean
        if m.dtype == bool:
            m = (m.astype(np.uint8) * 255)
        else:
            # scale if in [0,1]
            if m.max() <= 1.01:
                m = (m * 255.0).astype(np.uint8)
            else:
                m = m.astype(np.uint8)
    return m


def extract_hsv_histogram(image: np.ndarray, mask: Optional[np.ndarray] = None, bins: int = 32) -> np.ndarray:
    # Ensure image is numpy HxWx3 uint8
    img = _to_numpy_image(image)
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Prepare mask if given
    if mask is not None:
        m = _to_numpy_mask(mask, target_shape=(hsv.shape[0], hsv.shape[1]))
    else:
        m = None

    # Define histogram parameters
    hist_bins = [bins, bins, bins]
    hist_range = [0, 180, 0, 256, 0, 256]  # HSV ranges

    # Calculate histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], m, hist_bins, hist_range)
    # Normalize to sum 1 (L1) — stable even if empty
    hist_sum = hist.sum()
    if hist_sum == 0:
        # Return zero histogram if nothing to compute (avoid divide by zero)
        hist = np.zeros(hist.flatten().shape, dtype=np.float32)
    else:
        hist = (hist / (hist_sum + 1e-9)).astype(np.float32)

    return hist.flatten()


def get_dominant_color(image: np.ndarray, mask: Optional[np.ndarray] = None, k: int = 3) -> str:
    img = _to_numpy_image(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if mask is not None:
        m = _to_numpy_mask(mask, target_shape=(hsv.shape[0], hsv.shape[1]))
        # get masked pixels
        pixels = hsv[m > 0]
    else:
        pixels = hsv.reshape(-1, 3)

    if pixels is None or len(pixels) == 0:
        return "unknown"

    # Use median HSV with robust checks
    median_hsv = np.median(pixels, axis=0)
    h, s, v = median_hsv.astype(float)

    # Heuristic thresholds
    if v < 50:
        return "black"
    elif s < 30:
        if v > 200:
            return "white"
        else:
            return "gray"
    elif h < 10 or h > 170:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 125:
        return "blue"
    elif h < 155:
        return "purple"
    else:
        return "pink"


def crop_and_resize(image: np.ndarray, bbox: List[float], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = _to_numpy_image(image)
    h, w = img.shape[0], img.shape[1]

    # Ensure bbox numbers are ints and clipped
    x1, y1, x2, y2 = [int(float(coord)) for coord in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Handle degenerate boxes
    if x2 <= x1 or y2 <= y1:
        # Return black image
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    cropped = img[y1:y2, x1:x2]

    if cropped is None or cropped.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    resized = cv2.resize(cropped, (target_size[0], target_size[1]), interpolation=cv2.INTER_LANCZOS4)
    # Ensure uint8
    if resized.dtype != np.uint8:
        if resized.max() <= 1.01:
            resized = (resized * 255.0).astype(np.uint8)
        else:
            resized = resized.astype(np.uint8)
    return resized


def normalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height
    ]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def l2_normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(tensor, p=2, dim=dim)


def weighted_fusion(embeddings: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    assert len(embeddings) == len(weights), "Number of embeddings must match weights"
    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1.0, got {sum(weights)}"
    
    # Stack and weight
    stacked = torch.stack(embeddings, dim=0)
    weights_tensor = torch.tensor(weights, device=stacked.device).view(-1, 1)
    
    fused = (stacked * weights_tensor).sum(dim=0)
    
    # Normalize
    return l2_normalize(fused)


def cosine_similarity_matrix(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    # Normalize
    embeddings1 = l2_normalize(embeddings1)
    embeddings2 = l2_normalize(embeddings2)
    
    # Compute similarity
    return torch.mm(embeddings1, embeddings2.t())


def batch_cosine_similarity(query_embedding: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
    query_embedding = l2_normalize(query_embedding.unsqueeze(0))
    candidate_embeddings = l2_normalize(candidate_embeddings)
    
    similarities = torch.mm(query_embedding, candidate_embeddings.t()).squeeze(0)
    return similarities


def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5, sigma: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    N = boxes.shape[0]
    
    for i in range(N):
        max_idx = i + np.argmax(scores[i:])
        
        # Swap
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        
        # Suppress overlapping boxes
        for j in range(i + 1, N):
            iou = compute_iou(boxes[i].tolist(), boxes[j].tolist())
            
            # Gaussian suppression
            weight = np.exp(-(iou * iou) / sigma)
            scores[j] *= weight
    
    # Filter by threshold
    keep = scores > 0.01
    return boxes[keep], scores[keep]


def create_circular_mask(h: int, w: int, center: Tuple[int, int], radius: int) -> np.ndarray:
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8) * 255


def visualize_detections(image: np.ndarray, boxes: List[List[float]], labels: List[str], scores: List[float]) -> np.ndarray:
    vis_image = image.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        text = f"{label}: {score:.2f}"
        cv2.putText(vis_image, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_image


def parse_color_from_text(text: str) -> Optional[str]:
    text_lower = text.lower()
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", 
              "white", "black", "gray", "brown", "beige", "navy"]
    
    for color in colors:
        if color in text_lower:
            return color
    
    return None


def compute_embedding_similarity_batch(query_emb: torch.Tensor, database_embs: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    N = database_embs.shape[0]
    similarities = torch.zeros(N, device=query_emb.device)
    
    query_emb = l2_normalize(query_emb.unsqueeze(0))
    
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch_embs = l2_normalize(database_embs[i:end_idx])
        batch_sims = torch.mm(query_emb, batch_embs.t()).squeeze(0)
        similarities[i:end_idx] = batch_sims
    
    return similarities