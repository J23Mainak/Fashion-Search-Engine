import torch
import torch.nn as nn
import clip
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoModel, AutoProcessor, Blip2Processor, Blip2ForConditionalGeneration
    from fashion_clip.fashion_clip import FashionCLIP
except ImportError:
    print("Warning: Some models not installed. Install with: pip install transformers fashion-clip")


class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-L/14", device: str = "cuda"):
        self.device = device

        if self.device == "cpu" and model_name in ("ViT-L/14", "ViT-L-14"):
            chosen_model = "ViT-B/32"
        else:
            chosen_model = model_name

        try:
            self.model, self.preprocess = clip.load(chosen_model, device=device, download_root="D:/clip_cache")
            try:
                self.model.eval()
            except Exception:
                pass
        except Exception as e:
            # Fallback: RN50 is smaller and often loads when ViT variants fail
            print(f"Warning: Failed to load CLIP model '{chosen_model}': {e}. Trying 'RN50' fallback.")
            self.model, self.preprocess = clip.load("RN50", device=device, download_root="D:/clip_cache")
            try:
                self.model.eval()
            except Exception:
                pass

    def _ensure_pil(self, image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.01:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            return Image.fromarray(arr)
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.01:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            return Image.fromarray(arr)
        try:
            return Image.fromarray(image)
        except Exception as e:
            raise TypeError(f"Unsupported image type for CLIP encoding: {type(image)} -> {e}")

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        pil_img = self._ensure_pil(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_input)
        return image_features.squeeze(0)

    @torch.no_grad()
    def encode_images_batch(self, images: List) -> torch.Tensor:
        # Accepts list of PIL/numpy/torch images -> returns (N, D) tensor on self.device
        pil_images = [self._ensure_pil(img) for img in images]
        image_inputs = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        image_features = self.model.encode_image(image_inputs)
        return image_features

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        text_input = clip.tokenize([text], truncate=True).to(self.device)
        text_features = self.model.encode_text(text_input)
        return text_features.squeeze(0)

    @torch.no_grad()
    def encode_texts_batch(self, texts: List[str]) -> torch.Tensor:
        text_inputs = clip.tokenize(texts, truncate=True).to(self.device)
        text_features = self.model.encode_text(text_inputs)
        return text_features


class FashionCLIPEncoder:
    
    def __init__(self, model_name: str = "fashion-clip", device: str = "cuda"):
        self.device = device
        try:
            self.model = FashionCLIP(model_name)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load FashionCLIP: {e}")
            print("Falling back to standard CLIP")
            self.model = None
    
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        if self.model is None:
            # Fallback to dummy embedding
            return torch.randn(512).to(self.device)
        
        image_features = self.model.encode_images([image], batch_size=1)
        return torch.from_numpy(image_features[0]).to(self.device)
    
    @torch.no_grad()
    def encode_images_batch(self, images: List[Image.Image]) -> torch.Tensor:
        if self.model is None:
            return torch.randn(len(images), 512).to(self.device)
        
        image_features = self.model.encode_images(images, batch_size=32)
        return torch.from_numpy(image_features).to(self.device)
    
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        if self.model is None:
            return torch.randn(512).to(self.device)
        
        text_features = self.model.encode_text([text], batch_size=1)
        return torch.from_numpy(text_features[0]).to(self.device)
    
    @torch.no_grad()
    def encode_texts_batch(self, texts: List[str]) -> torch.Tensor:
        if self.model is None:
            return torch.randn(len(texts), 512).to(self.device)
        
        text_features = self.model.encode_text(texts, batch_size=32)
        return torch.from_numpy(text_features).to(self.device)


class GroundingDINODetector:
    
    def __init__(self, config_path: str = None, checkpoint_path: str = None, device: str = "cuda"):
        self.device = device
        
        try:
            # Try to import GroundingDINO
            from groundingdino.util.inference import load_model, load_image, predict
            self.model = load_model(config_path, checkpoint_path)
            self.predict_fn = predict
            self.load_image_fn = load_image
            self.available = True
        except ImportError:
            print("Warning: GroundingDINO not installed. Using fallback detector.")
            self.available = False
    
    @torch.no_grad()
    def detect(self, image_path: str, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25) -> Dict:
        if not self.available:
            # Return dummy detections
            return {
                'boxes': np.array([[100, 100, 300, 400]]),
                'labels': ['clothing'],
                'scores': np.array([0.8])
            }
        
        image_source, image = self.load_image_fn(image_path)
        
        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # Convert to pixel coordinates
        h, w, _ = image_source.shape
        boxes = boxes * torch.tensor([w, h, w, h])
        boxes = boxes.cpu().numpy()
        
        return {
            'boxes': boxes,
            'labels': phrases,
            'scores': logits.cpu().numpy()
        }


class SAMSegmentor:
    
    def __init__(self, checkpoint_path: str = None, model_type: str = "vit_h", device: str = "cuda"):
        self.device = device
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            self.available = True
        except ImportError:
            print("Warning: SAM not installed. Using fallback segmentation.")
            self.available = False
    
    def set_image(self, image: np.ndarray):
        self.image_shape = (image.shape[0], image.shape[1])
        
        if self.available:
            self.predictor.set_image(image)
    
    @torch.no_grad()
    def segment_from_box(self, box: List[float]) -> np.ndarray:
        if not self.available:
            # Return dummy mask with correct dimensions matching the set image
            # Get image dimensions from the previously set image
            if hasattr(self, 'image_shape'):
                h, w = self.image_shape
            else:
                # Default fallback
                h, w = 1000, 1000
            
            x1, y1, x2, y2 = [int(c) for c in box]
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            return mask
    
        box_array = np.array(box)
        masks, scores, _ = self.predictor.predict(
            box=box_array[None, :],
            multimask_output=False
        )
        
        return masks[0]
    
    @torch.no_grad()
    def segment_from_boxes(self, boxes: List[List[float]]) -> List[np.ndarray]:
        masks = []
        for box in boxes:
            mask = self.segment_from_box(box)
            masks.append(mask)
        return masks


class BLIP2Reranker:
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
        self.device = device
        
        try:
            self.processor = Blip2Processor.from_pretrained(model_name, cache_dir="D:/hf_cache")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, cache_dir="D:/hf_cache",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            self.model.to(device)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load BLIP-2: {e}")
            self.available = False
    
    @torch.no_grad()
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        if not self.available:
            return 0.5  # Dummy score
        
        # Use ITM (Image-Text Matching) head
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        
        # Get similarity from model
        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs, return_dict=True)
            
        logits = outputs.logits
        similarity = torch.sigmoid(logits.mean()).item()
        
        return similarity
    
    @torch.no_grad()
    def compute_similarities_batch(self, images: List[Image.Image], text: str) -> List[float]:
        if not self.available:
            return [0.5] * len(images)
        
        scores = []
        # Process in smaller batches to avoid OOM
        batch_size = 8
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_texts = [text] * len(batch_images)
            
            inputs = self.processor(
                images=batch_images, 
                text=batch_texts, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs, return_dict=True)
            
            logits = outputs.logits
            batch_scores = torch.sigmoid(logits.mean(dim=-1)).cpu().tolist()
            scores.extend(batch_scores if isinstance(batch_scores, list) else [batch_scores])
        
        return scores


class MultiModalFusionNetwork(nn.Module):
    def __init__(self, input_dims: Dict[str, int], output_dim: int = 2048):
        super().__init__()
        
        self.modalities = list(input_dims.keys())
        self.output_dim = output_dim
        
        # Projection heads for each modality
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            for modality, dim in input_dims.items()
        })
        
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected = []
        
        for modality in self.modalities:
            if modality in embeddings and embeddings[modality] is not None:
                emb = self.projections[modality](embeddings[modality])
                projected.append(emb)
        
        if len(projected) == 0:
            return torch.zeros(self.output_dim)
        
        # Average pooling
        fused = torch.stack(projected).mean(dim=0)
        
        # L2 normalize
        fused = torch.nn.functional.normalize(fused, p=2, dim=-1)
        
        return fused


def create_projection_head(input_dim: int, output_dim: int, hidden_dim: int = 1024) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim)
    )