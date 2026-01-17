import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple
import faiss

import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from models.encoders import (
    CLIPEncoder, FashionCLIPEncoder, 
    GroundingDINODetector, SAMSegmentor,
    MultiModalFusionNetwork, create_projection_head
)
from utils.image_utils import (
    load_image, extract_hsv_histogram, 
    get_dominant_color, crop_and_resize,
    l2_normalize
)


class MultiVectorIndexer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.get_device()
        
        print(f"Initializing Indexer on device: {self.device}")
        
        # Initialize encoders
        print("Loading CLIP encoder...")
        self.clip_encoder = CLIPEncoder(
            model_name=self.config.CLIP_MODEL,
            device=self.device
        )
        
        print("Loading FashionCLIP encoder...")
        self.fashion_clip_encoder = FashionCLIPEncoder(device=self.device)
        
        print("Loading Grounding DINO detector...")
        self.detector = GroundingDINODetector(
            config_path=self.config.GROUNDING_DINO_CONFIG,
            checkpoint_path=self.config.GROUNDING_DINO_CHECKPOINT,
            device=self.device
        )
        
        print("Loading SAM segmentor...")
        self.segmentor = SAMSegmentor(
            checkpoint_path=self.config.SAM_CHECKPOINT,
            model_type=self.config.SAM_MODEL_TYPE,
            device=self.device
        )
        
        # Projection heads to unified 2048-D space
        self.projection_global = create_projection_head(
            self.config.CLIP_DIM, 
            self.config.EMBED_DIM
        ).to(self.device)
        
        self.projection_fashion = create_projection_head(
            self.config.FASHION_CLIP_DIM,
            self.config.EMBED_DIM
        ).to(self.device)
        
        self.projection_item = create_projection_head(
            self.config.CLIP_DIM,
            self.config.EMBED_DIM
        ).to(self.device)
        
        self.projection_scene = create_projection_head(
            self.config.CLIP_DIM,
            self.config.EMBED_DIM
        ).to(self.device)
        
        # Storage for embeddings and metadata
        self.image_ids = []
        self.image_paths = []
        self.embeddings_global = []
        self.embeddings_person = []
        self.embeddings_items = {}  # {image_id: [item_embeddings]}
        self.embeddings_scene = []
        self.metadata = {}  # {image_id: detailed_metadata}
        
    def extract_features_single_image(self, image_path: str, image_id: str) -> Dict:
        try:
            # Load image
            image = load_image(image_path)
            image_np = np.array(image)

            def _to_numpy(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                return np.array(x)
            
            def _box_to_list(b):
                if b is None:
                    return None
                if isinstance(b, torch.Tensor):
                    return b.detach().cpu().tolist()
                try:
                    return list(b)
                except Exception:
                    return b

            # 1. Prepare batched crops: global, optional person, items
            images_to_encode = []
            images_roles = []

            # global
            images_to_encode.append(image)
            images_roles.append(('global', None))

            person_box = None
            person_idx_in_batch = None

            # 2. Detections
            detections = self.detector.detect(
                image_path=image_path,
                text_prompt=self.config.GROUNDING_TEXT_PROMPT,
                box_threshold=self.config.ITEM_CONFIDENCE_THRESHOLD
            )

            # 3. SAM masks (fallback does coarse masks)
            self.segmentor.set_image(image_np)
            boxes_list = []
            if 'boxes' in detections and detections['boxes'] is not None:
                boxes_arr = detections['boxes']
                # safe conversion: if torch tensor -> detach + cpu + list
                if isinstance(boxes_arr, torch.Tensor):
                    boxes_list = boxes_arr.detach().cpu().tolist()
                else:
                    try:
                        boxes_list = boxes_arr.tolist()
                    except Exception:
                        # fallback iterate
                        boxes_list = [list(b) for b in boxes_arr]
            else:
                boxes_list = []

            masks = self.segmentor.segment_from_boxes(boxes_list) if len(boxes_list) > 0 else []

            # 4. Person Crop
            person_boxes = [
                box for box, label in zip(detections.get('boxes', []), detections.get('labels', []))
                if 'person' in str(label).lower()
            ]
            if len(person_boxes) > 0:
                # choose largest person
                person_box = max(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                person_crop_np = crop_and_resize(image_np, person_box, (224, 224))
                person_image = Image.fromarray(person_crop_np)
                person_idx_in_batch = len(images_to_encode)
                images_to_encode.append(person_image)
                images_roles.append(('person', None))

            # 5. Item crops
            item_crops = []
            for idx_box, (box, label, score) in enumerate(zip(
                detections.get('boxes', []),
                detections.get('labels', []),
                detections.get('scores', [])
            )):
                if float(score) < self.config.ITEM_CONFIDENCE_THRESHOLD:
                    continue
                crop_np = crop_and_resize(image_np, box, (224, 224))
                item_image = Image.fromarray(crop_np)
                item_idx_in_batch = len(images_to_encode)
                images_to_encode.append(item_image)
                images_roles.append(('item', idx_box))
                mask_for_box = masks[idx_box] if (idx_box < len(masks)) else None
                item_crops.append({
                    'box': box,
                    'label': label,
                    'score': float(score),
                    'mask': mask_for_box,
                    'batch_pos': item_idx_in_batch
                })

            # 6. Single batched call to CLIP for all crops (fast)
            with torch.no_grad():
                clip_feats_batch = self.clip_encoder.encode_images_batch(images_to_encode)  # (N, D) tensor on encoder.device

            # detach and move to CPU for safe numpy/list conversions
            if isinstance(clip_feats_batch, torch.Tensor):
                clip_feats_batch = clip_feats_batch.detach().cpu()

            # 7. Map features back
            clip_global = clip_feats_batch[0]  # 1D tensor (on CPU)

            # Person embedding (optional)
            person_embedding = None
            if person_idx_in_batch is not None:
                person_emb = clip_feats_batch[person_idx_in_batch]
                person_embedding = self.projection_global(person_emb.unsqueeze(0).to(self.device)).squeeze(0)
                person_embedding = l2_normalize(person_embedding)

            # Items: embeddings + metadata
            item_embeddings = []
            item_metadata = []
            for it_idx, it in enumerate(item_crops):
                emb = clip_feats_batch[it['batch_pos']]
                emb_proj = self.projection_item(emb.unsqueeze(0).to(self.device)).squeeze(0)
                emb_proj = l2_normalize(emb_proj)
                item_embeddings.append(emb_proj)

                # color & histogram using the mask (mask may be numpy/torch/None)
                dominant_color = get_dominant_color(image_np, it['mask'])
                color_hist = extract_hsv_histogram(image_np, it['mask'])

                item_metadata.append({
                    'label': it['label'],
                    'bbox': [float(x) for x in _box_to_list(it['box'])],
                    'score': float(it['score']),
                    'color': dominant_color,
                    'color_hist': color_hist.tolist() if isinstance(color_hist, np.ndarray) else color_hist,
                    'embedding_idx': len(item_embeddings)-1
                })

            # 8. FashionCLIP global (may be fallback random if FashionCLIP unavailable)
            fashion_global = self.fashion_clip_encoder.encode_image(image)

            # If FashionCLIP produced numpy, convert to torch; else detach and move to device
            if isinstance(fashion_global, np.ndarray):
                fashion_global = torch.from_numpy(fashion_global).to(self.device)
            elif isinstance(fashion_global, torch.Tensor):
                fashion_global = fashion_global.detach().to(self.device)
            else:
                fashion_global = torch.tensor(fashion_global).to(self.device)

            # Project CLIP global (clip_global is on CPU) and FashionCLIP, then fuse
            clip_global_proj = self.projection_global(clip_global.unsqueeze(0).to(self.device)).squeeze(0)
            fashion_global_proj = self.projection_fashion(fashion_global.unsqueeze(0)).squeeze(0)
            global_embedding = (clip_global_proj + fashion_global_proj) / 2.0
            global_embedding = l2_normalize(global_embedding)

            # If person embedding missing, fallback to global embedding (previous behavior)
            if person_embedding is None:
                person_embedding = global_embedding

            # 9. Scene embedding: probe with scene texts using CLIP text encoder
            with torch.no_grad():
                scene_texts = [f"a photo taken in a {cat}" for cat in self.config.SCENE_CATEGORIES]
                scene_text_embs = self.clip_encoder.encode_texts_batch(scene_texts)
                if isinstance(scene_text_embs, np.ndarray):
                    scene_text_embs = torch.from_numpy(scene_text_embs)
                if isinstance(scene_text_embs, torch.Tensor):
                    scene_text_embs = scene_text_embs.detach().to(clip_global.device)
                else:
                    scene_text_embs = torch.tensor(scene_text_embs, device=clip_global.device)

                # Compute similarity with global image
                scene_sims = torch.matmul(
                    l2_normalize(clip_global.unsqueeze(0).to(clip_global.device)),
                    l2_normalize(scene_text_embs).t()
                ).squeeze(0)

                scene_weights = torch.softmax(scene_sims * 5, dim=0)
                scene_embedding = torch.sum(scene_text_embs * scene_weights.unsqueeze(1), dim=0)
                scene_embedding = self.projection_scene(scene_embedding.unsqueeze(0).to(self.device)).squeeze(0)
                scene_embedding = l2_normalize(scene_embedding)

            dominant_scene_idx = int(torch.argmax(scene_sims).item())
            dominant_scene = self.config.SCENE_CATEGORIES[dominant_scene_idx]

            # 10. Style classification using FashionCLIP text probes
            with torch.no_grad():
                style_texts = [f"{style} fashion style" for style in self.config.STYLE_CATEGORIES]
                style_text_embs = self.fashion_clip_encoder.encode_texts_batch(style_texts)

                if isinstance(style_text_embs, np.ndarray):
                    style_text_embs = torch.from_numpy(style_text_embs)
                if isinstance(style_text_embs, torch.Tensor):
                    style_text_embs = style_text_embs.detach().to(self.device)
                else:
                    style_text_embs = torch.tensor(style_text_embs, device=self.device)

                style_sims = torch.matmul(
                    l2_normalize(fashion_global.unsqueeze(0).to(self.device)),
                    l2_normalize(style_text_embs).t()
                ).squeeze(0)

                style_probs = torch.softmax(style_sims * 3, dim=0).detach().cpu().numpy()

            style_scores = {
                style: float(prob)
                for style, prob in zip(self.config.STYLE_CATEGORIES, style_probs)
            }

            # Package features (convert tensors to numpy for storage) — use detach() before numpy
            features = {
                'image_id': image_id,
                'image_path': image_path,
                'embeddings': {
                    'global': global_embedding.detach().cpu().numpy(),
                    'person': person_embedding.detach().cpu().numpy(),
                    'items': [emb.detach().cpu().numpy() for emb in item_embeddings],
                    'scene': scene_embedding.detach().cpu().numpy()
                },
                'metadata': {
                    'num_detections': len(detections.get('boxes', [])),
                    'person_box': _box_to_list(person_box),
                    'items': item_metadata,
                    'scene': dominant_scene,
                    'scene_scores': {cat: float(score) for cat, score in zip(self.config.SCENE_CATEGORIES, scene_sims.detach().cpu().numpy())},
                    'style_scores': style_scores
                }
            }

            return features

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def index_dataset(self, image_dir: str, batch_size: int = 1):
        print(f"\nIndexing images from: {image_dir}")
        
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(Path(image_dir).glob(ext))
        
        unique_paths = list(set([p.resolve() for p in image_paths]))
        image_paths = [Path(p) for p in unique_paths]
        
        print(f"Found {len(image_paths)} images")
        
        # Process each image
        for idx, image_path in enumerate(tqdm(image_paths, desc="Extracting features")):
            image_id = f"img_{idx:06d}"
            
            features = self.extract_features_single_image(str(image_path), image_id)
            
            if features is None:
                continue
            
            # Store data
            self.image_ids.append(image_id)
            self.image_paths.append(str(image_path))
            self.embeddings_global.append(features['embeddings']['global'])
            self.embeddings_person.append(features['embeddings']['person'])
            self.embeddings_scene.append(features['embeddings']['scene'])
            self.embeddings_items[image_id] = features['embeddings']['items']
            self.metadata[image_id] = features['metadata']
        
        print(f"\nSuccessfully indexed {len(self.image_ids)} images")
        
    def build_faiss_indices(self):
        print("\nBuilding FAISS indices...")
        
        d = self.config.EMBED_DIM  # Dimension
        
        # Convert to numpy arrays
        global_embs = np.array(self.embeddings_global).astype('float32')
        person_embs = np.array(self.embeddings_person).astype('float32')
        scene_embs = np.array(self.embeddings_scene).astype('float32')
        
        item_embs_list = []
        item_to_image_map = []  # Track which image each item belongs to
        
        for image_id in self.image_ids:
            items = self.embeddings_items.get(image_id, [])
            for item_emb in items:
                item_embs_list.append(item_emb)
                item_to_image_map.append(image_id)
        
        if len(item_embs_list) > 0:
            item_embs = np.array(item_embs_list).astype('float32')
        else:
            item_embs = np.zeros((0, d), dtype='float32')
        
        # Create indices based on dataset size
        n_images = len(self.image_ids)
        
        if n_images < 10000:
            # Small dataset: use HNSW for best accuracy
            print(f"Creating HNSW indices for {n_images} images")
            
            # Global index
            global_index = faiss.IndexHNSWFlat(d, self.config.HNSW_M)
            global_index.hnsw.efConstruction = self.config.HNSW_EF_CONSTRUCTION
            global_index.add(global_embs)
            
            # Person index
            person_index = faiss.IndexHNSWFlat(d, self.config.HNSW_M)
            person_index.hnsw.efConstruction = self.config.HNSW_EF_CONSTRUCTION
            person_index.add(person_embs)
            
            # Scene index
            scene_index = faiss.IndexHNSWFlat(d, self.config.HNSW_M)
            scene_index.hnsw.efConstruction = self.config.HNSW_EF_CONSTRUCTION
            scene_index.add(scene_embs)
            
            # Item index
            if item_embs.shape[0] > 0:
                item_index = faiss.IndexHNSWFlat(d, self.config.HNSW_M)
                item_index.hnsw.efConstruction = self.config.HNSW_EF_CONSTRUCTION
                item_index.add(item_embs)
            else:
                item_index = faiss.IndexFlatL2(d)
        
        else:
            # Large dataset: IVF + PQ for compression
            print(f"Creating IVF-PQ indices for {n_images} images")
            
            nlist = min(4096, n_images // 100)  # Number of clusters
            
            quantizer = faiss.IndexFlatL2(d)
            global_index = faiss.IndexIVFPQ(quantizer, d, nlist, 64, 8)
            global_index.train(global_embs)
            global_index.add(global_embs)
            
            # Similar for others...
            person_index = faiss.IndexIVFPQ(quantizer, d, nlist, 64, 8)
            person_index.train(person_embs)
            person_index.add(person_embs)
            
            scene_index = faiss.IndexIVFPQ(quantizer, d, nlist, 64, 8)
            scene_index.train(scene_embs)
            scene_index.add(scene_embs)
            
            if item_embs.shape[0] > 0:
                item_index = faiss.IndexIVFPQ(quantizer, d, nlist, 64, 8)
                item_index.train(item_embs)
                item_index.add(item_embs)
            else:
                item_index = faiss.IndexFlatL2(d)
        
        # Store indices
        self.indices = {
            'global': global_index,
            'person': person_index,
            'scene': scene_index,
            'item': item_index
        }
        
        self.item_to_image_map = item_to_image_map
        
        print("-> FAISS indices built successfully")
        
    def save(self, output_dir: str = None):
        output_dir = output_dir or str(self.config.INDEX_DIR)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving indices to: {output_dir}")
        
        # Save FAISS indices
        faiss.write_index(self.indices['global'], f"{output_dir}/global_index.faiss")
        faiss.write_index(self.indices['person'], f"{output_dir}/person_index.faiss")
        faiss.write_index(self.indices['scene'], f"{output_dir}/scene_index.faiss")
        faiss.write_index(self.indices['item'], f"{output_dir}/item_index.faiss")
        
        # Save metadata
        metadata_bundle = {
            'image_ids': self.image_ids,
            'image_paths': self.image_paths,
            'metadata': self.metadata,
            'item_to_image_map': self.item_to_image_map,
            'config': {
                'embed_dim': self.config.EMBED_DIM,
                'num_images': len(self.image_ids)
            }
        }
        
        with open(f"{output_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata_bundle, f)
        
        summary = {
            'num_images': len(self.image_ids),
            'num_items_detected': len(self.item_to_image_map),
            'avg_items_per_image': len(self.item_to_image_map) / max(len(self.image_ids), 1),
            'image_paths_sample': self.image_paths[:5]
        }
        
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("-> All data saved successfully")
        print(f"  - Global index: {len(self.image_ids)} vectors")
        print(f"  - Item index: {len(self.item_to_image_map)} vectors")
        print(f"  - Metadata: {len(self.metadata)} images")


def main():
    print("\nIndexer (Part A)")
    
    # Load config
    config = Config()
    config.print_config()
    
    config.IMAGES_DIR = Path("D:/fashion_retrieval/data/images")  # Update this path
    
    # Create indexer
    indexer = MultiVectorIndexer(config)
    
    # Index dataset
    indexer.index_dataset(str(config.IMAGES_DIR))
    
    # Build FAISS indices
    indexer.build_faiss_indices()
    
    # Save everything
    indexer.save()
    
    print("Indexing complete!")
    print(f"Indexed {len(indexer.image_ids)} images")
    print(f"Saved to: {config.INDEX_DIR}")


if __name__ == "__main__":
    main()