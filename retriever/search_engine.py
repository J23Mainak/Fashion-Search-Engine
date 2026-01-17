import os
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from models.encoders import (
    CLIPEncoder, FashionCLIPEncoder,
    BLIP2Reranker, create_projection_head
)
from utils.image_utils import (
    load_image, l2_normalize, 
    compute_iou, batch_cosine_similarity
)
from retriever.query_parser import QueryParser, ParsedQuery


class MultiVectorRetriever:
    def __init__(self, index_dir: str = None, config: Config = None):
        self.config = config or Config()
        self.device = self.config.get_device()
        
        index_dir = index_dir or str(self.config.INDEX_DIR)
        
        print(f"Loading retrieval system from: {index_dir}")
        
        # Load FAISS indices
        print("Loading FAISS indices...")
        self.indices = {
            'global': faiss.read_index(f"{index_dir}/global_index.faiss"),
            'person': faiss.read_index(f"{index_dir}/person_index.faiss"),
            'scene': faiss.read_index(f"{index_dir}/scene_index.faiss"),
            'item': faiss.read_index(f"{index_dir}/item_index.faiss")
        }
        
        # Set search parameters
        for index_name, index in self.indices.items():
            if hasattr(index, 'hnsw'):
                index.hnsw.efSearch = self.config.HNSW_EF_SEARCH
        
        # Load metadata
        print("Loading metadata...")
        with open(f"{index_dir}/metadata.pkl", 'rb') as f:
            metadata_bundle = pickle.load(f)
        
        self.image_ids = metadata_bundle['image_ids']
        self.image_paths = metadata_bundle['image_paths']
        self.metadata = metadata_bundle['metadata']
        self.item_to_image_map = metadata_bundle['item_to_image_map']
        
        print(f"-> Loaded {len(self.image_ids)} images")
        
        # Initialize encoders
        print("Loading encoders...")
        self.clip_encoder = CLIPEncoder(
            model_name=self.config.CLIP_MODEL,
            device=self.device
        )
        
        self.fashion_clip_encoder = FashionCLIPEncoder(device=self.device)
        
        # Load projection heads (same as indexer)
        print("Loading projection heads...")
        self.projection_global = create_projection_head(
            self.config.CLIP_DIM,
            self.config.EMBED_DIM
        ).to(self.device)
        
        self.projection_fashion = create_projection_head(
            self.config.FASHION_CLIP_DIM,
            self.config.EMBED_DIM
        ).to(self.device)
        
        self.projection_scene = create_projection_head(
            self.config.CLIP_DIM,
            self.config.EMBED_DIM
        ).to(self.device)
        
        # Initialize reranker
        print("Loading BLIP-2 reranker...")
        self.reranker = BLIP2Reranker(
            model_name=self.config.BLIP2_MODEL,
            device=self.device
        )
        
        # Initialize query parser
        self.query_parser = QueryParser()
        
        print("-> Retriever ready!")
    
    def encode_query(self, query_text: str, parsed_query: Optional[ParsedQuery] = None) -> Dict[str, torch.Tensor]:
        if parsed_query is None:
            parsed_query = self.query_parser.parse(query_text)
        
        embeddings = {}
        
        with torch.no_grad():
            # Global query embedding
            clip_global = self.clip_encoder.encode_text(query_text)
            fashion_global = self.fashion_clip_encoder.encode_text(query_text)
            
            clip_global_proj = self.projection_global(clip_global.unsqueeze(0)).squeeze(0)
            fashion_global_proj = self.projection_fashion(fashion_global.unsqueeze(0)).squeeze(0)
            
            global_emb = (clip_global_proj + fashion_global_proj) / 2
            embeddings['global'] = l2_normalize(global_emb)
            
            # Person embedding (same as global for query)
            embeddings['person'] = embeddings['global']
            
            # Item-specific embeddings
            item_embeddings = []
            for item in parsed_query.items:
                # Create item-specific query
                item_parts = []
                if item['color_modifier']:
                    item_parts.append(item['color_modifier'])
                if item['color']:
                    item_parts.append(item['color'])
                if item['type']:
                    item_parts.append(item['type'])
                
                if item_parts:
                    item_query = ' '.join(item_parts)
                    item_emb = self.clip_encoder.encode_text(item_query)
                    item_emb_proj = self.projection_global(item_emb.unsqueeze(0)).squeeze(0)
                    item_embeddings.append(l2_normalize(item_emb_proj))
            
            embeddings['items'] = item_embeddings
            
            # Scene embedding
            if parsed_query.scene:
                scene_query = f"photo taken in {parsed_query.scene}"
                scene_emb = self.clip_encoder.encode_text(scene_query)
                scene_emb_proj = self.projection_scene(scene_emb.unsqueeze(0)).squeeze(0)
                embeddings['scene'] = l2_normalize(scene_emb_proj)
            else:
                embeddings['scene'] = embeddings['global']
        
        return embeddings
    
    def stage1_search(self, query_embeddings: Dict[str, torch.Tensor], top_k: int = 200) -> Dict[str, float]:
        candidate_scores = defaultdict(lambda: {'global': 0, 'person': 0, 'items': 0, 'scene': 0, 'count': 0})
        
        # 1. Search global index
        global_emb = query_embeddings['global'].cpu().numpy().reshape(1, -1).astype('float32')
        global_distances, global_indices = self.indices['global'].search(global_emb, top_k)
        
        for idx, dist in zip(global_indices[0], global_distances[0]):
            if idx < len(self.image_ids):
                image_id = self.image_ids[idx]
                # Convert L2 distance to similarity
                similarity = 1.0 / (1.0 + dist)
                candidate_scores[image_id]['global'] = similarity
                candidate_scores[image_id]['count'] += 1
        
        # 2. Search person index
        person_emb = query_embeddings['person'].cpu().numpy().reshape(1, -1).astype('float32')
        person_distances, person_indices = self.indices['person'].search(person_emb, top_k)
        
        for idx, dist in zip(person_indices[0], person_distances[0]):
            if idx < len(self.image_ids):
                image_id = self.image_ids[idx]
                similarity = 1.0 / (1.0 + dist)
                candidate_scores[image_id]['person'] = similarity
                candidate_scores[image_id]['count'] += 1
        
        # 3. Search item index (for each item query)
        item_image_scores = defaultdict(list)
        
        for item_emb in query_embeddings['items']:
            item_emb_np = item_emb.cpu().numpy().reshape(1, -1).astype('float32')
            item_distances, item_indices = self.indices['item'].search(item_emb_np, top_k)
            
            for idx, dist in zip(item_indices[0], item_distances[0]):
                if idx < len(self.item_to_image_map):
                    image_id = self.item_to_image_map[idx]
                    similarity = 1.0 / (1.0 + dist)
                    item_image_scores[image_id].append(similarity)
        
        # Aggregate item scores (max or average)
        for image_id, scores in item_image_scores.items():
            # Use max score if multiple items match
            candidate_scores[image_id]['items'] = max(scores)
            candidate_scores[image_id]['count'] += 1
        
        # 4. Search scene index
        scene_emb = query_embeddings['scene'].cpu().numpy().reshape(1, -1).astype('float32')
        scene_distances, scene_indices = self.indices['scene'].search(scene_emb, top_k)
        
        for idx, dist in zip(scene_indices[0], scene_distances[0]):
            if idx < len(self.image_ids):
                image_id = self.image_ids[idx]
                similarity = 1.0 / (1.0 + dist)
                candidate_scores[image_id]['scene'] = similarity
                candidate_scores[image_id]['count'] += 1
        
        # Compute weighted aggregate scores
        final_scores = {}
        
        for image_id, scores in candidate_scores.items():
            # Weighted combination
            agg_score = (
                self.config.WEIGHT_GLOBAL * scores['global'] +
                self.config.WEIGHT_PERSON * scores['person'] +
                self.config.WEIGHT_ITEMS * scores['items'] +
                self.config.WEIGHT_SCENE * scores['scene']
            )
            
            # Normalize by number of index hits (favor images that match multiple indices)
            agg_score *= (scores['count'] / 4.0)
            
            final_scores[image_id] = agg_score
        
        return final_scores
    
    def stage2_rerank(self, candidate_scores: Dict[str, float], query_text: str, parsed_query: ParsedQuery, top_n: int = 100) -> List[Tuple[str, float]]:
        
        # Get top-N candidates from stage 1
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if len(sorted_candidates) == 0:
            return []
        
        # Rerank with BLIP-2
        reranked = []
        
        for image_id, stage1_score in sorted_candidates:
            # Load image
            image_idx = self.image_ids.index(image_id)
            image_path = self.image_paths[image_idx]
            
            try:
                image = load_image(image_path)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                reranked.append((image_id, stage1_score, 0.5, 1.0))
                continue
            
            # Compute cross-attention similarity
            reranker_score = self.reranker.compute_similarity(image, query_text)
            
            # Deterministic checks
            deterministic_score = self._deterministic_checks(image_id, parsed_query)
            
            # Compute final score
            final_score = (
                self.config.ALPHA_RERANK * reranker_score +
                (1 - self.config.ALPHA_RERANK) * stage1_score +
                self.config.GAMMA_DETERMINISTIC * deterministic_score
            )
            
            reranked.append((image_id, final_score, reranker_score, deterministic_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return [(img_id, score) for img_id, score, _, _ in reranked]
    
    def _deterministic_checks(self, image_id: str, parsed_query: ParsedQuery) -> float:
        score = 1.0
        
        meta = self.metadata.get(image_id, {})
        
        # Check if scene matches
        if parsed_query.scene:
            detected_scene = meta.get('scene', '')
            if detected_scene and detected_scene != parsed_query.scene:
                score *= 0.8  # Penalize scene mismatch
        
        # Check if style matches
        if parsed_query.style:
            style_scores = meta.get('style_scores', {})
            style_prob = style_scores.get(parsed_query.style, 0.1)
            score *= (0.5 + 0.5 * style_prob)  # Boost if style matches
        
        # Check item colors (compositional binding)
        if len(parsed_query.items) > 0:
            detected_items = meta.get('items', [])
            
            for query_item in parsed_query.items:
                query_color = query_item['color']
                query_type = query_item['type']
                
                if query_color:
                    # Check if any detected item has matching color
                    color_found = any(
                        item.get('color', '') == query_color
                        for item in detected_items
                    )
                    
                    if not color_found:
                        score *= 0.7  # Penalize color mismatch
        
        return score
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or self.config.FINAL_TOP_K
        
        print(f"\nQuery: {query}")
        
        # Parse query
        parsed_query = self.query_parser.parse(query)
        print(f"Parsed items: {parsed_query.items}")
        print(f"Parsed scene: {parsed_query.scene}")
        print(f"Parsed style: {parsed_query.style}")
        
        # Encode query
        query_embeddings = self.encode_query(query, parsed_query)
        
        # Stage 1: Fast search
        print(f"Stage 1: Searching {len(self.image_ids)} images...")
        candidate_scores = self.stage1_search(query_embeddings, top_k=self.config.STAGE1_TOP_K)
        print(f"Found {len(candidate_scores)} candidates")
        
        # Stage 2: Reranking
        print(f"Stage 2: Reranking top {self.config.STAGE2_TOP_K} candidates...")
        reranked_results = self.stage2_rerank(
            candidate_scores,
            query,
            parsed_query,
            top_n=self.config.STAGE2_TOP_K
        )
        
        # Format results
        results = []
        for image_id, score in reranked_results[:top_k]:
            image_idx = self.image_ids.index(image_id)
            image_path = self.image_paths[image_idx]
            meta = self.metadata.get(image_id, {})
            
            results.append({
                'image_id': image_id,
                'image_path': image_path,
                'score': float(score),
                'metadata': {
                    'scene': meta.get('scene', 'unknown'),
                    'style_scores': meta.get('style_scores', {}),
                    'num_items': len(meta.get('items', []))
                }
            })
        
        print(f"-> Retrieved {len(results)} results")
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = None) -> List[List[Dict]]:
        results = []
        
        for query in queries:
            query_results = self.retrieve(query, top_k=top_k)
            results.append(query_results)
        
        return results


def main():
    print("\nRetriever (Part B)")
    
    # Load config
    config = Config()
    
    # Create retriever
    retriever = MultiVectorRetriever(config=config)
    
    print("-> Testing with evaluation queries")
    
    for query in config.EVALUATION_QUERIES:
        print("\n" + "-" * 20)
        results = retriever.retrieve(query, top_k=10)
        
        print(f"\nTop 5 results for: {query}")
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {Path(result['image_path']).name} (score: {result['score']:.3f})")
            print(f"   Scene: {result['metadata']['scene']}, Items: {result['metadata']['num_items']}")


if __name__ == "__main__":
    main()