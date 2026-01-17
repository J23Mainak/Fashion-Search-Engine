import os
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    IMAGES_DIR = DATA_DIR / "images"  
    INDEX_DIR = DATA_DIR / "indices"
    METADATA_DIR = DATA_DIR / "metadata"
    MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
    
    for dir_path in [DATA_DIR, INDEX_DIR, METADATA_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Primary encoders
    CLIP_MODEL = "ViT-L/14"  # Main CLIP model
    FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"  # Fashion-specialized
    BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"  # Reranker
    
    # Detection and segmentation
    GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"
    
    # Pose estimation
    POSE_MODEL = "ViTPose"

    EMBED_DIM = 2048  # Final unified embedding dimension
    CLIP_DIM = 768  # CLIP ViT-L/14 output
    FASHION_CLIP_DIM = 512  # FashionCLIP output
    SCENE_DIM = 768
    POSE_DIM = 64
    STYLE_DIM = 128
    
    # Weights for different embedding branches (must sum to 1.0)
    WEIGHT_GLOBAL = 0.25
    WEIGHT_PERSON = 0.15
    WEIGHT_ITEMS = 0.40  # Split among detected items
    WEIGHT_SCENE = 0.15
    WEIGHT_POSE = 0.05
    
    # For 3200 images (your dataset)
    FAISS_INDEX_TYPE = "HNSW"  # High accuracy for smaller dataset
    HNSW_M = 32  # Number of connections
    HNSW_EF_CONSTRUCTION = 200  # Build-time accuracy
    HNSW_EF_SEARCH = 128  # Query-time accuracy
    
    # For scaling to 1M+ images:
    # FAISS_INDEX_TYPE = "HNSW_PQ"
    # PQ_M = 64  # Number of subquantizers
    # PQ_NBITS = 8  # Bits per subquantizer

    PERSON_CONFIDENCE_THRESHOLD = 0.5
    ITEM_CONFIDENCE_THRESHOLD = 0.3
    GROUNDING_TEXT_PROMPT = "person . shirt . pants . dress . jacket . coat . tie . shoes . hat . bag . watch . glasses . suit . blazer . skirt . shorts . sweater . hoodie"
    
    # Scene categories
    SCENE_CATEGORIES = ["office", "street", "park", "home", "indoor", "outdoor", "urban", "natural"]
    
    # Stage 1: Coarse retrieval
    STAGE1_TOP_K = 200  # Number of candidates from FAISS
    
    # Stage 2: Reranking
    STAGE2_TOP_K = 100  # Number to rerank
    FINAL_TOP_K = 10  # Final results to return
    
    # Reranker weights
    ALPHA_RERANK = 0.70  # Weight for reranker score
    BETA_GROUNDING = 0.25  # Weight for grounding confidence
    GAMMA_DETERMINISTIC = 0.05  # Weight for deterministic checks
    
    # (For fine-tuning)
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    LEARNING_RATE_PRETRAINED = 1e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 20
    TEMPERATURE_INFONCE = 0.07
    
    # Loss weights
    LOSS_WEIGHT_CONTRASTIVE = 1.0
    LOSS_WEIGHT_CAPTIONING = 0.5
    LOSS_WEIGHT_GROUNDING = 0.8
    LOSS_WEIGHT_ATTRIBUTE = 0.3
    
    # Hard negative mining
    HARD_NEGATIVE_RATIO = 0.3  # 30% of batch are compositional hard negatives
    
    IMAGE_SIZE = 1024  # For detection/segmentation
    CLIP_IMAGE_SIZE = 224  # For CLIP encoding
    MAX_DETECTIONS = 10  # Max number of items to detect per image
    
    USE_GPU = True
    GPU_ID = 0
    NUM_WORKERS = 4
    
    NUM_COLOR_BINS = 32  # For HSV histogram
    COLOR_PALETTE = {
        "red": ([0, 100, 100], [10, 255, 255]),
        "orange": ([11, 100, 100], [25, 255, 255]),
        "yellow": ([26, 100, 100], [35, 255, 255]),
        "green": ([36, 100, 100], [85, 255, 255]),
        "blue": ([86, 100, 100], [125, 255, 255]),
        "purple": ([126, 100, 100], [155, 255, 255]),
        "pink": ([156, 100, 100], [170, 255, 255]),
        "white": ([0, 0, 200], [180, 30, 255]),
        "black": ([0, 0, 0], [180, 255, 50]),
        "gray": ([0, 0, 51], [180, 30, 199]),
    }
    
    STYLE_CATEGORIES = [
        "casual", "formal", "business", "street", "sporty", 
        "elegant", "vintage", "bohemian", "minimalist", "trendy"
    ]
    
    LOG_LEVEL = "INFO"
    LOG_FILE = PROJECT_ROOT / "logs" / "retrieval.log"
    
    # Fashion-specific keywords for query enhancement
    GARMENT_KEYWORDS = [
        "shirt", "pants", "dress", "jacket", "coat", "tie", "suit",
        "blazer", "skirt", "shorts", "sweater", "hoodie", "shoes",
        "hat", "bag", "watch", "glasses", "belt", "scarf"
    ]
    
    COLOR_KEYWORDS = [
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "white", "black", "gray", "brown", "beige", "navy", "maroon"
    ]
    
    LOCATION_KEYWORDS = [
        "office", "street", "park", "home", "indoor", "outdoor",
        "urban", "city", "beach", "cafe", "restaurant"
    ]
    
    STYLE_KEYWORDS = [
        "casual", "formal", "professional", "business", "elegant",
        "sporty", "street", "vintage", "modern", "classic"
    ]
    
    EVALUATION_QUERIES = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting"
    ]
    
    @classmethod
    def get_device(cls):
        import torch
        if cls.USE_GPU and torch.cuda.is_available():
            return torch.device(f'cuda:{cls.GPU_ID}')
        return torch.device('cpu')
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("Fashion retrieval system - Configurations")

        print(f"Images Directory: {cls.IMAGES_DIR}")
        print(f"Index Directory: {cls.INDEX_DIR}")
        print(f"Device: {cls.get_device()}")
        print(f"Embedding Dimension: {cls.EMBED_DIM}")
        print(f"FAISS Index Type: {cls.FAISS_INDEX_TYPE}")
        print(f"Stage 1 Top-K: {cls.STAGE1_TOP_K}")
        print(f"Final Top-K: {cls.FINAL_TOP_K}")