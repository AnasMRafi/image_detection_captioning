"""
FastAPI Server for Image Captioning.

This server provides REST API endpoints for generating captions from images
using the trained CNN-LSTM attention model.

Endpoints:
    POST /caption  - Generate caption for an uploaded image
    GET /health    - Server health check

Hardware: Optimized for Apple M3 Pro with MPS acceleration
Expected Latency: < 500ms per image

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Author: AI Assistant
"""

import io
import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import local modules
from config import (
    ENCODER_PATH, DECODER_PATH, VOCAB_PATH,
    ENCODER_DIM, ATTENTION_DIM, DECODER_DIM, EMBEDDING_DIM, DROPOUT,
    IMAGE_SIZE, MAX_CAPTION_LENGTH, BEAM_SIZE,
    get_device
)
from utils import Vocabulary, get_image_transforms


# ==============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ==============================================================================

class CaptionResponse(BaseModel):
    """Response model for caption endpoint."""
    success: bool
    caption: str
    confidence: float
    inference_time_ms: float
    alternatives: Optional[List[dict]] = None


class DetectionResult(BaseModel):
    """Single detection result."""
    box: List[float]  # [x1, y1, x2, y2]
    label: int
    score: float
    class_name: str


class DetectionResponse(BaseModel):
    """Response model for detection endpoint."""
    success: bool
    detections: List[DetectionResult]
    count: int
    inference_time_ms: float


class AnalysisResponse(BaseModel):
    """Response model for combined detection + captioning endpoint."""
    success: bool
    caption: str
    caption_confidence: float
    detections: List[DetectionResult]
    detection_count: int
    combined_description: str  # Natural language combining both
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    device: str
    models_loaded: dict  # Changed to dict for multiple models


class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = False
    error: str
    code: str


# ==============================================================================
# MODEL DEFINITIONS (Same as in training notebook)
# ==============================================================================

class EncoderCNN(nn.Module):
    """
    CNN Encoder for image feature extraction.
    
    Uses pretrained InceptionV3 backbone to extract spatial features
    that can be attended to by the decoder.
    
    Output Shape: [batch, 64, encoder_dim] (64 spatial locations)
    """
    
    def __init__(self, encoder_dim: int = ENCODER_DIM):
        super(EncoderCNN, self).__init__()
        
        # Load pretrained InceptionV3
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.inception.aux_logits = False
        
        # Replace final FC layer with identity (we want features, not classification)
        self.inception.fc = nn.Identity()
        
        # Project 2048-dim features to spatial feature map for attention
        # Creates 64 "pseudo-spatial" locations
        self.feature_projection = nn.Linear(2048, 64 * encoder_dim)
        self.encoder_dim = encoder_dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        # Get InceptionV3 features (after avgpool): [batch, 2048]
        features = self.inception(images)
        
        # Project to spatial feature map: [batch, 64 * encoder_dim]
        features = self.feature_projection(features)
        
        # Reshape to [batch, 64, encoder_dim] for attention
        batch_size = features.size(0)
        features = features.view(batch_size, 64, self.encoder_dim)
        
        return features


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    
    Computes attention weights over encoder features based on decoder state.
    """
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super(BahdanauAttention, self).__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
    
    def forward(self, encoder_out, decoder_hidden):
        encoder_proj = self.W_encoder(encoder_out)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)
        combined = torch.tanh(encoder_proj + decoder_proj)
        attention_scores = self.V(combined).squeeze(2)
        alpha = F.softmax(attention_scores, dim=1)
        context = (alpha.unsqueeze(2) * encoder_out).sum(dim=1)
        return context, alpha


class DecoderWithAttention(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention + Object Detection Guidance.
    Enhanced to accept detected object labels for guided caption generation.
    """
    
    # Object embedding constants
    NUM_OBJECT_CLASSES = 80
    OBJECT_EMBED_DIM = 128
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        encoder_dim: int = ENCODER_DIM,
        decoder_dim: int = DECODER_DIM,
        attention_dim: int = ATTENTION_DIM,
        dropout: float = DROPOUT,
        num_object_classes: int = 80,
        object_embed_dim: int = 128
    ):
        super(DecoderWithAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Object embedding for detected objects (detection-guided captioning)
        self.object_embedding = nn.Embedding(num_object_classes + 1, object_embed_dim)
        self.object_fc = nn.Linear(object_embed_dim, decoder_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
    
    def init_hidden_state(self, encoder_out, detected_objects=None):
        """Initialize LSTM states with optional object context."""
        mean_encoder = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        
        # Add object context to hidden state if objects detected
        if detected_objects is not None and detected_objects.numel() > 0:
            obj_embed = self.object_embedding(detected_objects)
            obj_context = obj_embed.mean(dim=1)
            obj_hidden = self.object_fc(obj_context)
            h = h + obj_hidden
        
        return h, c


# ==============================================================================
# INFERENCE FUNCTIONS
# ==============================================================================

def beam_search_inference(
    encoder: nn.Module,
    decoder: nn.Module,
    image: torch.Tensor,
    vocab: Vocabulary,
    beam_size: int = BEAM_SIZE,
    max_length: int = MAX_CAPTION_LENGTH,
    device: torch.device = None,
    detected_objects: torch.Tensor = None  # NEW: detected object indices
) -> tuple:
    """
    Generate caption using beam search decoding.
    
    Args:
        encoder: Trained CNN encoder
        decoder: Trained LSTM decoder
        image: Preprocessed image tensor [1, 3, H, W]
        vocab: Vocabulary instance
        beam_size: Number of beams
        max_length: Maximum caption length
        device: Computation device
        detected_objects: Tensor of detected COCO class indices [1, num_objects]
    
    Returns:
        Tuple of (best_caption, confidence, all_captions)
    """
    if device is None:
        device = image.device
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode image
        encoder_out = encoder(image)  # [1, 64, encoder_dim]
        
        k = beam_size
        vocab_size = decoder.vocab_size
        
        # Initialize beams
        seqs = torch.LongTensor([[vocab.word2idx[vocab.START_TOKEN]]]).to(device)
        seqs = seqs.expand(k, -1)
        top_k_scores = torch.zeros(k).to(device)
        
        # Expand encoder output for beam
        enc_out = encoder_out.expand(k, -1, -1)
        
        # Expand detected objects for beam (if provided)
        det_obj_expanded = None
        if detected_objects is not None:
            det_obj_expanded = detected_objects.expand(k, -1)
        
        # Initialize hidden states (with object context if available)
        h, c = decoder.init_hidden_state(enc_out, det_obj_expanded)
        
        complete_seqs = []
        complete_scores = []
        
        for step in range(max_length):
            prev_words = seqs[:, -1]
            embeddings = decoder.embedding(prev_words)
            
            context, alpha = decoder.attention(enc_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context
            
            h, c = decoder.decode_step(
                torch.cat([embeddings, context], dim=1), (h, c)
            )
            
            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.unsqueeze(1) + scores
            
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)
                prev_beam_idx = torch.zeros(k, dtype=torch.long, device=device)
            else:
                scores = scores.view(-1)
                top_k_scores, top_k_idx = scores.topk(k, dim=0)
                prev_beam_idx = top_k_idx // vocab_size
                top_k_words = top_k_idx % vocab_size
            
            seqs = torch.cat([seqs[prev_beam_idx], top_k_words.unsqueeze(1)], dim=1)
            enc_out = enc_out[prev_beam_idx]
            h = h[prev_beam_idx]
            c = c[prev_beam_idx]
            
            # Check for completed sequences
            incomplete_idx = []
            for idx in range(len(top_k_words)):
                if top_k_words[idx] == vocab.word2idx[vocab.END_TOKEN]:
                    complete_seqs.append(seqs[idx].tolist())
                    complete_scores.append(top_k_scores[idx].item())
                else:
                    incomplete_idx.append(idx)
            
            if len(incomplete_idx) == 0:
                break
            
            if len(incomplete_idx) < k:
                incomplete_idx = torch.LongTensor(incomplete_idx).to(device)
                seqs = seqs[incomplete_idx]
                enc_out = enc_out[incomplete_idx]
                h = h[incomplete_idx]
                c = c[incomplete_idx]
                top_k_scores = top_k_scores[incomplete_idx]
                k = len(incomplete_idx)
        
        # Handle case where no sequence completed
        if len(complete_seqs) == 0:
            complete_seqs = [seqs[0].tolist()]
            complete_scores = [top_k_scores[0].item()]
        
        # Get all captions sorted by score
        sorted_idx = np.argsort(complete_scores)[::-1]
        
        all_captions = []
        for idx in sorted_idx[:3]:  # Top 3 alternatives
            caption = vocab.decode(complete_seqs[idx], skip_special=True)
            score = np.exp(complete_scores[idx] / len(complete_seqs[idx]))
            all_captions.append({
                "caption": caption,
                "score": float(score)
            })
        
        best_caption = all_captions[0]["caption"]
        best_score = all_captions[0]["score"]
        
        return best_caption, best_score, all_captions


# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Image Captioning API",
    description="Generate natural language captions for images using CNN-LSTM with attention",
    version="1.0.0"
)

# Add CORS middleware for Flutter app access
# In production, restrict origins to your app's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and config
encoder = None
decoder = None
vocab = None
detector = None  # Faster R-CNN object detection model
device = None
transform = None

# COCO class labels for detection
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


@app.on_event("startup")
async def load_models():
    """
    Load models on server startup.
    
    This is called once when the server starts, not for each request.
    Loading models at startup improves response time significantly.
    """
    global encoder, decoder, vocab, detector, device, transform
    
    print("=" * 60)
    print("LOADING AI MODELS")
    print("=" * 60)
    
    # Detect device
    device = get_device()
    
    # ====================
    # LOAD CAPTIONING MODELS
    # ====================
    if ENCODER_PATH.exists() and DECODER_PATH.exists() and VOCAB_PATH.exists():
        # Load vocabulary
        print("\n📚 Loading vocabulary...")
        vocab = Vocabulary.load(str(VOCAB_PATH))
        
        # Initialize and load encoder
        print("\n🖼️  Loading caption encoder...")
        encoder = EncoderCNN(encoder_dim=ENCODER_DIM)
        encoder.load_state_dict(torch.load(str(ENCODER_PATH), map_location=device))
        encoder = encoder.to(device)
        encoder.eval()
        print(f"   Encoder loaded from: {ENCODER_PATH}")
        
        # Initialize and load decoder
        print("\n📝 Loading caption decoder...")
        decoder = DecoderWithAttention(
            vocab_size=len(vocab),
            embedding_dim=EMBEDDING_DIM,
            encoder_dim=ENCODER_DIM,
            decoder_dim=DECODER_DIM,
            attention_dim=ATTENTION_DIM,
            dropout=DROPOUT
        )
        # decoder.load_state_dict(torch.load(str(DECODER_PATH), map_location=device), strict=False)
        decoder.load_state_dict(torch.load(str(DECODER_PATH), map_location=device))
        decoder = decoder.to(device)
        decoder.eval()
        print(f"   Decoder loaded from: {DECODER_PATH}")
        
        # Initialize image transforms
        transform = get_image_transforms(IMAGE_SIZE, is_training=False)
        print("✅ Captioning models loaded!")
    else:
        print("\n⚠️  Captioning models not found - caption endpoint will be unavailable")
    
    # ====================
    # LOAD DETECTION MODEL (YOLOv8 - Best Accuracy + Speed Balance)
    # ====================
    print("\n🔍 Loading object detection model...")
    try:
        from ultralytics import YOLO
        
        # Using YOLOv12x (extra large) for BEST accuracy - State of the Art 2024/2025
        # Options: yolo12n, yolo12s, yolo12m, yolo12l, yolo12x
        detector = YOLO('yolo12x.pt')  # Downloads automatically if not present
        
        # Set device for YOLO
        detector.to(device)
        
        print("   Model: YOLOv12x (extra large) - State of the Art")
        print("   Classes: 80 (COCO)")
        print("   mAP: 55.2% - Best accuracy available")
        print("✅ Detection model loaded!")
    except Exception as e:
        print(f"⚠️  Could not load detection model: {e}")
        detector = None
    
    # ====================
    # WARM UP MODELS (Pre-compile MPS/CUDA kernels)
    # ====================
    print("\n🔥 Warming up models (first inference is slow)...")
    try:
        # Create dummy image tensor
        dummy_image = torch.randn(1, 3, 299, 299).to(device)
        
        # Warm up captioning model
        if encoder is not None and decoder is not None:
            with torch.no_grad():
                _ = encoder(dummy_image)
            print("   ✅ Caption encoder warmed up")
        
        # Warm up detection model (YOLOv8)
        if detector is not None:
            # Create a dummy image as numpy array for YOLO
            import numpy as np
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = detector.predict(dummy_img, verbose=False)
            print("   ✅ Detection model warmed up")
        
        print("   Models ready for fast inference!")
    except Exception as e:
        print(f"   ⚠️ Warmup failed (not critical): {e}")
    
    print("\n" + "=" * 60)
    print("✅ SERVER READY - Models warmed up!")
    print("=" * 60)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check server health and model status.
    
    Returns:
        Health status including device info and model availability
    """
    return HealthResponse(
        status="healthy",
        device=str(device) if device else "not initialized",
        models_loaded={
            "captioner": encoder is not None and decoder is not None,
            "detector": detector is not None
        }
    )


@app.post("/caption", response_model=CaptionResponse)
async def generate_caption(image: UploadFile = File(...)):
    """
    Generate a caption for an uploaded image.
    
    Args:
        image: Image file (JPEG, PNG, etc.)
    
    Returns:
        Generated caption with confidence score and alternatives
    
    Raises:
        HTTPException: If image is invalid or models not loaded
    """
    # Check if models are loaded
    if encoder is None or decoder is None or vocab is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "Models not loaded. Please check server logs.",
                "code": "MODEL_NOT_LOADED"
            }
        )
    
    # Validate file type (be lenient - allow None and infer from filename)
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    
    content_type = image.content_type
    filename = image.filename or ""
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ""
    
    # Check either content_type OR file extension
    type_ok = content_type in allowed_types if content_type else False
    ext_ok = f".{file_ext}" in allowed_extensions if file_ext else False
    
    if not (type_ok or ext_ok):
        # If neither matches and we have actual data, try to process anyway
        # (PIL will fail if it's not a valid image)
        if content_type and content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": f"Invalid image type: {content_type}. Allowed: JPEG, PNG, WebP",
                    "code": "INVALID_IMAGE_TYPE"
                }
            )
    
    # Check file size (max 10MB)
    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Image too large. Maximum size is 10MB.",
                "code": "IMAGE_TOO_LARGE"
            }
        )
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load and preprocess image
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Generate caption using beam search
        caption, confidence, alternatives = beam_search_inference(
            encoder, decoder, image_tensor, vocab,
            beam_size=BEAM_SIZE, max_length=MAX_CAPTION_LENGTH, device=device
        )
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        return CaptionResponse(
            success=True,
            caption=caption,
            confidence=min(confidence, 1.0),  # Cap at 1.0
            inference_time_ms=round(inference_time_ms, 2),
            alternatives=alternatives[1:] if len(alternatives) > 1 else None
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error processing image: {str(e)}",
                "code": "PROCESSING_ERROR"
            }
        )


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    image: UploadFile = File(...),
    confidence: float = 0.5
):
    """
    Detect objects in an uploaded image using YOLOv8.
    
    Args:
        image: Image file (JPEG, PNG, etc.)
        confidence: Minimum confidence threshold (0.0 - 1.0)
    
    Returns:
        List of detected objects with bounding boxes and class names
    """
    # Check if detection model is loaded
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "Object detection model not loaded.",
                "code": "MODEL_NOT_LOADED"
            }
        )
    
    # Validate confidence threshold
    confidence = max(0.1, min(0.99, confidence))
    
    # Read image contents
    contents = await image.read()
    
    # Check file size (max 10MB)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Image too large. Maximum size is 10MB.",
                "code": "IMAGE_TOO_LARGE"
            }
        )
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load image
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_size = pil_image.size  # (width, height)
        
        # Convert PIL to numpy array for YOLO
        import numpy as np
        image_array = np.array(pil_image)
        
        # Run YOLOv8 detection
        results = detector.predict(
            image_array, 
            conf=confidence,
            verbose=False,
            device=device
        )[0]
        
        # Extract detections from YOLO results
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            classes = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()
            
            # YOLO uses COCO class names
            class_names = results.names
            
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box
                detection = DetectionResult(
                    box=[float(x1), float(y1), float(x2), float(y2)],
                    label=int(cls),
                    score=float(score),
                    class_name=class_names.get(cls, "unknown")
                )
                detections.append(detection)
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            success=True,
            detections=detections,
            count=len(detections),
            inference_time_ms=round(inference_time_ms, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Error processing image: {str(e)}",
                "code": "PROCESSING_ERROR"
            }
        )


# ==============================================================================
# COMBINED ANALYSIS ENDPOINT (Detection + Guided Captioning)
# ==============================================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    confidence: float = 0.5
):
    """
    Combined detection + captioning endpoint.
    Runs YOLO detection and uses detected objects to guide caption generation.
    """
    # Check if both models are loaded
    if detector is None or encoder is None or decoder is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "Models not fully loaded.",
                "code": "MODEL_NOT_LOADED"
            }
        )
    
    contents = await image.read()
    
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail={"error": "Image too large"})
    
    try:
        start_time = time.time()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # STEP 1: Run YOLO Detection
        import numpy as np
        image_array = np.array(pil_image)
        
        results = detector.predict(image_array, conf=confidence, verbose=False, device=device)[0]
        
        detections = []
        detected_class_indices = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()
            class_names = results.names
            
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box
                detection = DetectionResult(
                    box=[float(x1), float(y1), float(x2), float(y2)],
                    label=int(cls),
                    score=float(score),
                    class_name=class_names.get(cls, "unknown")
                )
                detections.append(detection)
                detected_class_indices.append(int(cls))
        
        # STEP 2: Run Guided Captioning
        transform = get_image_transforms(IMAGE_SIZE, is_training=False)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Prepare detected objects tensor
        detected_objects = None
        if detected_class_indices:
            top_objects = detected_class_indices[:10]
            detected_objects = torch.LongTensor([top_objects]).to(device)
        
        # Generate caption with object guidance
        caption, caption_confidence, alternatives = beam_search_inference(
            encoder, decoder, image_tensor, vocab,
            beam_size=BEAM_SIZE, max_length=MAX_CAPTION_LENGTH, device=device,
            detected_objects=detected_objects
        )
        
        # STEP 3: Create Combined Description
        if detections:
            object_list = ", ".join([d.class_name for d in detections[:5]])
            combined = f"{caption} (Detected: {object_list})"
        else:
            combined = caption
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            success=True,
            caption=caption,
            caption_confidence=min(caption_confidence, 1.0),
            detections=detections,
            detection_count=len(detections),
            combined_description=combined,
            inference_time_ms=round(inference_time_ms, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": f"Error analyzing image: {str(e)}"}
        )


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Image Captioning API Server...")
    print("Access the API at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
