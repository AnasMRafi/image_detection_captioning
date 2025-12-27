"""
Configuration file for Image Captioning Model.

This module contains all hyperparameters, paths, and settings for training
the CNN-LSTM attention-based image captioning model. Optimized for Apple
Silicon M3 Pro with 18GB unified memory.

Author: AI Assistant
Hardware Target: Apple Mac M3 Pro (18GB Unified Memory)
"""

import os
from pathlib import Path

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Base directory is the parent of this config file (backend folder)
BASE_DIR = Path(__file__).parent.absolute()

# Dataset paths - Flickr8k dataset structure
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "Images"
CAPTIONS_FILE = DATASET_DIR / "captions.txt"

# Model output paths - where trained weights will be saved
MODELS_DIR = BASE_DIR / "models"
ENCODER_PATH = MODELS_DIR / "encoder.pth"
DECODER_PATH = MODELS_DIR / "decoder.pth"
VOCAB_PATH = MODELS_DIR / "vocab.pkl"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# ==============================================================================
# These dimensions define the neural network architecture

# Encoder (CNN) Configuration
ENCODER_DIM = 2048          # Output dimension from InceptionV3/ResNet backbone
                            # InceptionV3 mixed_7c layer outputs 2048 channels

# Attention Mechanism Configuration  
ATTENTION_DIM = 512         # Hidden dimension for Bahdanau attention scoring
                            # Larger = more expressive but slower

# Decoder (LSTM) Configuration
DECODER_DIM = 512           # LSTM hidden state dimension
                            # Controls the "memory capacity" of the decoder
EMBEDDING_DIM = 256         # Word embedding dimension
                            # Maps vocabulary indices to dense vectors

# Regularization
DROPOUT = 0.5               # Dropout probability for regularization
                            # Applied to LSTM outputs and attention

# ==============================================================================
# VOCABULARY SETTINGS
# ==============================================================================
# Special tokens for sequence modeling
PAD_TOKEN = "<PAD>"         # Padding token for batching variable-length sequences
START_TOKEN = "<START>"     # Signals the beginning of a caption
END_TOKEN = "<END>"         # Signals the end of a caption
UNK_TOKEN = "<UNK>"         # Unknown token for out-of-vocabulary words

# Vocabulary filtering
MIN_WORD_FREQ = 5           # Words appearing less than this are replaced with <UNK>
                            # Reduces vocabulary size and improves generalization
MAX_VOCAB_SIZE = 10000      # Maximum vocabulary size (safety limit)

# Sequence length
MAX_CAPTION_LENGTH = 50     # Maximum number of words per caption
                            # Longer captions are truncated

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
# Optimized for 18GB M3 Pro unified memory

# Batch sizes - carefully tuned for memory constraints
# Rule of thumb: batch_size * model_size < available_memory
BATCH_SIZE = 24             # Training batch size with encoder frozen
                            # Can increase to 32 if memory allows
BATCH_SIZE_FINE_TUNE = 8    # Batch size when encoder is unfrozen
                            # Smaller due to additional gradient memory

# Workers for data loading
NUM_WORKERS = 4             # Number of parallel data loading processes
                            # M3 Pro has 12 cores, 4 workers is efficient

# Training duration
EPOCHS = 20                 # Maximum number of training epochs
EARLY_STOPPING_PATIENCE = 5 # Stop if validation doesn't improve for N epochs

# Optimizer settings
LEARNING_RATE = 4e-4        # Initial learning rate for Adam optimizer
ENCODER_LR = 1e-5           # Lower LR for fine-tuning pretrained encoder
WEIGHT_DECAY = 1e-5         # L2 regularization to prevent overfitting

# Gradient clipping - prevents exploding gradients in RNNs
GRAD_CLIP = 5.0             # Maximum gradient norm

# Learning rate scheduling
LR_PATIENCE = 3             # Reduce LR if no improvement for N epochs
LR_FACTOR = 0.8             # Multiply LR by this factor when reducing

# Teacher forcing - during training, use ground truth vs predicted words
TEACHER_FORCING_RATIO = 1.0 # Start at 1.0 (always use ground truth)
                            # Can decay to 0.5 during training

# ==============================================================================
# DATA SPLIT RATIOS
# ==============================================================================
TRAIN_RATIO = 0.8           # 80% of data for training
VAL_RATIO = 0.1             # 10% for validation (hyperparameter tuning)
TEST_RATIO = 0.1            # 10% for final evaluation (BLEU scores)

# ==============================================================================
# IMAGE PREPROCESSING
# ==============================================================================
# InceptionV3 expects 299x299 images, but 224x224 works with adaptive pooling
IMAGE_SIZE = 299            # Input image size (width and height)

# ImageNet normalization statistics (required for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==============================================================================
# INFERENCE SETTINGS
# ==============================================================================
BEAM_SIZE = 3               # Number of beams for beam search decoding
                            # Higher = better quality but slower

# ==============================================================================
# LOGGING AND CHECKPOINTING
# ==============================================================================
LOG_INTERVAL = 100          # Print training stats every N batches
CHECKPOINT_INTERVAL = 1     # Save checkpoint every N epochs
SAVE_BEST_ONLY = True       # Only save model if validation improves

# ==============================================================================
# DEVICE CONFIGURATION (AUTO-DETECTED)
# ==============================================================================
# This will be set at runtime based on available hardware
# Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
DEVICE = None               # Set by training script

def get_device():
    """
    Automatically detect and return the best available compute device.
    
    Priority order:
    1. MPS (Metal Performance Shaders) - For Apple Silicon Macs
    2. CUDA - For NVIDIA GPUs
    3. CPU - Fallback for any system
    
    Returns:
        torch.device: The selected compute device
    """
    import torch
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using MPS (Metal Performance Shaders) on Apple Silicon")
        print(f"   Memory: ~18GB unified memory available")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU - Training will be significantly slower")
    
    return device


if __name__ == "__main__":
    # Print configuration summary when run directly
    print("=" * 60)
    print("IMAGE CAPTIONING MODEL CONFIGURATION")
    print("=" * 60)
    print(f"\n📁 Paths:")
    print(f"   Dataset: {DATASET_DIR}")
    print(f"   Images: {IMAGES_DIR}")
    print(f"   Captions: {CAPTIONS_FILE}")
    print(f"   Models: {MODELS_DIR}")
    
    print(f"\n🧠 Model Architecture:")
    print(f"   Encoder dim: {ENCODER_DIM}")
    print(f"   Attention dim: {ATTENTION_DIM}")
    print(f"   Decoder dim: {DECODER_DIM}")
    print(f"   Embedding dim: {EMBEDDING_DIM}")
    
    print(f"\n📊 Training:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Early stopping: {EARLY_STOPPING_PATIENCE} epochs")
    
    print(f"\n🖥️  Device:")
    get_device()
    print("=" * 60)
