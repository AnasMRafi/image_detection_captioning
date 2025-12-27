"""
Image Captioning Model - Jupyter Notebook Version

This script is designed to be copied cell-by-cell into a Jupyter notebook.
Each section marked with ### CELL [N] is a separate cell.

The model architecture follows the "Show, Attend and Tell" paper:
- Encoder: InceptionV3 CNN for feature extraction
- Attention: Bahdanau (additive) attention mechanism  
- Decoder: LSTM with attention for caption generation

Hardware Target: Apple Mac M3 Pro with 18GB unified memory
Expected Training Time: 2-4 hours for 20 epochs
"""

# ==============================================================================
### CELL 1: Environment Setup & Device Detection
# ==============================================================================
# Run this cell first to set up the computing environment

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import re
from typing import Dict, List, Tuple, Optional

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Detect and configure the best available device
# Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("[OK] Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"[OK] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("[!!] Using CPU (training will be slower)")

# Verify MPS is working correctly on Apple Silicon
if device.type == 'mps':
    try:
        test_tensor = torch.ones(1, device=device)
        _ = test_tensor * 2
        print("   MPS test passed: Tensor operations working correctly")
    except Exception as e:
        print(f"   [!!] MPS test failed: {e}")
        print("   Falling back to CPU")
        device = torch.device('cpu')

print(f"\n==== Final device selected: {device}")


# ==============================================================================
### CELL 2: Configuration Constants
# ==============================================================================
# All hyperparameters and paths in one place

# Path Configuration - ADJUST THESE IF NEEDED
BASE_DIR = Path.cwd()  # Current directory (should be backend/)
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "Images"
CAPTIONS_FILE = DATASET_DIR / "captions.txt"
MODELS_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model save paths
ENCODER_PATH = MODELS_DIR / "encoder.pth"
DECODER_PATH = MODELS_DIR / "decoder.pth"
VOCAB_PATH = MODELS_DIR / "vocab.pkl"

# Model Architecture Hyperparameters
ENCODER_DIM = 2048      # InceptionV3 feature dimension
ATTENTION_DIM = 512     # Attention layer dimension
DECODER_DIM = 512       # LSTM hidden state dimension
EMBEDDING_DIM = 256     # Word embedding dimension
DROPOUT = 0.5           # Dropout rate

# Vocabulary Settings
MIN_WORD_FREQ = 5       # Minimum word frequency to include in vocab
MAX_CAPTION_LENGTH = 50 # Maximum caption length

# Training Hyperparameters (Optimized for M3 Pro 18GB RAM)
BATCH_SIZE = 24
NUM_WORKERS = 0         # Set to 0 for MPS compatibility
EPOCHS = 20
LEARNING_RATE = 4e-4
GRAD_CLIP = 5.0
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 3
LR_FACTOR = 0.8

# Data Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# Image Preprocessing
IMAGE_SIZE = 299        # InceptionV3 input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference
BEAM_SIZE = 3

print("[OK] Configuration loaded")
print(f"   Dataset: {DATASET_DIR}")
print(f"   Models will be saved to: {MODELS_DIR}")


# ==============================================================================
### CELL 3: Utility Functions
# ==============================================================================
# Helper functions for data processing

def parse_captions(captions_file: str, images_dir: str) -> Dict[str, List[str]]:
    """
    Parse the Flickr8k captions.txt file.
    
    The file format is:
        image,caption (header row - skipped)
        1000268201_693b08cb0e.jpg,A child in a pink dress...
        1000268201_693b08cb0e.jpg,A girl going into a wooden building.
        ...
    
    Args:
        captions_file: Path to captions.txt
        images_dir: Path to Images directory
        
    Returns:
        Dict mapping image filenames to list of captions
    """
    image_to_captions = {}
    images_path = Path(images_dir)
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header row
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Split only on first comma (caption may contain commas)
        parts = line.split(',', 1)
        if len(parts) != 2:
            continue
        
        image_name, caption = parts
        image_name = image_name.strip()
        caption = caption.strip()
        
        # Verify image exists
        if not (images_path / image_name).exists():
            continue
        
        # Clean caption
        caption = caption.lower()
        caption = re.sub(r'[^\w\s\']', '', caption)
        caption = ' '.join(caption.split())
        
        if image_name not in image_to_captions:
            image_to_captions[image_name] = []
        image_to_captions[image_name].append(caption)
    
    print(f"[OK] Parsed {sum(len(v) for v in image_to_captions.values())} captions")
    print(f"   Unique images: {len(image_to_captions)}")
    
    return image_to_captions


class Vocabulary:
    """Vocabulary for encoding/decoding captions."""
    
    PAD_TOKEN = '<PAD>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    UNK_TOKEN = '<UNK>'
    
    def __init__(self):
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.START_TOKEN: 1,
            self.END_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
    
    def build_vocab(self, captions: List[str], min_freq: int = 5):
        """Build vocabulary from list of captions."""
        # Count word frequencies
        for caption in captions:
            words = caption.lower().split()
            self.word_freq.update(words)
        
        # Add words meeting frequency threshold
        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"[OK] Vocabulary built: {len(self.word2idx)} words")
    
    def encode(self, caption: str, max_length: int = 50) -> List[int]:
        """Encode caption to indices with START/END tokens."""
        words = caption.lower().split()
        indices = [self.word2idx[self.START_TOKEN]]
        
        for word in words[:max_length - 2]:
            indices.append(self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]))
        
        indices.append(self.word2idx[self.END_TOKEN])
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Decode indices back to caption string."""
        special = {0, 1, 2, 3} if skip_special else set()
        words = [self.idx2word.get(idx, self.UNK_TOKEN) 
                 for idx in indices if idx not in special]
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        vocab = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        return vocab


def get_image_transforms(image_size: int, is_training: bool = False):
    """Get image transformation pipeline."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


print("[OK] Utility functions defined")


# ==============================================================================
### CELL 4: Load and Parse Dataset
# ==============================================================================
# Load the Flickr8k dataset

print("=" * 60)
print("LOADING FLICKR8K DATASET")
print("=" * 60)

# Parse captions from the CSV file
image_to_captions = parse_captions(str(CAPTIONS_FILE), str(IMAGES_DIR))

# Display sample data
print("\n**** Sample Images and Captions:")
sample_images = list(image_to_captions.keys())[:3]

for img_name in sample_images:
    print(f"\n   Image: {img_name}")
    for i, caption in enumerate(image_to_captions[img_name][:2], 1):
        print(f"   Caption {i}: {caption[:60]}...")


# ==============================================================================
### CELL 5: Build Vocabulary
# ==============================================================================
# Create the vocabulary from all captions

print("\n" + "=" * 60)
print("BUILDING VOCABULARY")
print("=" * 60)

# Collect all captions
all_captions = []
for captions in image_to_captions.values():
    all_captions.extend(captions)

print(f"   Total captions: {len(all_captions):,}")

# Build vocabulary
vocab = Vocabulary()
vocab.build_vocab(all_captions, min_freq=MIN_WORD_FREQ)

# Save vocabulary
vocab.save(str(VOCAB_PATH))
print(f"   Saved to: {VOCAB_PATH}")


# ==============================================================================
### CELL 6: Dataset Class and DataLoaders
# ==============================================================================
# Custom PyTorch Dataset for image-caption pairs

class FlickrDataset(Dataset):
    """PyTorch Dataset for Flickr8k image captioning."""
    
    def __init__(
        self,
        image_names: List[str],
        image_to_captions: Dict[str, List[str]],
        vocab: Vocabulary,
        images_dir: str,
        transform = None,
        max_length: int = MAX_CAPTION_LENGTH
    ):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform or get_image_transforms(IMAGE_SIZE, is_training=False)
        self.max_length = max_length
        
        # Expand to (image_name, caption) pairs
        self.samples = []
        for img_name in image_names:
            for caption in image_to_captions.get(img_name, []):
                self.samples.append((img_name, caption))
        
        print(f"   Created dataset with {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, caption_text = self.samples[idx]
        
        # Load and transform image
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Encode caption
        caption_indices = self.vocab.encode(caption_text, self.max_length)
        cap_length = len(caption_indices)
        
        # Pad to max_length
        padded = caption_indices + [0] * (self.max_length - cap_length)
        caption = torch.LongTensor(padded[:self.max_length])
        
        return image, caption, cap_length


def collate_fn(batch):
    """Sort by caption length for pack_padded_sequence."""
    batch.sort(key=lambda x: x[2], reverse=True)
    images, captions, lengths = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    lengths = torch.LongTensor(lengths)
    return images, captions, lengths


# Create train/val/test splits
print("\n" + "=" * 60)
print("CREATING DATA SPLITS")
print("=" * 60)

all_images = list(image_to_captions.keys())
random.shuffle(all_images)

n_total = len(all_images)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_images = all_images[:n_train]
val_images = all_images[n_train:n_train + n_val]
test_images = all_images[n_train + n_val:]

print(f"   Total unique images: {n_total:,}")
print(f"   Train: {len(train_images):,}")
print(f"   Validation: {len(val_images):,}")
print(f"   Test: {len(test_images):,}")

# Create datasets
train_transform = get_image_transforms(IMAGE_SIZE, is_training=True)
eval_transform = get_image_transforms(IMAGE_SIZE, is_training=False)

print("\n   Creating datasets...")
train_dataset = FlickrDataset(train_images, image_to_captions, vocab, str(IMAGES_DIR), transform=train_transform)
val_dataset = FlickrDataset(val_images, image_to_captions, vocab, str(IMAGES_DIR), transform=eval_transform)
test_dataset = FlickrDataset(test_images, image_to_captions, vocab, str(IMAGES_DIR), transform=eval_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

print(f"\n   Train batches: {len(train_loader):,}")
print(f"   Validation batches: {len(val_loader):,}")
print(f"   Test batches: {len(test_loader):,}")


# ==============================================================================
### CELL 7: Encoder Model (CNN)
# ==============================================================================
# InceptionV3-based encoder for image feature extraction

class EncoderCNN(nn.Module):
    """CNN Encoder using pretrained InceptionV3."""
    
    def __init__(self, encoder_dim: int = ENCODER_DIM, fine_tune: bool = False):
        super(EncoderCNN, self).__init__()
        
        # Load pretrained InceptionV3
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.inception.aux_logits = False
        
        # Replace the final FC layer with identity (we want features, not classification)
        self.inception.fc = nn.Identity()
        
        # The output will be 2048-dim after avgpool
        # We need to reshape to get spatial features for attention
        # We'll use a linear projection to create pseudo-spatial locations
        self.feature_projection = nn.Linear(2048, 64 * encoder_dim)
        self.encoder_dim = encoder_dim
        
        # Freeze backbone initially
        self.set_fine_tune(fine_tune)
    
    def set_fine_tune(self, fine_tune: bool = False):
        for param in self.inception.parameters():
            param.requires_grad = fine_tune
        for param in self.feature_projection.parameters():
            param.requires_grad = True
    
    def forward(self, images):
        """
        Args:
            images: [batch, 3, 299, 299]
        Returns:
            features: [batch, 64, encoder_dim]
        """
        # Get InceptionV3 features (after avgpool, before original fc)
        # Output: [batch, 2048]
        features = self.inception(images)
        
        # Project to spatial feature map
        # [batch, 2048] -> [batch, 64 * encoder_dim]
        features = self.feature_projection(features)
        
        # Reshape to [batch, 64, encoder_dim] for attention
        batch_size = features.size(0)
        features = features.view(batch_size, 64, self.encoder_dim)
        
        return features


print("[OK] EncoderCNN defined")


# ==============================================================================
### CELL 8: Attention Mechanism
# ==============================================================================
# Bahdanau (additive) attention

class BahdanauAttention(nn.Module):
    """Bahdanau Attention Mechanism."""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super(BahdanauAttention, self).__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
    
    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out: [batch, 64, encoder_dim]
            decoder_hidden: [batch, decoder_dim]
        Returns:
            context: [batch, encoder_dim]
            alpha: [batch, 64] attention weights
        """
        encoder_proj = self.W_encoder(encoder_out)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)
        combined = torch.tanh(encoder_proj + decoder_proj)
        attention_scores = self.V(combined).squeeze(2)
        alpha = F.softmax(attention_scores, dim=1)
        context = (alpha.unsqueeze(2) * encoder_out).sum(dim=1)
        return context, alpha


print("[OK] BahdanauAttention defined")


# ==============================================================================
### CELL 9: Decoder Model (LSTM with Attention + Object Guidance)
# ==============================================================================
# LSTM decoder enhanced with object detection context

# Number of COCO object classes
NUM_OBJECT_CLASSES = 80
OBJECT_EMBED_DIM = 128

class DecoderWithAttention(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention + Object Detection Guidance.
    
    Enhanced to accept detected object labels for guided caption generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        encoder_dim: int = ENCODER_DIM,
        decoder_dim: int = DECODER_DIM,
        attention_dim: int = ATTENTION_DIM,
        dropout: float = DROPOUT,
        num_object_classes: int = NUM_OBJECT_CLASSES,
        object_embed_dim: int = OBJECT_EMBED_DIM
    ):
        super(DecoderWithAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # NEW: Object embedding for detected objects
        self.object_embedding = nn.Embedding(num_object_classes + 1, object_embed_dim)
        self.object_fc = nn.Linear(object_embed_dim, decoder_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.object_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, encoder_out, detected_objects=None):
        """Initialize LSTM states with optional object context."""
        mean_encoder = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        
        # Add object context to hidden state
        if detected_objects is not None and detected_objects.numel() > 0:
            obj_embed = self.object_embedding(detected_objects)
            obj_context = obj_embed.mean(dim=1)
            obj_hidden = self.object_fc(obj_context)
            h = h + obj_hidden
        
        return h, c
    
    def forward(self, encoder_out, captions, caption_lengths, detected_objects=None):
        """
        Forward pass with teacher forcing + optional object guidance.
        
        Args:
            encoder_out: [batch, 64, encoder_dim]
            captions: [batch, max_length]
            caption_lengths: [batch]
            detected_objects: [batch, max_objects] - detected COCO class indices (optional)
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        
        # Sort by caption length
        caption_lengths_sorted, sort_idx = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        captions = captions[sort_idx]
        
        if detected_objects is not None:
            detected_objects = detected_objects[sort_idx]
        
        decode_lengths = (caption_lengths_sorted - 1).tolist()
        max_decode_length = max(decode_lengths)
        
        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(encoder_out, detected_objects)
        
        predictions = torch.zeros(batch_size, max_decode_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_decode_length, num_pixels).to(device)
        
        for t in range(max_decode_length):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            context, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context
            
            lstm_input = torch.cat([embeddings[:batch_size_t, t], context], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            
            preds = self.fc(self.dropout(h))
            
            predictions[:batch_size_t, t] = preds
            alphas[:batch_size_t, t] = alpha
        
        return predictions, alphas, captions, decode_lengths


print("[OK] DecoderWithAttention with Object Guidance defined")


# ==============================================================================
### CELL 10: Initialize Models
# ==============================================================================
# Create encoder and decoder

print("\n" + "=" * 60)
print("INITIALIZING MODELS")
print("=" * 60)

encoder = EncoderCNN(encoder_dim=ENCODER_DIM, fine_tune=False)
encoder = encoder.to(device)
print(f"   Encoder: InceptionV3 backbone (frozen)")

decoder = DecoderWithAttention(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    encoder_dim=ENCODER_DIM,
    decoder_dim=DECODER_DIM,
    attention_dim=ATTENTION_DIM,
    dropout=DROPOUT
)
decoder = decoder.to(device)
print(f"   Decoder: LSTM with Bahdanau Attention")
print(f"   Vocabulary size: {len(vocab):,}")

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Optimizer
optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

# LR Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE
)

print(f"\n   Optimizer: Adam (lr={LEARNING_RATE})")
print(f"   Loss: CrossEntropyLoss (ignoring PAD)")


# ==============================================================================
### CELL 11: Training Functions
# ==============================================================================
# Train and validate for one epoch

def train_epoch(encoder, decoder, train_loader, criterion, optimizer, device, grad_clip=GRAD_CLIP):
    """Train for one epoch."""
    encoder.train()
    decoder.train()
    
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (images, captions, lengths) in enumerate(progress_bar):
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)
        
        encoder_out = encoder(images)
        predictions, alphas, caps_sorted, decode_lengths = decoder(encoder_out, captions, lengths)
        
        targets = caps_sorted[:, 1:]
        
        # Pack predictions and targets
        predictions_packed = torch.zeros(sum(decode_lengths), decoder.vocab_size).to(device)
        targets_packed = torch.zeros(sum(decode_lengths)).long().to(device)
        
        idx = 0
        for i, length in enumerate(decode_lengths):
            predictions_packed[idx:idx+length] = predictions[i, :length]
            targets_packed[idx:idx+length] = targets[i, :length]
            idx += length
        
        loss = criterion(predictions_packed, targets_packed)
        
        # Attention regularization
        alpha_c = 1.0
        alphas_sum = sum([alphas[i, :dl].sum(dim=0) for i, dl in enumerate(decode_lengths)]) / len(decode_lengths)
        attention_reg = alpha_c * ((1.0 - alphas_sum) ** 2).mean()
        loss = loss + attention_reg
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate_epoch(encoder, decoder, val_loader, criterion, device):
    """Validate the model."""
    encoder.eval()
    decoder.eval()
    
    total_loss = 0.0
    
    with torch.no_grad():
        for images, captions, lengths in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)
            
            encoder_out = encoder(images)
            predictions, alphas, caps_sorted, decode_lengths = decoder(encoder_out, captions, lengths)
            
            targets = caps_sorted[:, 1:]
            
            predictions_packed = torch.zeros(sum(decode_lengths), decoder.vocab_size).to(device)
            targets_packed = torch.zeros(sum(decode_lengths)).long().to(device)
            
            idx = 0
            for i, length in enumerate(decode_lengths):
                predictions_packed[idx:idx+length] = predictions[i, :length]
                targets_packed[idx:idx+length] = targets[i, :length]
                idx += length
            
            loss = criterion(predictions_packed, targets_packed)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


print("[OK] Training functions defined")


# ==============================================================================
### CELL 12: Training Loop
# ==============================================================================
# Main training loop with early stopping

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_val_loss = float('inf')
epochs_no_improve = 0
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    print(f"\n==== Epoch {epoch}/{EPOCHS}")
    print("-" * 40)
    
    train_loss = train_epoch(encoder, decoder, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    val_loss = validate_epoch(encoder, decoder, val_loader, criterion, device)
    val_losses.append(val_loss)
    
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss:   {val_loss:.4f}")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        print(f"   [OK] New best model! Saving checkpoint...")
        torch.save(encoder.state_dict(), str(ENCODER_PATH))
        torch.save(decoder.state_dict(), str(DECODER_PATH))
    else:
        epochs_no_improve += 1
        print(f"   [!!] No improvement for {epochs_no_improve} epoch(s)")
    
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\n[STOP] Early stopping triggered after {epoch} epochs")
        break

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"   Best validation loss: {best_val_loss:.4f}")
print(f"   Models saved to: {MODELS_DIR}")


# ==============================================================================
### CELL 13: Plot Training Curves
# ==============================================================================
# Visualize training progress

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(MODELS_DIR / 'training_curves.png'), dpi=150)
plt.show()
print(f"   Saved to: {MODELS_DIR / 'training_curves.png'}")


# ==============================================================================
### CELL 14: Load Best Model and Generate Captions
# ==============================================================================
# Test caption generation

print("\n" + "=" * 60)
print("TESTING CAPTION GENERATION")
print("=" * 60)

# Load best checkpoint
encoder.load_state_dict(torch.load(str(ENCODER_PATH), map_location=device))
decoder.load_state_dict(torch.load(str(DECODER_PATH), map_location=device))
encoder.eval()
decoder.eval()

def generate_caption_greedy(encoder, decoder, image, vocab, max_length=50):
    """Generate caption using greedy decoding."""
    with torch.no_grad():
        encoder_out = encoder(image)
        h, c = decoder.init_hidden_state(encoder_out)
        
        word_idx = vocab.word2idx[vocab.START_TOKEN]
        words = []
        
        for _ in range(max_length):
            word_tensor = torch.LongTensor([word_idx]).to(image.device)
            embedding = decoder.embedding(word_tensor)
            
            context, alpha = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context
            
            lstm_input = torch.cat([embedding, context], dim=1)
            h, c = decoder.decode_step(lstm_input, (h, c))
            
            scores = decoder.fc(h)
            word_idx = scores.argmax(dim=1).item()
            
            if word_idx == vocab.word2idx[vocab.END_TOKEN]:
                break
            
            words.append(vocab.idx2word.get(word_idx, vocab.UNK_TOKEN))
        
        return words

# Test on a few images
transform = get_image_transforms(IMAGE_SIZE, is_training=False)

for idx, img_name in enumerate(test_images[:5]):
    img_path = IMAGES_DIR / img_name
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    ground_truth = image_to_captions[img_name][0]
    predicted_words = generate_caption_greedy(encoder, decoder, image_tensor, vocab)
    predicted = ' '.join(predicted_words)
    
    print(f"\n**** Image: {img_name}")
    print(f"   GT:   {ground_truth}")
    print(f"   Pred: {predicted}")

print("\n[OK] Caption generation test complete!")


# ==============================================================================
### CELL 15: Overfitting/Underfitting Analysis
# ==============================================================================
# Analyze training behavior to detect overfitting or underfitting

print("\n" + "=" * 60)
print("OVERFITTING/UNDERFITTING ANALYSIS")
print("=" * 60)

def analyze_training(train_losses, val_losses):
    """
    Analyze training curves to detect overfitting or underfitting.
    
    Criteria:
    - Overfitting: Val loss increases while train loss decreases
    - Underfitting: Both losses are high and not decreasing
    - Good fit: Both losses decrease and converge
    """
    if len(train_losses) < 2:
        return "Not enough epochs to analyze"
    
    # Calculate statistics
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    min_val = min(val_losses)
    best_epoch = val_losses.index(min_val) + 1
    
    # Calculate loss gap (overfitting indicator)
    loss_gap = final_val - final_train
    gap_ratio = loss_gap / final_train if final_train > 0 else 0
    
    # Detect trend in validation loss
    late_val_losses = val_losses[-3:] if len(val_losses) >= 3 else val_losses
    val_trend = late_val_losses[-1] - late_val_losses[0]
    
    # Diagnosis
    diagnosis = []
    
    print(f"\n==== Training Statistics:")
    print(f"   Final Train Loss: {final_train:.4f}")
    print(f"   Final Val Loss:   {final_val:.4f}")
    print(f"   Loss Gap:         {loss_gap:.4f} ({gap_ratio*100:.1f}%)")
    print(f"   Best Epoch:       {best_epoch}")
    print(f"   Total Epochs:     {len(train_losses)}")
    
    # Overfitting detection
    if gap_ratio > 0.2:  # >20% gap
        diagnosis.append("🔴 OVERFITTING DETECTED")
        diagnosis.append("   - Large gap between train and val loss")
        diagnosis.append("   - Model memorizing training data")
        diagnosis.append("   Recommendations:")
        diagnosis.append("   • Increase dropout (currently {:.1f})".format(DROPOUT))
        diagnosis.append("   • Add more data augmentation")
        diagnosis.append("   • Reduce model complexity")
        diagnosis.append("   • Use early stopping (already enabled)")
    elif gap_ratio > 0.1:  # 10-20% gap
        diagnosis.append("🟡 SLIGHT OVERFITTING")
        diagnosis.append("   - Moderate gap between train and val loss")
        diagnosis.append("   - Consider more regularization")
    else:
        diagnosis.append("🟢 NO OVERFITTING")
        diagnosis.append("   - Train and val losses are close")
    
    # Underfitting detection
    if final_train > 3.0 and final_val > 3.0:
        diagnosis.append("\n🔴 UNDERFITTING DETECTED")
        diagnosis.append("   - Both losses are still high")
        diagnosis.append("   Recommendations:")
        diagnosis.append("   • Train for more epochs")
        diagnosis.append("   • Increase model capacity")
        diagnosis.append("   • Reduce regularization")
        diagnosis.append("   • Check learning rate")
    
    # Convergence check
    if val_trend > 0.1:
        diagnosis.append("\n🟡 VALIDATION LOSS INCREASING")
        diagnosis.append("   - Model may be starting to overfit")
        diagnosis.append("   - Best model was at epoch {}".format(best_epoch))
    elif val_trend < -0.05:
        diagnosis.append("\n🟢 STILL IMPROVING")
        diagnosis.append("   - Validation loss still decreasing")
        diagnosis.append("   - Consider training longer")
    else:
        diagnosis.append("\n🟢 CONVERGED")
        diagnosis.append("   - Model has reached stable performance")
    
    return "\n".join(diagnosis)

# Run analysis
analysis_result = analyze_training(train_losses, val_losses)
print(analysis_result)

# Plot with overfitting indicators
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curves
ax1 = axes[0]
epochs = range(1, len(train_losses) + 1)
ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
ax1.fill_between(epochs, train_losses, val_losses, alpha=0.3, color='orange', label='Gap (Overfitting)')
ax1.axhline(y=min(val_losses), color='g', linestyle='--', alpha=0.5, label=f'Best Val: {min(val_losses):.4f}')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss\n(Gap indicates overfitting)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss gap over epochs
ax2 = axes[1]
loss_gaps = [v - t for t, v in zip(train_losses, val_losses)]
colors = ['green' if g < 0.1 else 'orange' if g < 0.2 else 'red' for g in loss_gaps]
ax2.bar(epochs, loss_gaps, color=colors, alpha=0.7)
ax2.axhline(y=0.1, color='orange', linestyle='--', label='Slight overfit threshold')
ax2.axhline(y=0.2, color='red', linestyle='--', label='Overfit threshold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Val Loss - Train Loss')
ax2.set_title('Overfitting Gap per Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(MODELS_DIR / 'overfitting_analysis.png'), dpi=150)
plt.show()
print(f"\n   Saved to: {MODELS_DIR / 'overfitting_analysis.png'}")


# ==============================================================================
### CELL 16: BLEU Score Evaluation
# ==============================================================================
# Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores on test set

print("\n" + "=" * 60)
print("BLEU SCORE EVALUATION")
print("=" * 60)

from collections import Counter
import math

def ngrams(sequence, n):
    """Generate n-grams from a sequence."""
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]

def modified_precision(candidate, references, n):
    """Calculate modified precision for n-grams."""
    candidate_ngrams = ngrams(candidate, n)
    if len(candidate_ngrams) == 0:
        return 0.0
    
    candidate_counts = Counter(candidate_ngrams)
    max_ref_counts = Counter()
    
    for ref in references:
        ref_ngrams = ngrams(ref, n)
        ref_counts = Counter(ref_ngrams)
        for ngram in candidate_counts:
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), ref_counts.get(ngram, 0))
    
    clipped_counts = {ngram: min(count, max_ref_counts.get(ngram, 0)) 
                      for ngram, count in candidate_counts.items()}
    
    return sum(clipped_counts.values()) / len(candidate_ngrams)

def brevity_penalty(candidate, references):
    """Calculate brevity penalty."""
    c = len(candidate)
    ref_lengths = [len(r) for r in references]
    r = min(ref_lengths, key=lambda x: abs(x - c))
    
    if c >= r:
        return 1.0
    else:
        return math.exp(1 - r / c) if c > 0 else 0.0

def bleu_score(candidate, references, max_n=4):
    """
    Calculate BLEU score.
    
    Returns:
        dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    scores = {}
    
    for n in range(1, max_n + 1):
        p_n = modified_precision(candidate, references, n)
        scores[f'BLEU-{n}'] = p_n
    
    # Combined BLEU (geometric mean)
    bp = brevity_penalty(candidate, references)
    
    # Calculate log of geometric mean
    log_precisions = []
    for n in range(1, max_n + 1):
        p_n = scores[f'BLEU-{n}']
        if p_n > 0:
            log_precisions.append(math.log(p_n))
    
    if len(log_precisions) == max_n:
        geometric_mean = math.exp(sum(log_precisions) / max_n)
        scores['BLEU'] = bp * geometric_mean
    else:
        scores['BLEU'] = 0.0
    
    return scores

# Evaluate on test set
print("\n==== Evaluating on test set...")

all_bleu_scores = {'BLEU-1': [], 'BLEU-2': [], 'BLEU-3': [], 'BLEU-4': [], 'BLEU': []}

transform = get_image_transforms(IMAGE_SIZE, is_training=False)

for img_name in tqdm(test_images[:100], desc="Calculating BLEU"):  # Limit to 100 for speed
    img_path = IMAGES_DIR / img_name
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get references (ground truth captions)
    references = [cap.split() for cap in image_to_captions[img_name]]
    
    # Generate caption
    predicted_words = generate_caption_greedy(encoder, decoder, image_tensor, vocab)
    
    # Calculate BLEU
    scores = bleu_score(predicted_words, references)
    for key in all_bleu_scores:
        all_bleu_scores[key].append(scores[key])

# Calculate averages
print("\n==== BLEU Score Results (Test Set):")
print("-" * 40)
avg_scores = {}
for metric, values in all_bleu_scores.items():
    avg = sum(values) / len(values)
    avg_scores[metric] = avg
    print(f"   {metric}: {avg:.4f} ({avg*100:.2f}%)")

# Interpretation
print("\n📖 BLEU Score Interpretation:")
print("   BLEU-1: Word-level accuracy (unigrams)")
print("   BLEU-2: 2-word phrase accuracy")
print("   BLEU-3: 3-word phrase accuracy")
print("   BLEU-4: 4-word phrase accuracy (most important)")
print("   BLEU:   Combined score with brevity penalty")

bleu4 = avg_scores['BLEU-4']
if bleu4 > 0.30:
    print("\n   🟢 BLEU-4 > 0.30: Excellent performance!")
elif bleu4 > 0.20:
    print("\n   🟢 BLEU-4 > 0.20: Good performance")
elif bleu4 > 0.10:
    print("\n   🟡 BLEU-4 > 0.10: Acceptable performance")
else:
    print("\n   🔴 BLEU-4 < 0.10: Needs improvement")

# Plot BLEU scores
plt.figure(figsize=(10, 5))
metrics = list(avg_scores.keys())
values = list(avg_scores.values())
colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
bars = plt.bar(metrics, values, color=colors, alpha=0.8)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('BLEU Score Breakdown')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(str(MODELS_DIR / 'bleu_scores.png'), dpi=150)
plt.show()
print(f"\n   Saved to: {MODELS_DIR / 'bleu_scores.png'}")


# ==============================================================================
### CELL 17: Attention Visualization
# ==============================================================================
# Visualize attention weights over image regions

print("\n" + "=" * 60)
print("ATTENTION VISUALIZATION")
print("=" * 60)

def visualize_attention(image_path, encoder, decoder, vocab, transform, device):
    """
    Visualize attention weights for each generated word.
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    image = transform(original_image).unsqueeze(0).to(device)
    
    # Get encoder features
    with torch.no_grad():
        encoder_out = encoder(image)  # [1, 64, 2048]
        h, c = decoder.init_hidden_state(encoder_out)
        
        word_idx = vocab.word2idx[vocab.START_TOKEN]
        words = []
        attentions = []
        
        for _ in range(30):
            word_tensor = torch.LongTensor([word_idx]).to(device)
            embedding = decoder.embedding(word_tensor)
            
            context, alpha = decoder.attention(encoder_out, h)
            attentions.append(alpha.cpu().numpy().reshape(8, 8))
            
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context
            
            lstm_input = torch.cat([embedding, context], dim=1)
            h, c = decoder.decode_step(lstm_input, (h, c))
            
            scores = decoder.fc(h)
            word_idx = scores.argmax(dim=1).item()
            
            if word_idx == vocab.word2idx[vocab.END_TOKEN]:
                break
            
            words.append(vocab.idx2word.get(word_idx, vocab.UNK_TOKEN))
    
    return words, attentions, original_image

# Visualize a few examples
print("\n🎨 Generating attention visualizations...")

for idx, img_name in enumerate(test_images[:3]):
    img_path = IMAGES_DIR / img_name
    words, attentions, orig_img = visualize_attention(
        img_path, encoder, decoder, vocab, transform, device
    )
    
    # Plot
    n_words = min(len(words), 8)  # Show max 8 words
    fig, axes = plt.subplots(2, n_words, figsize=(2.5 * n_words, 6))
    
    for i in range(n_words):
        # Original image
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        axes[0, i].set_title(words[i], fontsize=10)
        
        # Attention heatmap
        attention_map = attentions[i]
        axes[1, i].imshow(orig_img)
        attention_resized = np.array(Image.fromarray(attention_map).resize(orig_img.size, Image.BILINEAR))
        axes[1, i].imshow(attention_resized, alpha=0.6, cmap='jet')
        axes[1, i].axis('off')
    
    plt.suptitle(f"Caption: {' '.join(words)}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / f'attention_viz_{idx+1}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n   **** Image: {img_name}")
    print(f"   ==== Caption: {' '.join(words)}")

print(f"\n   Saved to: {MODELS_DIR}/attention_viz_*.png")


# ==============================================================================
### CELL 18: Visual Sample Predictions
# ==============================================================================
# Show image-caption pairs side by side

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS WITH IMAGES")
print("=" * 60)

def show_predictions(test_images, encoder, decoder, vocab, image_to_captions, 
                     images_dir, transform, device, n_samples=6):
    """
    Display a grid of sample predictions with ground truth.
    """
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx, img_name in enumerate(test_images[:n_samples]):
        ax = axes[idx]
        
        # Load and display image
        img_path = images_dir / img_name
        image = Image.open(img_path).convert('RGB')
        ax.imshow(image)
        ax.axis('off')
        
        # Generate caption
        image_tensor = transform(image).unsqueeze(0).to(device)
        predicted_words = generate_caption_greedy(encoder, decoder, image_tensor, vocab)
        predicted = ' '.join(predicted_words)
        
        # Get ground truth
        gt = image_to_captions[img_name][0]
        
        # Add title
        ax.set_title(f"Pred: {predicted[:50]}...\nGT: {gt[:50]}...", 
                     fontsize=9, wrap=True)
    
    # Hide empty subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / 'sample_predictions.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n   Saved to: {MODELS_DIR / 'sample_predictions.png'}")

show_predictions(
    test_images, encoder, decoder, vocab, image_to_captions,
    IMAGES_DIR, transform, device, n_samples=6
)


# ==============================================================================
### CELL 19: Final Summary
# ==============================================================================
# Print comprehensive training summary

print("\n" + "=" * 60)
print("==== TRAINING SUMMARY REPORT")
print("=" * 60)

print(f"""
==== Configuration:
   • Encoder: InceptionV3 (frozen)
   • Decoder: LSTM with Bahdanau Attention
   • Embedding dim: {EMBEDDING_DIM}
   • Hidden dim: {DECODER_DIM}
   • Attention dim: {ATTENTION_DIM}
   • Dropout: {DROPOUT}

📈 Training Results:
   • Total epochs: {len(train_losses)}
   • Best epoch: {val_losses.index(min(val_losses)) + 1}
   • Final train loss: {train_losses[-1]:.4f}
   • Final val loss: {val_losses[-1]:.4f}
   • Best val loss: {min(val_losses):.4f}

==== BLEU Scores (Test Set):
   • BLEU-1: {avg_scores['BLEU-1']:.4f}
   • BLEU-2: {avg_scores['BLEU-2']:.4f}
   • BLEU-3: {avg_scores['BLEU-3']:.4f}
   • BLEU-4: {avg_scores['BLEU-4']:.4f}
   • Combined BLEU: {avg_scores['BLEU']:.4f}

>>> Saved Files:
   • Encoder: {ENCODER_PATH}
   • Decoder: {DECODER_PATH}
   • Vocabulary: {VOCAB_PATH}
   • Training curves: {MODELS_DIR / 'training_curves.png'}
   • Overfitting analysis: {MODELS_DIR / 'overfitting_analysis.png'}
   • BLEU scores: {MODELS_DIR / 'bleu_scores.png'}
   • Attention visualizations: {MODELS_DIR / 'attention_viz_*.png'}
   • Sample predictions: {MODELS_DIR / 'sample_predictions.png'}
""")

print("[OK] All cells complete! Notebook ready for presentation.")
