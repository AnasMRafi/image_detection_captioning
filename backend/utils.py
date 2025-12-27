"""
Utility functions for Image Captioning Model.

This module contains helper functions for:
- Parsing the Flickr8k captions dataset
- Building and managing vocabulary
- Image preprocessing
- Beam search decoding for inference
- Visualization utilities

All functions include detailed docstrings with tensor shapes documented.
"""

import os
import re
import pickle
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


# ==============================================================================
# DATA PARSING UTILITIES
# ==============================================================================

def parse_captions(captions_file: str, images_dir: str) -> Dict[str, List[str]]:
    """
    Parse the Flickr8k captions.txt file into a dictionary mapping images to captions.
    
    The Flickr8k dataset has a specific format:
    - First row is a header: "image,caption"
    - Each subsequent row: "image_filename.jpg,caption text here"
    - Each image has 5 different captions from different annotators
    - Captions may contain commas, so we split only on the FIRST comma
    
    Args:
        captions_file: Path to captions.txt file
            Format: CSV with header row, columns are (image, caption)
        images_dir: Path to directory containing image files
            Used to verify that referenced images actually exist
    
    Returns:
        Dictionary mapping image filenames to lists of caption strings
        Example: {"dog.jpg": ["A dog runs", "Dog playing", ...]}
    
    Raises:
        FileNotFoundError: If captions_file doesn't exist
    
    Example:
        >>> captions = parse_captions("captions.txt", "Images/")
        >>> print(f"Loaded {len(captions)} unique images")
        >>> print(f"First image has {len(list(captions.values())[0])} captions")
    """
    image_to_captions: Dict[str, List[str]] = {}
    images_dir = Path(images_dir)
    
    # Track statistics for logging
    total_pairs = 0
    missing_images = set()
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip the header row (first line: "image,caption")
    # This is critical - the header would otherwise be treated as a data row
    for line_num, line in enumerate(lines[1:], start=2):  # start=2 for 1-indexed with header
        line = line.strip()
        if not line:
            continue
        
        # Split only on the FIRST comma to handle captions containing commas
        # Example: "image.jpg,A man, woman, and child walking"
        # Should split to: ["image.jpg", "A man, woman, and child walking"]
        parts = line.split(',', maxsplit=1)
        
        if len(parts) != 2:
            print(f"⚠️  Line {line_num}: Invalid format, skipping: {line[:50]}...")
            continue
        
        image_name, caption = parts
        image_name = image_name.strip()
        caption = caption.strip()
        
        # Verify the image file exists before adding
        image_path = images_dir / image_name
        if not image_path.exists():
            if image_name not in missing_images:
                missing_images.add(image_name)
            continue
        
        # Initialize list for new images, append caption
        if image_name not in image_to_captions:
            image_to_captions[image_name] = []
        
        image_to_captions[image_name].append(caption)
        total_pairs += 1
    
    # Log parsing statistics
    print(f"📊 Caption Parsing Statistics:")
    print(f"   Total image-caption pairs: {total_pairs:,}")
    print(f"   Unique images: {len(image_to_captions):,}")
    print(f"   Average captions per image: {total_pairs / len(image_to_captions):.1f}")
    if missing_images:
        print(f"   ⚠️  Missing images (skipped): {len(missing_images)}")
    
    return image_to_captions


def clean_caption(caption: str) -> str:
    """
    Clean and normalize a caption string for vocabulary building.
    
    Preprocessing steps:
    1. Convert to lowercase for case-insensitive matching
    2. Remove punctuation except apostrophes (for contractions like "don't")
    3. Remove extra whitespace
    4. Strip leading/trailing whitespace
    
    Args:
        caption: Raw caption string from dataset
    
    Returns:
        Cleaned caption string with normalized formatting
    
    Example:
        >>> clean_caption("A Dog's running!! Fast.")
        "a dog's running fast"
    """
    # Convert to lowercase
    caption = caption.lower()
    
    # Remove punctuation except apostrophes (keep contractions like "don't")
    # This regex matches any character that's not alphanumeric, whitespace, or apostrophe
    caption = re.sub(r"[^\w\s']", '', caption)
    
    # Replace multiple spaces with single space
    caption = re.sub(r'\s+', ' ', caption)
    
    # Strip leading/trailing whitespace
    caption = caption.strip()
    
    return caption


# ==============================================================================
# VOCABULARY MANAGEMENT
# ==============================================================================

class Vocabulary:
    """
    Vocabulary class for mapping between words and integer indices.
    
    This class handles:
    - Building vocabulary from a corpus with frequency thresholding
    - Converting words to indices and back
    - Special tokens for sequence modeling (PAD, START, END, UNK)
    
    Attributes:
        word2idx: Dictionary mapping words to integer indices
        idx2word: Dictionary mapping integer indices to words
        word_freq: Counter of word frequencies from training data
    
    Special Tokens:
        <PAD> (index 0): Padding for batching variable-length sequences
        <START> (index 1): Beginning of caption marker
        <END> (index 2): End of caption marker
        <UNK> (index 3): Unknown/out-of-vocabulary words
    """
    
    # Class constants for special tokens
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self):
        """Initialize vocabulary with special tokens."""
        # Initialize mappings with special tokens at fixed indices
        # Order matters! These indices are used throughout the model
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,    # Must be 0 for efficient masking
            self.START_TOKEN: 1,
            self.END_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.word_freq: Counter = Counter()
    
    def build_vocab(self, captions: List[str], min_freq: int = 5, max_size: int = 10000) -> None:
        """
        Build vocabulary from a list of captions.
        
        Words appearing less than min_freq times are excluded and will be
        mapped to <UNK> during encoding. This reduces vocabulary size and
        helps the model generalize better.
        
        Args:
            captions: List of caption strings (should be cleaned/normalized)
            min_freq: Minimum word frequency to include in vocabulary
                     Words below this threshold become <UNK>
            max_size: Maximum vocabulary size (including special tokens)
                     Most frequent words are kept if limit exceeded
        
        Example:
            >>> vocab = Vocabulary()
            >>> vocab.build_vocab(["a dog runs", "the cat sits"], min_freq=1)
            >>> print(f"Vocabulary size: {len(vocab)}")
        """
        # Count all word frequencies across all captions
        for caption in captions:
            cleaned = clean_caption(caption)
            words = cleaned.split()
            self.word_freq.update(words)
        
        # Filter words by minimum frequency and get most common up to max_size
        # Subtract 4 for the special tokens already in vocabulary
        filtered_words = [
            word for word, freq in self.word_freq.most_common(max_size - 4)
            if freq >= min_freq
        ]
        
        # Add filtered words to vocabulary
        for word in filtered_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        # Log vocabulary statistics
        total_words = sum(self.word_freq.values())
        covered_words = sum(
            self.word_freq[word] for word in self.word2idx 
            if word not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        )
        coverage = covered_words / total_words * 100 if total_words > 0 else 0
        
        print(f"📚 Vocabulary Statistics:")
        print(f"   Total unique words in corpus: {len(self.word_freq):,}")
        print(f"   Words in vocabulary: {len(self.word2idx):,}")
        print(f"   Words filtered (freq < {min_freq}): {len(self.word_freq) - len(self.word2idx) + 4:,}")
        print(f"   Coverage: {coverage:.1f}% of word occurrences")
    
    def encode(self, caption: str, max_length: int = 50) -> List[int]:
        """
        Convert a caption string to a list of word indices.
        
        The caption is:
        1. Cleaned/normalized
        2. Tokenized by whitespace
        3. Wrapped with START and END tokens
        4. Truncated/padded to max_length
        
        Args:
            caption: Raw caption string
            max_length: Maximum sequence length (including START/END)
                       Longer sequences are truncated, shorter are NOT padded
        
        Returns:
            List of integer indices representing the caption
            Format: [START_idx, word1_idx, word2_idx, ..., END_idx]
        
        Example:
            >>> vocab.encode("a dog runs")
            [1, 45, 123, 89, 2]  # [START, a, dog, runs, END]
        """
        cleaned = clean_caption(caption)
        words = cleaned.split()
        
        # Start with START token
        indices = [self.word2idx[self.START_TOKEN]]
        
        # Add word indices (use UNK for unknown words)
        for word in words[:max_length - 2]:  # Reserve space for START and END
            idx = self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
            indices.append(idx)
        
        # Add END token
        indices.append(self.word2idx[self.END_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Convert a list of word indices back to a caption string.
        
        Args:
            indices: List of integer indices from model output
            skip_special: If True, omit PAD, START, END tokens from output
                         UNK tokens are included as "<UNK>"
        
        Returns:
            Decoded caption string with words joined by spaces
        
        Example:
            >>> vocab.decode([1, 45, 123, 89, 2])
            "a dog runs"  # with skip_special=True
        """
        special_tokens = {
            self.word2idx[self.PAD_TOKEN],
            self.word2idx[self.START_TOKEN],
            self.word2idx[self.END_TOKEN],
        }
        
        words = []
        for idx in indices:
            # Stop at END token
            if idx == self.word2idx[self.END_TOKEN]:
                break
            
            # Skip special tokens if requested
            if skip_special and idx in special_tokens:
                continue
            
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            words.append(word)
        
        return ' '.join(words)
    
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.word2idx)
    
    def save(self, path: str) -> None:
        """
        Save vocabulary to a pickle file.
        
        Args:
            path: File path for the pickle file
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
            }, f)
        print(f"💾 Vocabulary saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """
        Load vocabulary from a pickle file.
        
        Args:
            path: File path to the pickle file
        
        Returns:
            Vocabulary instance with loaded mappings
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        # word_freq is optional (older vocab files may not have it)
        vocab.word_freq = Counter(data.get('word_freq', {}))
        
        print(f"📂 Vocabulary loaded from: {path}")
        print(f"   Size: {len(vocab):,} words")
        
        return vocab


# ==============================================================================
# IMAGE PREPROCESSING
# ==============================================================================

def get_image_transforms(image_size: int = 299, is_training: bool = True) -> transforms.Compose:
    """
    Get image transformation pipeline for preprocessing.
    
    Training transforms include data augmentation (random crop, flip, color jitter)
    to improve model generalization. Validation/test transforms only resize and
    normalize without augmentation.
    
    Args:
        image_size: Target image size (height and width)
                   InceptionV3 expects 299x299, ResNet expects 224x224
        is_training: If True, include data augmentation transforms
    
    Returns:
        torchvision.transforms.Compose object for image preprocessing
    
    Output Tensor Shape:
        [3, image_size, image_size] - CHW format, normalized to ImageNet stats
    
    Example:
        >>> train_transforms = get_image_transforms(299, is_training=True)
        >>> image = Image.open("dog.jpg")
        >>> tensor = train_transforms(image)
        >>> print(tensor.shape)  # torch.Size([3, 299, 299])
    """
    # ImageNet normalization statistics (required for pretrained models)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    if is_training:
        # Training: apply data augmentation for better generalization
        return transforms.Compose([
            # Resize slightly larger for random crop
            transforms.Resize((image_size + 30, image_size + 30)),
            # Random crop to target size (spatial augmentation)
            transforms.RandomCrop(image_size),
            # Horizontal flip with 50% probability
            transforms.RandomHorizontalFlip(p=0.5),
            # Color jitter for lighting robustness
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # Convert PIL Image to tensor (scales to [0, 1])
            transforms.ToTensor(),
            # Normalize using ImageNet statistics
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        # Validation/Test: deterministic preprocessing, no augmentation
        return transforms.Compose([
            # Resize to target size
            transforms.Resize((image_size, image_size)),
            # Convert to tensor
            transforms.ToTensor(),
            # Normalize
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


def load_image(image_path: str, transform: Optional[transforms.Compose] = None) -> torch.Tensor:
    """
    Load and preprocess an image from disk.
    
    Args:
        image_path: Path to the image file
        transform: Optional transform pipeline. If None, uses default eval transforms
    
    Returns:
        Preprocessed image tensor
        Shape: [3, H, W] or [1, 3, H, W] if batched
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        PIL.UnidentifiedImageError: If file is not a valid image
    """
    # Open and convert to RGB (handles grayscale and RGBA images)
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    if transform is None:
        transform = get_image_transforms(is_training=False)
    
    return transform(image)


# ==============================================================================
# BEAM SEARCH DECODING
# ==============================================================================

def beam_search(
    encoder_out: torch.Tensor,
    decoder: 'DecoderWithAttention',
    vocab: Vocabulary,
    beam_size: int = 3,
    max_length: int = 50,
    device: torch.device = None
) -> Tuple[List[int], float, List[torch.Tensor]]:
    """
    Generate a caption using beam search decoding.
    
    Beam search maintains 'beam_size' candidate sequences at each step,
    expanding each with the most probable next words. This explores more
    of the probability space than greedy decoding and typically produces
    better captions.
    
    Algorithm:
    1. Start with beam_size copies of the START token
    2. For each position:
       a. Expand each candidate with top-k next words
       b. Score all beam_size * vocab_size possibilities
       c. Keep top beam_size candidates by total log probability
    3. Return the highest-scoring complete sequence
    
    Args:
        encoder_out: Encoded image features from CNN encoder
                    Shape: [1, num_pixels (64), encoder_dim (2048)]
        decoder: Trained decoder model with attention
        vocab: Vocabulary instance for token mappings
        beam_size: Number of candidate sequences to maintain
                  Higher = better quality but slower
        max_length: Maximum caption length to generate
        device: Computation device (CPU/CUDA/MPS)
    
    Returns:
        Tuple of:
        - best_sequence: List of word indices for best caption
        - best_score: Log probability score of best caption
        - attention_weights: List of attention tensors for visualization
                            Each tensor has shape [1, num_pixels]
    
    Example:
        >>> caption_indices, score, attn = beam_search(features, decoder, vocab)
        >>> caption = vocab.decode(caption_indices)
        >>> print(f"Caption: {caption} (score: {score:.3f})")
    """
    if device is None:
        device = encoder_out.device
    
    # Get dimensions
    k = beam_size
    vocab_size = len(vocab)
    
    # Tensors to store top k sub-sequences and their scores
    # Shape: [k, current_length]
    seqs = torch.LongTensor([[vocab.word2idx[vocab.START_TOKEN]]]).to(device)
    seqs = seqs.expand(k, -1)  # [k, 1]
    
    # Cumulative log probabilities for each sequence
    # Shape: [k]
    top_k_scores = torch.zeros(k).to(device)
    
    # Store attention weights for visualization
    all_attention_weights = []
    
    # Lists to store completed sequences
    complete_seqs = []
    complete_seqs_scores = []
    complete_seqs_attention = []
    
    # Expand encoder output to beam size
    # Shape: [k, num_pixels, encoder_dim]
    encoder_out = encoder_out.expand(k, -1, -1)
    
    # Initialize decoder hidden state
    # mean pool encoder output for initial hidden/cell state
    h, c = decoder.init_hidden_state(encoder_out)
    
    # Decode step by step
    step = 1
    while step <= max_length:
        # Get embeddings for the last word in each sequence
        # Shape: [k, embedding_dim]
        prev_words = seqs[:, -1]
        embeddings = decoder.embedding(prev_words)
        
        # Attention-weighted encoding
        # Shape: attention_weighted_encoding [k, encoder_dim], alpha [k, num_pixels]
        attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)
        all_attention_weights.append(alpha)
        
        # Gate the attention
        gate = decoder.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        
        # Decode one step
        h, c = decoder.decode_step(
            torch.cat([embeddings, attention_weighted_encoding], dim=1),
            (h, c)
        )
        
        # Get vocabulary scores
        # Shape: [k, vocab_size]
        scores = decoder.fc(decoder.dropout(h))
        scores = F.log_softmax(scores, dim=1)
        
        # Add previous scores
        # Shape: [k, vocab_size]
        scores = top_k_scores.unsqueeze(1) + scores
        
        if step == 1:
            # For the first step, all beams have the same prefix (START)
            # So only consider top k words from the first beam
            top_k_scores, top_k_words = scores[0].topk(k, dim=0)
            prev_beam_idx = torch.zeros(k, dtype=torch.long, device=device)
        else:
            # Reshape to [k * vocab_size] and get top k
            scores = scores.view(-1)
            top_k_scores, top_k_idx = scores.topk(k, dim=0)
            
            # Which beam did these top k come from?
            prev_beam_idx = top_k_idx // vocab_size
            # What are the next words?
            top_k_words = top_k_idx % vocab_size
        
        # Update sequences
        seqs = torch.cat([seqs[prev_beam_idx], top_k_words.unsqueeze(1)], dim=1)
        
        # Update encoder output and hidden states for continuing beams
        encoder_out = encoder_out[prev_beam_idx]
        h = h[prev_beam_idx]
        c = c[prev_beam_idx]
        
        # Check for completed sequences (END token)
        incomplete_idx = []
        for idx in range(k):
            if top_k_words[idx] == vocab.word2idx[vocab.END_TOKEN]:
                complete_seqs.append(seqs[idx].tolist())
                complete_seqs_scores.append(top_k_scores[idx].item())
                complete_seqs_attention.append([aw[idx] for aw in all_attention_weights])
            else:
                incomplete_idx.append(idx)
        
        # If all sequences are complete, stop
        if len(incomplete_idx) == 0:
            break
        
        # Continue with incomplete sequences
        if len(incomplete_idx) < k:
            incomplete_idx = torch.LongTensor(incomplete_idx).to(device)
            seqs = seqs[incomplete_idx]
            encoder_out = encoder_out[incomplete_idx]
            h = h[incomplete_idx]
            c = c[incomplete_idx]
            top_k_scores = top_k_scores[incomplete_idx]
            k = len(incomplete_idx)
        
        step += 1
    
    # If no complete sequences, use the best incomplete one
    if len(complete_seqs) == 0:
        complete_seqs = [seqs[0].tolist()]
        complete_seqs_scores = [top_k_scores[0].item()]
        complete_seqs_attention = [all_attention_weights]
    
    # Return the best complete sequence
    best_idx = np.argmax(complete_seqs_scores)
    best_seq = complete_seqs[best_idx]
    best_score = complete_seqs_scores[best_idx]
    best_attention = complete_seqs_attention[best_idx]
    
    return best_seq, best_score, best_attention


# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def visualize_attention(
    image_path: str,
    caption_words: List[str],
    attention_weights: List[torch.Tensor],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights as heatmaps overlaid on the image.
    
    Creates a grid showing the original image and attention heatmaps
    for each generated word, helping understand what the model
    "looks at" when generating each word.
    
    Args:
        image_path: Path to the original image
        caption_words: List of generated words (from vocab.decode)
        attention_weights: List of attention tensors from beam search
                          Each tensor shape: [num_pixels] (e.g., 64 for 8x8)
        save_path: Optional path to save the visualization
    
    Example:
        >>> visualize_attention(
        ...     "dog.jpg",
        ...     ["a", "dog", "runs"],
        ...     attention_list,
        ...     save_path="attention_viz.png"
        ... )
    """
    import matplotlib.pyplot as plt
    
    # Load and prepare image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    
    # Number of words to visualize (limit for readability)
    num_words = min(len(caption_words), len(attention_weights))
    
    # Create figure with subplots
    fig_width = min(4 * (num_words + 1), 20)
    fig, axes = plt.subplots(1, num_words + 1, figsize=(fig_width, 4))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')
    
    # Show attention for each word
    for idx, (word, alpha) in enumerate(zip(caption_words, attention_weights)):
        if idx >= len(axes) - 1:
            break
        
        ax = axes[idx + 1]
        
        # Reshape attention to 2D (assuming 8x8 grid from encoder)
        alpha = alpha.cpu().detach().numpy()
        if len(alpha.shape) == 1:
            grid_size = int(np.sqrt(len(alpha)))
            alpha = alpha.reshape(grid_size, grid_size)
        
        # Resize attention to image size
        alpha_resized = np.array(Image.fromarray(alpha).resize((224, 224), Image.BILINEAR))
        
        # Show image with attention overlay
        ax.imshow(image)
        ax.imshow(alpha_resized, alpha=0.6, cmap='jet')
        ax.set_title(word, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Attention visualization saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test the utilities when run directly
    print("=" * 60)
    print("UTILITY FUNCTIONS TEST")
    print("=" * 60)
    
    # Test vocabulary
    print("\n📚 Testing Vocabulary...")
    vocab = Vocabulary()
    test_captions = [
        "A dog runs in the park",
        "The dog is running fast",
        "A cat sits on the mat",
        "The cat is sleeping peacefully",
    ]
    vocab.build_vocab(test_captions, min_freq=1)
    
    # Test encoding/decoding
    encoded = vocab.encode("a dog runs")
    decoded = vocab.decode(encoded)
    print(f"   Encoded 'a dog runs': {encoded}")
    print(f"   Decoded back: '{decoded}'")
    
    print("\n✅ All utility functions ready!")
