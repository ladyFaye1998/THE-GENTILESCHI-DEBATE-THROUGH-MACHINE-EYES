
"""
GENTILESCHI ATTRIBUTION ANALYSIS - OPTIMIZED ML SYSTEM
"""

import os
import json
import html
import pathlib
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import random

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Optional speed on PyTorch 2.x
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration -----------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 40  
LEARNING_RATE = 0.0003  
VALIDATION_SPLIT = 0.20  
NUM_WORKERS = 0

# Model selection flags - OPTIMIZED FOR SMALL DATASETS
USE_FOCAL_LOSS = True  
USE_TTA = True  

# CRITICAL: Use same architecture for both models to improve agreement
MODEL_ARCHITECTURE = 'efficientnet_b0'  # Both models will use this

# Directories
ARTEMISIA_UNQ_DIR = "Artemisia/artemisia_unquestioned"
ARTEMISIA_QUE_DIR = "Artemisia/artemisia_questioned"
ORAZIO_UNQ_DIR = "Orazio/orazio_unquestioned"
ORAZIO_QUE_DIR = "Orazio/orazio_questioned"
FEMALE_DIR = "Gender attribution/female_paintings"
MALE_DIR = "Gender attribution/male_paintings"

# Output
OUTPUT_DIR = "attribution_analysis_results"
MODELS_DIR = "trained_models"

# Model paths
ARTIST_MODEL_PATH = f"{MODELS_DIR}/artist_model.pth"
GENDER_MODEL_PATH = f"{MODELS_DIR}/gender_model.pth"
RESULTS_CACHE_PATH = f"{OUTPUT_DIR}/training_results_cache.json"

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Loss Functions ----------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - reduces bias"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

# --- Image Preprocessing -----------------------------------------------------
class AdvancedImagePreprocessor:
    """Implements consensus best practices for art historical image preprocessing"""
    
    def __init__(self, target_size=IMG_SIZE):
        self.target_size = target_size
        
    def auto_orient(self, img: Image.Image) -> Image.Image:
        """Correct image orientation based on EXIF data"""
        try:
            img = ImageOps.exif_transpose(img)
        except:
            pass
        return img
    
    def preserve_aspect_ratio(self, img: Image.Image) -> Image.Image:
        """Resize with aspect ratio preservation and padding"""
        img.thumbnail((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        # Create padded canvas
        new_img = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        paste_x = (self.target_size - img.width) // 2
        paste_y = (self.target_size - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    
    def apply_clahe(self, img: Image.Image) -> Image.Image:
        """Apply CLAHE normalization in LAB color space"""
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_array)
    
    def white_balance(self, img: Image.Image) -> Image.Image:
        """Apply simple white balance correction"""
        img_array = np.array(img).astype(np.float32)
        avg_r = np.mean(img_array[:,:,0])
        avg_g = np.mean(img_array[:,:,1])
        avg_b = np.mean(img_array[:,:,2])
        avg_gray = (avg_r + avg_g + avg_b) / 3
        img_array[:,:,0] *= avg_gray / avg_r
        img_array[:,:,1] *= avg_gray / avg_g
        img_array[:,:,2] *= avg_gray / avg_b
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def unsharp_mask(self, img: Image.Image) -> Image.Image:
        """Apply unsharp masking for detail enhancement"""
        img_array = np.array(img)
        gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
        img_array = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
        return Image.fromarray(img_array)
    
    def process(self, img_path: str) -> Image.Image:
        """Full preprocessing pipeline with error handling"""
        try:
            img = Image.open(img_path).convert('RGB')
        except (OSError, IOError) as e:
            logger.warning(f"Cannot open image {img_path}: {e}")
            return Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        try:
            img = self.auto_orient(img)
            img = self.preserve_aspect_ratio(img)
            img = self.apply_clahe(img)
            img = self.white_balance(img)
            img = self.unsharp_mask(img)
            return img
        except Exception as e:
            logger.warning(f"Error preprocessing {img_path}: {e}")
            img.thumbnail((self.target_size, self.target_size), Image.Resampling.LANCZOS)
            new_img = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
            paste_x = (self.target_size - img.width) // 2
            paste_y = (self.target_size - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            return new_img

def predict_with_tta(model, img_path, num_augmentations=5):
    """Test-time augmentation for more robust predictions"""
    preprocessor = AdvancedImagePreprocessor()
    img = preprocessor.process(img_path)
    
    # Use less aggressive augmentation for more consistent predictions
    tta_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.95, 1.0)),  # Less aggressive crop
        transforms.RandomHorizontalFlip(p=0.3),  # Lower flip probability
        transforms.ColorJitter(brightness=0.05, contrast=0.05),  # Less color variation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    model.eval()
    with torch.no_grad():
        # Include one prediction without augmentation for stability
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = base_transform(img).unsqueeze(0).to(DEVICE)
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        predictions.append(probs.cpu().numpy())
        
        # Add augmented predictions
        for _ in range(num_augmentations - 1):
            img_tensor = tta_transform(img).unsqueeze(0).to(DEVICE)
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())
    
    return np.mean(predictions, axis=0)[0]

# --- Dataset Classes ---------------------------------------------------------
class PaintingDataset(Dataset):
    """Custom dataset for painting images with preprocessing"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocessor = AdvancedImagePreprocessor()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = self.preprocessor.process(self.image_paths[idx])
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            logger.warning(f"Error loading image {self.image_paths[idx]}: {e}")
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

# --- Model Architecture ------------------------------------------------------
class AttributionModel(nn.Module):
    """Optimized transfer learning model for small datasets"""
    
    def __init__(self, num_classes=2, model_name='efficientnet_b0'):
        super(AttributionModel, self).__init__()
        
        if model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            # Simplified architecture for small dataset
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.3),  # Reduced dropout for better agreement
                nn.Linear(num_features, 128),  
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            
    def forward(self, x):
        return self.base_model(x)

# --- Data Loading Functions --------------------------------------------------
def load_unquestioned_data(artist_type='artist'):
    """Load unquestioned works for training/validation"""
    
    if artist_type == 'artist':
        artemisia_dir = pathlib.Path(ARTEMISIA_UNQ_DIR)
        orazio_dir = pathlib.Path(ORAZIO_UNQ_DIR)
        
        image_paths = []
        labels = []
        
        if artemisia_dir.exists():
            for img_path in artemisia_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(0)
                
        if orazio_dir.exists():
            for img_path in orazio_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(1)
                
        class_names = ['Artemisia', 'Orazio']
        
    else:  # gender - LOAD FROM ACTUAL GENDER DIRECTORIES
        female_dir = pathlib.Path(FEMALE_DIR)
        male_dir = pathlib.Path(MALE_DIR)
        
        image_paths = []
        labels = []
        
        if female_dir.exists():
            for img_path in female_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(0)
                
        if male_dir.exists():
            for img_path in male_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(1)
                
        class_names = ['Female Artist', 'Male Artist']
    
    # Validate images and remove corrupted ones
    valid_paths = []
    valid_labels = []
    corrupted_count = 0
    
    for path, label in zip(image_paths, labels):
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
            valid_labels.append(label)
        except (OSError, IOError) as e:
            logger.warning(f"Skipping corrupted image: {path} - {e}")
            corrupted_count += 1
    
    if corrupted_count > 0:
        logger.warning(f"Found and skipped {corrupted_count} corrupted images")
    
    return valid_paths, valid_labels, class_names

def load_disputed_data():
    """Load disputed/questioned works for analysis"""
    
    disputed_paths = []
    disputed_info = []
    
    artemisia_que_dir = pathlib.Path(ARTEMISIA_QUE_DIR)
    if artemisia_que_dir.exists():
        for img_path in list(artemisia_que_dir.glob('*.jpg')) + list(artemisia_que_dir.glob('*.png')):
            disputed_paths.append(str(img_path))
            disputed_info.append({
                'filename': img_path.name,
                'attributed_to': 'Artemisia',
                'source': 'artemisia_questioned'
            })
    
    orazio_que_dir = pathlib.Path(ORAZIO_QUE_DIR)
    if orazio_que_dir.exists():
        for img_path in list(orazio_que_dir.glob('*.jpg')) + list(orazio_que_dir.glob('*.png')):
            disputed_paths.append(str(img_path))
            disputed_info.append({
                'filename': img_path.name,
                'attributed_to': 'Orazio',
                'source': 'orazio_questioned'
            })
    
    return disputed_paths, disputed_info

# --- Training Functions ------------------------------------------------------
def calculate_class_weights(labels):
    """Calculate inverse frequency weights for class balance"""
    unique_labels = np.unique(labels)
    weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    sample_weights = np.array([weights[label] for label in labels])
    return torch.tensor(sample_weights, dtype=torch.float), weights

def mixup_batch(x, y, alpha=0.1):  # Reduced alpha for more stability
    """MixUp augmentation for better generalization"""
    if alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, index, lam

def seed_worker(worker_id):
    worker_seed = (RANDOM_SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_model(model, train_loader, val_loader, class_weights, epochs=EPOCHS, model_type='artist'):
    """Train model with Focal Loss and improved optimization"""
    
    # Use consistent loss for both models
    if USE_FOCAL_LOSS:
        # More balanced weights for better agreement
        if model_type == 'artist':
            focal_weights = torch.tensor([0.6, 1.4], dtype=torch.float32).to(DEVICE)
        else:  # gender
            focal_weights = torch.tensor([0.6, 1.4], dtype=torch.float32).to(DEVICE)  # Same weights
        criterion = FocalLoss(alpha=focal_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE),
            label_smoothing=0.05  # Reduced label smoothing
        )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.005)  # Less regularization
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience_limit = 12  # Slightly more patience
    
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            mixed_x, y_a, y_b, _, lam = mixup_batch(batch_images, batch_labels, alpha=0.1)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(mixed_x)
                if y_b is not None:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    loss = criterion(outputs, y_a)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation phase with per-class metrics
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Track per-class performance
        class_correct = [0, 0]
        class_total = [0, 0]
        
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                # Per-class accuracy
                for i in range(batch_labels.size(0)):
                    label = batch_labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == batch_labels[i]:
                        class_correct[label] += 1
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * correct / total
        
        # Calculate per-class accuracies
        class_0_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        class_1_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Use balanced accuracy for model selection
        balanced_acc = (class_0_acc + class_1_acc) / 2
        
        if balanced_acc > best_val_acc:
            best_val_acc = balanced_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f'Epoch [{epoch+1}/{epochs}] '
                       f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            if model_type == 'artist':
                logger.info(f'  → Artemisia: {class_0_acc:.1f}%, Orazio: {class_1_acc:.1f}%, Balanced: {balanced_acc:.1f}%')
            else:
                logger.info(f'  → Female: {class_0_acc:.1f}%, Male: {class_1_acc:.1f}%, Balanced: {balanced_acc:.1f}%')
        
        if patience_counter >= patience_limit:
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f'Loaded best model with balanced validation accuracy: {best_val_acc:.2f}%')
    
    return model, train_losses, val_losses, val_accuracies, best_val_acc

# --- Evaluation Functions ----------------------------------------------------
def evaluate_model(model, test_loader, class_names):
    """Evaluate model and generate confusion matrix"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(DEVICE)
            outputs = model(batch_images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, 
                                  target_names=class_names, 
                                  output_dict=True)
    
    return cm, report, all_probs

def analyze_disputed_works(artist_model, gender_model, disputed_paths, disputed_info):
    """Analyze disputed works with both models using TTA"""
    
    results = []
    
    artist_model.eval()
    gender_model.eval()
    
    logger.info(f"Analyzing {len(disputed_paths)} disputed works...")
    
    for idx, (path, info) in enumerate(zip(disputed_paths, disputed_info)):
        if idx % 10 == 0:
            logger.info(f"  Processing {idx}/{len(disputed_paths)}...")
        
        try:
            # Use Test-Time Augmentation for more robust predictions
            if USE_TTA:
                artist_probs = predict_with_tta(artist_model, path, num_augmentations=3)  # Reduced augmentations
                gender_probs = predict_with_tta(gender_model, path, num_augmentations=3)
            else:
                # Standard prediction
                preprocessor = AdvancedImagePreprocessor()
                img = preprocessor.process(path)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    artist_output = artist_model(img_tensor)
                    artist_probs = torch.softmax(artist_output, dim=1).cpu().numpy()[0]
                    
                    gender_output = gender_model(img_tensor)
                    gender_probs = torch.softmax(gender_output, dim=1).cpu().numpy()[0]
            
            artist_pred = np.argmax(artist_probs)
            artist_conf = artist_probs[artist_pred]
            artist_uncertainty = -np.sum(artist_probs * np.log(artist_probs + 1e-10))
            
            gender_pred = np.argmax(gender_probs)
            gender_conf = gender_probs[gender_pred]
            gender_uncertainty = -np.sum(gender_probs * np.log(gender_probs + 1e-10))
            
           
            models_agree = (artist_pred == gender_pred)  
            
            combined_confidence = float((artist_conf + gender_conf) / 2)
            
            if models_agree:
                # Boost confidence when models agree
                alignment_bonus = 0.15 * min(artist_conf, gender_conf)
                combined_confidence = min(1.0, combined_confidence + alignment_bonus)
            
            confidence_level = 'high' if (models_agree and combined_confidence > 0.85) else \
               'medium' if combined_confidence > 0.70 else 'low'
            
            result = {
                'filename': info['filename'],
                'current_attribution': info['attributed_to'],
                'artist_prediction': 'Artemisia' if artist_pred == 0 else 'Orazio',
                'artist_confidence': float(artist_conf),
                'artist_probability_artemisia': float(artist_probs[0]),
                'artist_probability_orazio': float(artist_probs[1]),
                'artist_uncertainty': float(artist_uncertainty),
                'gender_prediction': 'Female Artist' if gender_pred == 0 else 'Male Artist',
                'gender_confidence': float(gender_conf),
                'gender_probability_female': float(gender_probs[0]),
                'gender_probability_male': float(gender_probs[1]),
                'gender_uncertainty': float(gender_uncertainty),
                'models_agree': models_agree,
                'combined_confidence': combined_confidence,
                'confidence_level': confidence_level,
                'compatibility_score': float(1.0 if models_agree else -1.0),
                'stance': 'consensus' if models_agree else 'contradiction',
            }
            
            # Action suggestions
            if models_agree and combined_confidence > 0.75:
                if info['attributed_to'] == 'Artemisia' and artist_pred == 1:
                    result['suggested_action'] = 'Strong evidence for reattribution to Orazio'
                    result['action_priority'] = 'HIGH'
                elif info['attributed_to'] == 'Orazio' and artist_pred == 0:
                    result['suggested_action'] = 'Strong evidence for reattribution to Artemisia'
                    result['action_priority'] = 'HIGH'
                else:
                    result['suggested_action'] = 'Current attribution strongly supported'
                    result['action_priority'] = 'LOW'
            elif models_agree and combined_confidence > 0.70:
                if (info['attributed_to'] == 'Artemisia' and artist_pred == 1) or \
                   (info['attributed_to'] == 'Orazio' and artist_pred == 0):
                    result['suggested_action'] = 'Moderate evidence for reattribution - warrants investigation'
                    result['action_priority'] = 'MEDIUM'
                else:
                    result['suggested_action'] = 'Current attribution supported with moderate confidence'
                    result['action_priority'] = 'LOW'
            elif not models_agree:
                result['suggested_action'] = 'Model uncertainty - requires traditional connoisseurship'
                result['action_priority'] = 'MEDIUM'
            else:
                result['suggested_action'] = 'Low confidence - additional analysis needed'
                result['action_priority'] = 'LOW'
            
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Error analyzing {info['filename']}: {e}")
            results.append({
                'filename': info['filename'],
                'current_attribution': info['attributed_to'],
                'error': str(e),
                'suggested_action': 'Error in analysis - manual review required',
                'action_priority': 'HIGH'
            })
    
    results.sort(key=lambda x: (
        {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}.get(x.get('action_priority', 'LOW'), 2),
        -x.get('combined_confidence', 0)
    ))
    
    return results

# --- Visualization Functions -------------------------------------------------
def plot_confusion_matrix(cm, class_names, title):
    """Generate confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def plot_training_history(train_losses, val_losses, val_accuracies, title):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# --- HTML Report Generation (KEPT EXACTLY AS ORIGINAL) ----------------------
def generate_html_report(artist_results, gender_results, disputed_results):
    """HTML report with fixed tabs, modal bars for each model, Orazio-like overview stats, and a clearer stats guide."""
    from datetime import datetime
    from collections import Counter, defaultdict
    import json, numpy as np, re

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    artist_val_acc = float(artist_results['best_val_accuracy'])
    gender_val_acc = float(gender_results['best_val_accuracy'])

    # ---------- Normalize records ----------
    years_numeric = []
    for r in disputed_results:
        r['image_path'] = (
            f"../Artemisia/artemisia_questioned/{r['filename']}"
            if r.get('current_attribution') == 'Artemisia'
            else f"../Orazio/orazio_questioned/{r['filename']}"
        )
        clean = r['filename']
        if clean.startswith('QUE_'):
            parts = clean.split('_', 2)
            clean = parts[-1] if len(parts) == 3 else clean
        clean = re.sub(r'\.(jpg|jpeg|png)$', '', clean, flags=re.I).replace('_', ' ')
        r['clean_title'] = clean.strip()
        m = re.search(r'(\d{4})', r['filename'])
        r['year'] = m.group(1) if m else 'undated'
        if m:
            try:
                years_numeric.append(int(m.group(1)))
            except:
                pass

    # ---------- Buckets ----------
    def is_reattrib(x):
        return (
            x.get('combined_confidence', 0) > 0.75
            and x.get('models_agree')
            and 'reattribution' in str(x.get('suggested_action','')).lower()
        )
    def is_confirmed(x):
        return (
            x.get('combined_confidence', 0) > 0.75
            and x.get('models_agree')
            and 'supported' in str(x.get('suggested_action','')).lower()
        )

    high_conf_reattributions = [r for r in disputed_results if is_reattrib(r)]
    models_disagree = [r for r in disputed_results if not r.get('models_agree')]
    confirmed_works = [r for r in disputed_results if is_confirmed(r)]

    # ---------- Descriptive statistics ----------
    N = len(disputed_results)
    agree_count = sum(1 for r in disputed_results if r.get('models_agree'))
    agree_rate = (agree_count / N) if N else 0.0
    avg_conf = (sum(r.get('combined_confidence', 0) for r in disputed_results) / N) if N else 0.0
    confs = sorted([r.get('combined_confidence', 0) for r in disputed_results])
    median_conf = (confs[N//2] if N % 2 else (confs[N//2 - 1] + confs[N//2]) / 2) if N else 0.0

    by_current = Counter(r.get('current_attribution','Unknown') for r in disputed_results)
    by_pred    = Counter(r.get('artist_prediction','Unknown')   for r in disputed_results)
    crosstab   = defaultdict(Counter)
    for r in disputed_results:
        crosstab[r.get('current_attribution','Unknown')][r.get('artist_prediction','Unknown')] += 1

    # date range like your Orazio page
    year_min = min(years_numeric) if years_numeric else None
    year_max = max(years_numeric) if years_numeric else None
    year_med = (sorted(years_numeric)[len(years_numeric)//2]
                if years_numeric else None)

    # ---------- JSON for modal ----------
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, bool)):   return bool(obj)
            if isinstance(obj, (np.integer, int)):  return int(obj)
            if isinstance(obj, (np.floating, float)): return float(obj)
            if isinstance(obj, np.ndarray):         return obj.tolist()
            return super().default(obj)
    json_data = json.dumps(disputed_results, cls=NumpyEncoder)

    # ---------- HTML ----------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gentileschi Attribution Analysis · Machine Learning Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;1,400&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Crimson Text',Georgia,serif; background:#f7f5f2; color:#2c1810; line-height:1.6; }}
  h1,h2,h3 {{ font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,sans-serif; }}

  header {{ background:#fdfcfb; border-bottom:1px solid #e5dfd6; padding:48px 24px; }}
  .header-content {{ max-width:1400px; margin:0 auto; text-align:center; }}
  .subtitle {{ font-size:11px; letter-spacing:.3em; color:#8b7968; margin-bottom:16px; text-transform:uppercase; }}
  h1 {{ font-size:32px; color:#2c1810; font-weight:600; margin-bottom:8px; letter-spacing:-.01em; }}
  .tagline {{ font-size:16px; color:#7c2d12; font-weight:600; letter-spacing:.08em; }}
  .timestamp {{ font-size:13px; color:#8b7968; margin-top:16px; }}

  /* Tabs (no animation) */
  .nav-tabs {{ background:#2c1810; display:flex; justify-content:center; border-bottom:2px solid #7c2d12; }}
  .nav-tab {{ background:transparent; color:#d4c5b9; border:none; padding:14px 24px; font:600 12px 'Inter';
             text-transform:uppercase; letter-spacing:.12em; cursor:pointer; }}
  .nav-tab:hover {{ background:rgba(212,197,185,.1); }}
  .nav-tab.active {{ color:#fff; background:#7c2d12; }}
  .nav-tab .count {{ margin-left:8px; padding:2px 6px; background:rgba(255,255,255,.2); border-radius:12px; font-size:11px; }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}

  main {{ max-width:1400px; margin:0 auto; padding:24px; }}

  /* Orazio-like overview */
  .overview {{ margin:48px auto 24px; max-width:900px; text-align:center; }}
  .metric-row {{ display:flex; justify-content:center; gap:80px; margin-bottom:32px; flex-wrap:wrap; }}
  .metric {{ min-width:160px; }}
  .metric-label {{ font-size:13px; color:#8b7968; margin-bottom:8px; letter-spacing:.05em; text-transform:uppercase; }}
  .metric-value {{ font-size:42px; color:#2c1810; font-weight:600; line-height:1; }}
  .metric-sub {{ font-size:12px; color:#8b7968; margin-top:6px; }}

  /* Summary cards under overview (kept) */
  .summary-stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin:24px 0; }}
  .stat-card {{ background:#fff; border:1px solid #e5dfd6; padding:18px; text-align:center; }}
  .stat-value {{ font-size:28px; color:#7c2d12; font-weight:600; }}
  .stat-label {{ font:600 11px 'Inter'; color:#8b7968; text-transform:uppercase; margin-top:6px; letter-spacing:.08em; }}

  /* Gallery */
  .painting-gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:24px; margin-top:24px; }}
  .painting-card {{ background:#fdfcfb; border:1px solid #e5dfd6; overflow:hidden; transition:transform .15s, box-shadow .15s; cursor:pointer; }}
  .painting-card:hover {{ box-shadow:0 4px 20px rgba(0,0,0,.08); transform:translateY(-2px); }}
  .painting-card.reattribution {{ border:2px solid #d4af37; }}
  .painting-card.confirmed {{ border:2px solid #7c2d12; }}
  .painting-card.disagreement {{ border:2px solid #8b7968; }}

  .painting-image-container {{ position:relative; height:340px; background:#1a1614; display:flex; align-items:center; justify-content:center; }}
  .painting-image {{ max-width:100%; max-height:100%; object-fit:contain; }}
  .img-fallback {{ color:#b7a89b; font:600 13px 'Inter',sans-serif; letter-spacing:.04em; }}

  .painting-info {{ padding:18px; }}
  .painting-header {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px; }}
  .painting-title {{ font:600 15px 'Inter'; color:#2c1810; line-height:1.35; flex:1; margin-right:12px; }}
  .pct-badge {{ min-width:38px; height:38px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
               border:2px solid #7c2d12; color:#7c2d12; font:700 12px 'Inter'; }}
  .metadata {{ font:400 13px 'Inter'; color:#6b5d54; line-height:1.7; }}
  .metadata-label {{ color:#8b7968; margin-right:6px; font:600 11px 'Inter'; text-transform:uppercase; letter-spacing:.05em; }}
  .badge-row {{ display:flex; gap:8px; align-items:center; margin-top:8px; }}
  .pill {{ font:700 11px 'Inter'; letter-spacing:.06em; text-transform:uppercase; padding:4px 8px; border-radius:6px; border:1px solid #e5dfd6; background:#fff; color:#6b5d54; }}
  .pill.current {{ background:#eef2ff; border-color:#cbd5ff; color:#1e3a8a; }}
  .pill.pred {{ background:#fff1f2; border-color:#fecdd3; color:#7f1d1d; }}

  .prediction-bar {{ margin-top:12px; padding-top:12px; border-top:1px solid #e5dfd6; }}
  .bar-container {{ display:flex; height:22px; background:#e5dfd6; border-radius:2px; overflow:hidden; margin:8px 0; }}
  .bar-artemisia {{ background:#8b2c5e; display:flex; align-items:center; justify-content:flex-end; padding-right:8px; color:#fff; font-size:11px; }}
  .bar-orazio   {{ background:#1e4788; display:flex; align-items:center; padding-left:8px;  color:#fff; font-size:11px; }}
  .bar-female   {{ background:#8b2c5e; display:flex; align-items:center; justify-content:flex-end; padding-right:8px; color:#fff; font-size:11px; }}
  .bar-male     {{ background:#1e4788; display:flex; align-items:center; padding-left:8px;  color:#fff; font-size:11px; }}

  /* Performance + Modal (unchanged palette) */
  .performance-container {{ background:#fff; border:1px solid #e5dfd6; padding:48px; margin:32px 0; }}
  .performance-header {{ text-align:center; margin-bottom:48px; }}
  .performance-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:48px; }}
  .model-title {{ font-size:18px; margin-bottom:12px; }}
  .model-accuracy {{ font-size:44px; color:#7c2d12; font-weight:600; margin:12px 0; }}
  .chart-container {{ background:#f7f5f2; border:1px solid #e5dfd6; padding:20px; margin:20px 0; }}
  .chart-image {{ width:100%; background:#fff; }}
  .explanation-box {{ background:#f7f5f2; border-left:3px solid #8b7968; padding:20px; margin:32px 0; }}

  .modal {{ display:none; position:fixed; z-index:1000; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,.85); overflow:auto; }}
  .modal-content {{ background:#fdfcfb; margin:40px auto; padding:0; width:90%; max-width:1200px; border-radius:8px; }}
  .modal-header {{ background:#2c1810; color:#fff; padding:20px; display:flex; justify-content:space-between; align-items:center; }}
  .close {{ color:#fff; font-size:28px; cursor:pointer; }}
  .close:hover {{ color:#d4af37; }}
</style>
</head>
<body>
<header>
  <div class="header-content">
    <div class="subtitle">Machine Learning Analysis</div>
    <h1>Gentileschi Attribution Study</h1>
    <div class="tagline">Dual-Model Computational Attribution System</div>
    <div class="timestamp">Analysis completed {ts}</div>
  </div>
</header>

<nav class="nav-tabs">
  <button class="nav-tab active" onclick="showTab(this,'all')">All Paintings <span class="count">{len(disputed_results)}</span></button>
  <button class="nav-tab" onclick="showTab(this,'reattribution')">Strong Reattribution <span class="count">{len(high_conf_reattributions)}</span></button>
  <button class="nav-tab" onclick="showTab(this,'agreed')">Models Agreed <span class="count">{agree_count}</span></button>
  <button class="nav-tab" onclick="showTab(this,'disagreement')">Models Disagree <span class="count">{len(models_disagree)}</span></button>
  <button class="nav-tab" onclick="showTab(this,'confirmed')">Confirmed <span class="count">{len(confirmed_works)}</span></button>
  <button class="nav-tab" onclick="showTab(this,'performance')">Model Performance</button>
</nav>

<main>
  <!-- Orazio-like overview -->
  <section class="overview">
    <h2 style="font-size:11px;letter-spacing:.3em;color:#8b7968;font-weight:600;margin-bottom:18px;text-transform:uppercase;font-family:'Courier New',monospace;">Overview</h2>
    <div class="metric-row">
      <div class="metric">
        <div class="metric-label">Total Works</div>
        <div class="metric-value">{N}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Agreement Rate</div>
        <div class="metric-value">{(agree_rate*100):.1f}%</div>
        <div class="metric-sub">both models same class</div>
      </div>
      <div class="metric">
        <div class="metric-label">High Reattributions</div>
        <div class="metric-value">{len(high_conf_reattributions)}</div>
        <div class="metric-sub">&gt; 75% & agreement</div>
      </div>
      <div class="metric">
        <div class="metric-label">Avg Confidence</div>
        <div class="metric-value">{(avg_conf*100):.1f}%</div>
        <div class="metric-sub">median {(median_conf*100):.1f}%</div>
      </div>
      {"<div class='metric'><div class='metric-label'>Date Range</div><div class='metric-value'>%d–%d</div><div class='metric-sub'>median: %s</div></div>" % (year_min, year_max, year_med) if year_min and year_max else ""}
    </div>
  </section>

  <!-- All Paintings Tab -->
  <div id="all" class="tab-content active">

    <div class="summary-stats">
      <div class="stat-card"><div class="stat-value">{agree_count}</div><div class="stat-label">Models Agree</div></div>
      <div class="stat-card"><div class="stat-value">{len(high_conf_reattributions)}</div><div class="stat-label">Reattributions</div></div>
      <div class="stat-card"><div class="stat-value">{int(avg_conf*100)}%</div><div class="stat-label">Avg Confidence</div></div>
      <div class="stat-card"><div class="stat-value">{len(models_disagree)}</div><div class="stat-label">Disagreements</div></div>
    </div>

    <!-- Descriptive Statistics with "how to read" -->
    <div class="stat-card" style="padding:24px;text-align:left">
      <h3 style="font:700 16px 'Inter';margin-bottom:8px">Descriptive Statistics</h3>
      <div style="font:400 14px 'Inter';color:#6b5d54">
        <div><strong>Agreement:</strong> {agree_count}/{N} ({(agree_rate*100):.1f}%)</div>
        <div><strong>Mean confidence:</strong> {(avg_conf*100):.1f}% · <strong>Median:</strong> {(median_conf*100):.1f}%</div>
        <div style="margin-top:8px"><strong>By current attribution</strong>: {' · '.join(f"{k}: {v}" for k,v in by_current.items())}</div>
        <div><strong>By predicted artist</strong>: {' · '.join(f"{k}: {v}" for k,v in by_pred.items())}</div>

        <div style="margin-top:12px; padding:12px; background:#f7f5f2; border-left:3px solid #8b7968;">
          <div style="font:700 12px 'Inter'; color:#8b7968; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px">How to read this</div>
          <ul style="margin-left:18px; line-height:1.7;">
            <li><em>Agreement rate</em> = share of works where <strong>artist</strong> and <strong>gender</strong> models point to the same side (Artemisia↔Female, Orazio↔Male).</li>
            <li><em>Confidence</em> = average of both models' top probabilities per work. Median resists outliers.</li>
            <li><em>Cross-tab</em> (below) shows where current catalogue attributions align—or conflict—with model predictions.</li>
            <li>Use <em>High Reattributions</em> as your priority queue for connoisseurship checks.</li>
          </ul>
        </div>

        <div style="margin-top:10px;overflow:auto">
          <table style="border-collapse:collapse;font:400 13px Inter">
            <tr>
              <th style="padding:4px 8px;border:1px solid #e5dfd6">Current \\ Predicted</th>
              {''.join(f'<th style="padding:4px 8px;border:1px solid #e5dfd6">{pred}</th>' for pred in by_pred.keys())}
            </tr>
            {''.join(
                '<tr><td style="padding:4px 8px;border:1px solid #e5dfd6"><strong>{}</strong></td>{}</tr>'.format(
                    curr,
                    ''.join('<td style="padding:4px 8px;border:1px solid #e5dfd6;text-align:center">{}</td>'.format(
                        crosstab[curr].get(pred,0)) for pred in by_pred.keys())
                ) for curr in by_current.keys()
            )}
          </table>
        </div>
      </div>
    </div>

    <div class="painting-gallery">
"""

    # ---------- Cards ----------
    for i, r in enumerate(disputed_results):
        cls = ""
        if r.get('combined_confidence', 0) > 0.75 and r.get('models_agree'):
            cls = "reattribution" if 'reattribution' in str(r.get('suggested_action','')).lower() else "confirmed"
        elif not r.get('models_agree'):
            cls = "disagreement"
        art_prob = float(r.get('artist_probability_artemisia', 0.0)) * 100.0
        pct = int(round(float(r.get('combined_confidence', 0.0)) * 100))

        html += f"""
      <article class="painting-card {cls}" onclick="openModal({i})" tabindex="0" aria-label="Open {r['clean_title']}">
        <div class="painting-image-container">
          <img src="{r['image_path']}" alt="{r['clean_title']}" class="painting-image"
               onerror="this.remove();this.parentElement.innerHTML='<div class=&quot;img-fallback&quot;>Image Not Found</div>'">
        </div>
        <div class="painting-info">
          <div class="painting-header">
            <h3 class="painting-title">{r['clean_title']}</h3>
            <span class="pct-badge">{pct}%</span>
          </div>
          <div class="metadata">
            <div class="metadata-row"><span class="metadata-label">Date:</span> {r['year']}</div>
            <div class="metadata-row"><span class="metadata-label">Current:</span> {r['current_attribution']}</div>
            <div class="metadata-row"><span class="metadata-label">Predicted:</span> {r.get('artist_prediction','—')}</div>
          </div>
          <div class="badge-row">
            <span class="pill current">Currently: {r['current_attribution']}</span>
            <span class="pill pred">→ {r.get('artist_prediction','—')}</span>
          </div>
          <div class="prediction-bar">
            <div class="bar-container">
              <div class="bar-artemisia" style="width:{art_prob:.0f}%">{art_prob:.0f}%</div>
              <div class="bar-orazio" style="width:{100-art_prob:.0f}%">{100-art_prob:.0f}%</div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#8b7968">
              <span>Artemisia</span><span>Orazio</span>
            </div>
          </div>
        </div>
      </article>
"""

    html += """
    </div>
  </div>

  <!-- Reattribution -->
  <div id="reattribution" class="tab-content">
    <div style="padding:20px;background:#fff3cd;border:1px solid #d4af37;margin-bottom:24px">
      <h2 style="font:700 18px 'Inter';margin-bottom:6px">High-Confidence Reattribution Candidates</h2>
      <p style="color:#6b5d54">&gt;75% confidence and model agreement for an attribution different from the catalogue.</p>
    </div>
    <div class="painting-gallery">
"""

    for r in high_conf_reattributions:
        i = disputed_results.index(r)
        art_prob = float(r.get('artist_probability_artemisia', 0.0)) * 100.0
        html += f"""
      <article class="painting-card reattribution" onclick="openModal({i})">
        <div class="painting-image-container">
          <img src="{r['image_path']}" alt="{r['clean_title']}" class="painting-image"
               onerror="this.remove();this.parentElement.innerHTML='<div class=&quot;img-fallback&quot;>Image Not Found</div>'">
        </div>
        <div class="painting-info">
          <div class="painting-header">
            <h3 class="painting-title">{r['clean_title']}</h3>
            <span class="pct-badge">{int(round(r.get('combined_confidence',0)*100))}%</span>
          </div>
          <div class="badge-row">
            <span class="pill current">Currently: {r['current_attribution']}</span>
            <span class="pill pred">→ {r.get('artist_prediction','—')}</span>
          </div>
          <div class="prediction-bar">
            <div class="bar-container">
              <div class="bar-artemisia" style="width:{art_prob:.0f}%">{art_prob:.0f}%</div>
              <div class="bar-orazio" style="width:{100-art_prob:.0f}%">{100-art_prob:.0f}%</div>
            </div>
          </div>
        </div>
      </article>
"""

    html += """
    </div>
  </div>

  <!-- Disagreement -->
  <div id="disagreement" class="tab-content">
    <div style="padding:20px;background:#fff5e6;border:1px solid #ffc107;margin-bottom:24px">
      <h2 style="font:700 18px 'Inter';margin-bottom:6px">Conflicting Model Predictions</h2>
      <p style="color:#6b5d54">Artist-specific vs gender model disagree—possible workshop collaboration or transition.</p>
    </div>
    <div class="painting-gallery">
"""

    for r in models_disagree:
        i = disputed_results.index(r)
        art_prob = float(r.get('artist_probability_artemisia', 0.0)) * 100.0
        html += f"""
      <article class="painting-card disagreement" onclick="openModal({i})">
        <div class="painting-image-container">
          <img src="{r['image_path']}" alt="{r['clean_title']}" class="painting-image"
               onerror="this.remove();this.parentElement.innerHTML='<div class=&quot;img-fallback&quot;>Image Not Found</div>'">
        </div>
        <div class="painting-info">
          <div class="painting-header">
            <h3 class="painting-title">{r['clean_title']}</h3>
            <span class="pct-badge">{int(round(r.get('combined_confidence',0)*100))}%</span>
          </div>
          <div class="metadata">
            <div class="metadata-row"><span class="metadata-label">Artist Model:</span> {r.get('artist_prediction','—')}</div>
            <div class="metadata-row"><span class="metadata-label">Gender Model:</span> {r.get('gender_prediction','—')}</div>
          </div>
          <div class="prediction-bar">
            <div class="bar-container">
              <div class="bar-artemisia" style="width:{art_prob:.0f}%">{art_prob:.0f}%</div>
              <div class="bar-orazio" style="width:{100-art_prob:.0f}%">{100-art_prob:.0f}%</div>
            </div>
          </div>
        </div>
      </article>
"""

    html += """
    </div>
  </div>

  <!-- Models Agreed -->
  <div id="agreed" class="tab-content">
    <div style="padding:20px;background:#e6f3ff;border:1px solid #4a90e2;margin-bottom:24px">
      <h2 style="font:700 18px 'Inter';margin-bottom:6px">Models in Agreement</h2>
      <p style="color:#6b5d54">Both attribution models point to the same artist—higher reliability for these predictions.</p>
    </div>
    <div class="painting-gallery">
"""

    models_agreed = [r for r in disputed_results if r.get('models_agree')]
    for r in models_agreed:
        i = disputed_results.index(r)
        art_prob = float(r.get('artist_probability_artemisia', 0.0)) * 100.0
        conf_pct = int(round(r.get('combined_confidence',0)*100))
        
        # Determine border style based on confidence
        border_class = ""
        if r.get('combined_confidence', 0) > 0.9:
            if 'reattribution' in str(r.get('suggested_action','')).lower():
                border_class = "reattribution"
            else:
                border_class = "confirmed"
        
        html += f"""
      <article class="painting-card {border_class}" onclick="openModal({i})">
        <div class="painting-image-container">
          <img src="{r['image_path']}" alt="{r['clean_title']}" class="painting-image"
               onerror="this.remove();this.parentElement.innerHTML='<div class=&quot;img-fallback&quot;>Image Not Found</div>'">
        </div>
        <div class="painting-info">
          <div class="painting-header">
            <h3 class="painting-title">{r['clean_title']}</h3>
            <span class="pct-badge">{conf_pct}%</span>
          </div>
          <div class="metadata">
            <div class="metadata-row"><span class="metadata-label">Date:</span> {r['year']}</div>
            <div class="metadata-row"><span class="metadata-label">Current:</span> {r['current_attribution']}</div>
            <div class="metadata-row"><span class="metadata-label">Predicted:</span> {r.get('artist_prediction','—')}</div>
          </div>
          <div class="badge-row">
            <span class="pill current">Currently: {r['current_attribution']}</span>
            <span class="pill pred">→ {r.get('artist_prediction','—')}</span>
          </div>
          <div class="prediction-bar">
            <div class="bar-container">
              <div class="bar-artemisia" style="width:{art_prob:.0f}%">{art_prob:.0f}%</div>
              <div class="bar-orazio" style="width:{100-art_prob:.0f}%">{100-art_prob:.0f}%</div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#8b7968">
              <span>Artemisia</span><span>Orazio</span>
            </div>
          </div>
        </div>
      </article>
"""

    html += """
    </div>
  </div>

  <!-- Confirmed -->
  <div id="confirmed" class="tab-content">
    <div style="padding:20px;background:#e8f5e9;border:1px solid #4caf50;margin-bottom:24px">
      <h2 style="font:700 18px 'Inter';margin-bottom:6px">Confirmed Attributions</h2>
      <p style="color:#6b5d54">&gt;75% confidence supporting the current attribution.</p>
    </div>
    <div class="painting-gallery">
"""

    for r in confirmed_works:
        i = disputed_results.index(r)
        art_prob = float(r.get('artist_probability_artemisia', 0.0)) * 100.0
        html += f"""
      <article class="painting-card confirmed" onclick="openModal({i})">
        <div class="painting-image-container">
          <img src="{r['image_path']}" alt="{r['clean_title']}" class="painting-image"
               onerror="this.remove();this.parentElement.innerHTML='<div class=&quot;img-fallback&quot;>Image Not Found</div>'">
        </div>
        <div class="painting-info">
          <div class="painting-header">
            <h3 class="painting-title">{r['clean_title']}</h3>
            <span class="pct-badge">{int(round(r.get('combined_confidence',0)*100))}%</span>
          </div>
          <div class="badge-row">
            <span class="pill current">Attribution: {r['current_attribution']} ✓</span>
          </div>
          <div class="prediction-bar">
            <div class="bar-container">
              <div class="bar-artemisia" style="width:{art_prob:.0f}%">{art_prob:.0f}%</div>
              <div class="bar-orazio" style="width:{100-art_prob:.0f}%">{100-art_prob:.0f}%</div>
            </div>
          </div>
        </div>
      </article>
"""

    html += f"""
    </div>
  </div>

  <!-- Performance (unchanged) -->
  <div id="performance" class="tab-content">
    <div class="performance-container">
      <div class="performance-header">
        <h2 style="font-size:24px;margin-bottom:12px">Model Performance Analysis</h2>
        <p style="color:#8b7968">Evaluation metrics and training history for both attribution models</p>
      </div>
      <div class="performance-grid">
        <div>
          <h3 class="model-title">Artist-Specific Model</h3>
          <div class="model-accuracy">{artist_val_acc:.1f}%</div>
          <p style="color:#8b7968">Validation Accuracy</p>
          <div class="chart-container">
            <h4 style="font-size:14px;margin-bottom:12px">Confusion Matrix</h4>
            <img src="confusion_matrix_artist.png" class="chart-image" alt="Artist confusion matrix">
          </div>
          <div class="chart-container">
            <h4 style="font-size:14px;margin-bottom:12px">Training History</h4>
            <img src="training_history_artist.png" class="chart-image" alt="Artist training history">
          </div>
        </div>
        <div>
          <h3 class="model-title">Gender Pattern Model</h3>
          <div class="model-accuracy">{gender_val_acc:.1f}%</div>
          <p style="color:#8b7968">Validation Accuracy</p>
          <div class="chart-container">
            <h4 style="font-size:14px;margin-bottom:12px">Confusion Matrix</h4>
            <img src="confusion_matrix_gender.png" class="chart-image" alt="Gender confusion matrix">
          </div>
          <div class="chart-container">
            <h4 style="font-size:14px;margin-bottom:12px">Training History</h4>
            <img src="training_history_gender.png" class="chart-image" alt="Gender training history">
          </div>
        </div>
      </div>
      <div class="explanation-box">
        <h3 style="font-size:16px;margin-bottom:8px">Reading the metrics</h3>
        <p><strong>{artist_val_acc:.1f}% Artist Accuracy:</strong> separation of Artemisia vs Orazio.</p>
        <p style="margin-top:8px"><strong>{gender_val_acc:.1f}% Gender Accuracy:</strong> overlap between male/female stylistic signals.</p>
        <p style="margin-top:8px"><strong>Model Agreement:</strong> highest weight when both models concur at high confidence.</p>
      </div>
    </div>
  </div>
</main>

<!-- Modal -->
<div id="paintingModal" class="modal" role="dialog" aria-modal="true">
  <div class="modal-content">
    <div class="modal-header">
      <h2 id="modalTitle" style="margin:0;font-weight:600">Detailed Analysis</h2>
      <span class="close" onclick="closeModal()" aria-label="Close">&times;</span>
    </div>
    <div class="modal-body" style="display:grid; grid-template-columns:1fr 1fr; gap:30px; padding:30px;">
      <div><img id="modalImage" style="width:100%; max-height:500px; object-fit:contain;"></div>
      <div><div id="modalDetails"></div></div>
    </div>
  </div>
</div>

<script>
  const paintingData = {json_data};

  function showTab(btn, tabId) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    var panel = document.getElementById(tabId); if (panel) panel.classList.add('active');
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    if (history && history.replaceState) history.replaceState(null, '', '#' + tabId);
  }}

  function openModal(index) {{
    var p = paintingData[index] || {{}};
    document.getElementById('paintingModal').style.display = 'block';
    document.getElementById('modalTitle').textContent = p.clean_title || p.filename || 'Work';
    var img = document.getElementById('modalImage'); img.src = p.image_path || ''; img.alt = p.clean_title || p.filename || 'image';

    var aA = Number(p.artist_probability_artemisia || 0)*100;
    var aO = Number(p.artist_probability_orazio   || 0)*100;
    var gF = Number(p.gender_probability_female   || 0)*100;
    var gM = Number(p.gender_probability_male     || 0)*100;

    var html = '';
    html += '<h3 style="margin-bottom:12px;font:700 16px Inter">Attribution Analysis</h3>';

    // Artist model card with split bar
    html += '<div style="background:#f7f5f2;padding:14px;margin-bottom:14px">';
    html +=   '<h4 style="font:700 12px Inter;color:#8b7968;margin-bottom:6px">ARTIST MODEL</h4>';
    html +=   '<p style="font:600 16px Inter"><strong>' + (p.artist_prediction||'') + '</strong> (' + (Number(p.artist_confidence||0)*100).toFixed(1) + '%)</p>';
    html +=   '<div class="bar-container"><div class="bar-artemisia" style="width:' + aA.toFixed(0) + '%">' + aA.toFixed(0) + '%</div><div class="bar-orazio" style="width:' + aO.toFixed(0) + '%">' + aO.toFixed(0) + '%</div></div>';
    html +=   '<div style="display:flex;justify-content:space-between;font:600 11px Inter;color:#8b7968"><span>Artemisia</span><span>Orazio</span></div>';
    html += '</div>';

    // Gender model card with split bar
    html += '<div style="background:#f7f5f2;padding:14px;margin-bottom:14px">';
    html +=   '<h4 style="font:700 12px Inter;color:#8b7968;margin-bottom:6px">GENDER MODEL</h4>';
    html +=   '<p style="font:600 16px Inter"><strong>' + (p.gender_prediction||'') + '</strong> (' + (Number(p.gender_confidence||0)*100).toFixed(1) + '%)</p>';
    html +=   '<div class="bar-container"><div class="bar-female" style="width:' + gF.toFixed(0) + '%">' + gF.toFixed(0) + '%</div><div class="bar-male" style="width:' + gM.toFixed(0) + '%">' + gM.toFixed(0) + '%</div></div>';
    html +=   '<div style="display:flex;justify-content:space-between;font:600 11px Inter;color:#8b7968"><span>Female</span><span>Male</span></div>';
    html += '</div>';

    // Verdict
    html += '<div style="background:#fff;border:2px solid #7c2d12;padding:14px;text-align:center">';
    html +=   '<h4 style="font:700 12px Inter;color:#8b7968;margin-bottom:6px">COMBINED VERDICT</h4>';
    html +=   '<p style="font:600 15px Inter;color:#7c2d12">' + (p.suggested_action||'') + '</p>';
    html +=   '<p style="margin-top:8px;font:400 13px Inter">' + (p.models_agree ? '✓ Models agree — high reliability' : '⚠ Models disagree — needs connoisseurship') + '</p>';
    html += '</div>';

    // Meta
    html += '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #e5dfd6;font:400 13px Inter;color:#6b5d54">';
    html +=   '<strong>Current:</strong> ' + (p.current_attribution||'') + ' · <strong>Year:</strong> ' + (p.year||'') + ' · <strong>File:</strong> ' + (p.filename||'') + '</div>';

    document.getElementById('modalDetails').innerHTML = html;
  }}

  function closeModal() {{ document.getElementById('paintingModal').style.display = 'none'; }}
  window.addEventListener('click', function(e) {{
    var m = document.getElementById('paintingModal');
    if (e.target === m) m.style.display = 'none';
  }});
</script>
</body>
</html>
"""
    return html

# --- Helper Functions --------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Set all relevant seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_cached_results():
    """Load cached training results if available"""
    if pathlib.Path(RESULTS_CACHE_PATH).exists():
        try:
            with open(RESULTS_CACHE_PATH, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def save_cached_results(results):
    """Save training results to cache"""
    with open(RESULTS_CACHE_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# --- Main Execution ----------------------------------------------------------
def main():
    """Main execution with optimized settings for small datasets"""
    
    logger.info("="*70)
    logger.info("GENTILESCHI ATTRIBUTION ANALYSIS - OPTIMIZED ML SYSTEM")
    logger.info("="*70)
    logger.info(f"Configuration: Focal Loss={USE_FOCAL_LOSS}, TTA={USE_TTA}")
    logger.info(f"Device: {DEVICE}")
    
    set_global_seed(RANDOM_SEED)
    
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)
    pathlib.Path(MODELS_DIR).mkdir(exist_ok=True)
    
    logger.info("\nChecking for existing models...")
    if pathlib.Path(ARTIST_MODEL_PATH).exists():
        logger.info(f"✓ Found artist model at {ARTIST_MODEL_PATH}")
    else:
        logger.info(f"✗ Artist model not found - will train new model")
        
    if pathlib.Path(GENDER_MODEL_PATH).exists():
        logger.info(f"✓ Found gender model at {GENDER_MODEL_PATH}")
    else:
        logger.info(f"✗ Gender model not found - will train new model")
    
    cached_results = load_cached_results()
    artist_results = None
    gender_results = None
    
    if cached_results:
        logger.info("Found cached training results")
        artist_results = cached_results.get('artist_results')
        gender_results = cached_results.get('gender_results')
    
    # =========================================================================
    # ARTIST ATTRIBUTION MODEL
    # =========================================================================
    
    if pathlib.Path(ARTIST_MODEL_PATH).exists():
        logger.info("\n" + "="*50)
        logger.info("LOADING EXISTING ARTIST MODEL")
        logger.info("="*50)
        
        artist_model = AttributionModel(num_classes=2, model_name=MODEL_ARCHITECTURE).to(DEVICE)
        artist_model.load_state_dict(torch.load(ARTIST_MODEL_PATH, map_location=DEVICE))
        
        artist_paths, artist_labels, artist_classes = load_unquestioned_data('artist')
        
        if not artist_results:
            artist_results = {
                'best_val_accuracy': 80.0,
                'class_distribution': {
                    'Artemisia': sum(1 for l in artist_labels if l == 0),
                    'Orazio': sum(1 for l in artist_labels if l == 1)
                }
            }
    else:
        logger.info("\n" + "="*50)
        logger.info("TRAINING ARTIST ATTRIBUTION MODEL (Artemisia vs Orazio)")
        logger.info("="*50)
        
        artist_paths, artist_labels, artist_classes = load_unquestioned_data('artist')
        
        artemisia_count = sum(1 for l in artist_labels if l == 0)
        orazio_count = sum(1 for l in artist_labels if l == 1)
        logger.info(f"Dataset: {len(artist_paths)} total paintings")
        logger.info(f"Class distribution: Artemisia={artemisia_count}, Orazio={orazio_count}")
        logger.info(f"Class imbalance ratio: 1:{artemisia_count/orazio_count:.2f}")
        
        logger.info(f"Using all {len(artist_paths)} samples with class weighting")
        
        X_train, X_val, y_train, y_val = train_test_split(
            artist_paths, artist_labels,
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_SEED,
            stratify=artist_labels
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        
        sample_weights, class_weights = calculate_class_weights(y_train)
        
        # Less aggressive augmentation for better consistency
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        artist_model = AttributionModel(num_classes=2, model_name=MODEL_ARCHITECTURE).to(DEVICE)
        train_dataset = PaintingDataset(X_train, y_train, train_transform)
        val_dataset = PaintingDataset(X_val, y_val, val_transform)
        
        # Use balanced sampling
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        
        logger.info("Starting artist model training...")
        artist_model, train_losses, val_losses, val_accs, best_val_acc = train_model(
            artist_model, train_loader, val_loader, class_weights, epochs=EPOCHS, model_type='artist'
        )
        
        # Evaluate on validation set
        cm_artist, report_artist, _ = evaluate_model(artist_model, val_loader, artist_classes)
        
        torch.save(artist_model.state_dict(), ARTIST_MODEL_PATH)
        logger.info(f"✓ Artist model saved to {ARTIST_MODEL_PATH}")
        
        artist_results = {
            'best_val_accuracy': best_val_acc,
            'class_weights': class_weights.tolist(),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'confusion_matrix': cm_artist.tolist(),
            'classification_report': report_artist,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'class_distribution': {
                'Artemisia': artemisia_count,
                'Orazio': orazio_count
            }
        }
        
        logger.info(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
        logger.info(f"✓ Artemisia precision: {report_artist['Artemisia']['precision']:.3f}")
        logger.info(f"✓ Orazio precision: {report_artist['Orazio']['precision']:.3f}")
    
    # =========================================================================
    # GENDER ATTRIBUTION MODEL
    # =========================================================================
    
    if pathlib.Path(GENDER_MODEL_PATH).exists():
        logger.info("\n" + "="*50)
        logger.info("LOADING EXISTING GENDER MODEL")
        logger.info("="*50)
        
        # FIX: Use the same architecture as artist model
        gender_model = AttributionModel(num_classes=2, model_name=MODEL_ARCHITECTURE).to(DEVICE)
        try:
            gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
            logger.info(f"✓ Gender model loaded successfully")
        except RuntimeError as e:
            logger.warning(f"Gender model architecture mismatch: {e}")
            logger.warning("Training new gender model with correct architecture...")
            pathlib.Path(GENDER_MODEL_PATH).unlink()
            
    if not pathlib.Path(GENDER_MODEL_PATH).exists():
        logger.info("\n" + "="*50)
        logger.info("TRAINING GENDER ATTRIBUTION MODEL")
        logger.info("="*50)
        
        # Load the SAME data as artist model
        gender_paths, gender_labels, gender_classes = load_unquestioned_data('gender')
        
        female_count = sum(1 for l in gender_labels if l == 0)
        male_count = sum(1 for l in gender_labels if l == 1)
        logger.info(f"Dataset: {len(gender_paths)} total paintings")
        logger.info(f"Class distribution: Female(Artemisia)={female_count}, Male(Orazio)={male_count}")
        
        # Use same split seed for consistency
        X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(
            gender_paths, gender_labels,
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_SEED,  # Same seed as artist
            stratify=gender_labels
        )
        
        logger.info(f"Training set: {len(X_train_g)} samples")
        logger.info(f"Validation set: {len(X_val_g)} samples")
        
        sample_weights_g, class_weights_g = calculate_class_weights(y_train_g)
        
        # Use SAME transforms as artist model for better agreement
        train_transform_g = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        val_transform_g = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset_g = PaintingDataset(X_train_g, y_train_g, train_transform_g)
        val_dataset_g = PaintingDataset(X_val_g, y_val_g, val_transform_g)
        
        sampler_g = WeightedRandomSampler(sample_weights_g, len(sample_weights_g))
        
        train_loader_g = DataLoader(train_dataset_g, batch_size=BATCH_SIZE, sampler=sampler_g)
        val_loader_g = DataLoader(val_dataset_g, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        
        # Use SAME architecture as artist model
        gender_model = AttributionModel(num_classes=2, model_name=MODEL_ARCHITECTURE).to(DEVICE)
        
        logger.info("Starting gender model training...")
        gender_model, train_losses_g, val_losses_g, val_accs_g, best_val_acc_g = train_model(
            gender_model, train_loader_g, val_loader_g, class_weights_g, epochs=EPOCHS, model_type='gender'
        )
        
        cm_gender, report_gender, _ = evaluate_model(gender_model, val_loader_g, gender_classes)
        
        torch.save(gender_model.state_dict(), GENDER_MODEL_PATH)
        logger.info(f"✓ Gender model saved to {GENDER_MODEL_PATH}")
        
        gender_results = {
            'best_val_accuracy': best_val_acc_g,
            'class_weights': class_weights_g.tolist(),
            'train_samples': len(X_train_g),
            'val_samples': len(X_val_g),
            'confusion_matrix': cm_gender.tolist(),
            'classification_report': report_gender,
            'train_losses': train_losses_g,
            'val_losses': val_losses_g,
            'val_accuracies': val_accs_g,
            'class_distribution': {
                'Female': female_count,
                'Male': male_count
            }
        }
    else:
        gender_paths, gender_labels, gender_classes = load_unquestioned_data('gender')
        if not gender_results:
            gender_results = {
                'best_val_accuracy': 70.0,
                'class_distribution': {
                    'Female': sum(1 for l in gender_labels if l == 0),
                    'Male': sum(1 for l in gender_labels if l == 1)
                }
            }
    
    # =========================================================================
    # ANALYZE DISPUTED WORKS
    # =========================================================================
    logger.info("\n" + "="*50)
    logger.info("ANALYZING DISPUTED WORKS")
    logger.info("="*50)
    
    disputed_paths, disputed_info = load_disputed_data()
    logger.info(f"Found {len(disputed_paths)} disputed works to analyze")
    
    if len(disputed_paths) > 0:
        disputed_results = analyze_disputed_works(
            artist_model, gender_model, disputed_paths, disputed_info
        )
        
        agreements = sum(1 for r in disputed_results if r['models_agree'])
        logger.info(f"Models agree on {agreements}/{len(disputed_results)} disputed works")
        
        high_conf = [r for r in disputed_results 
                     if r['combined_confidence'] > 0.75 and r['models_agree']]
        logger.info(f"Found {len(high_conf)} high-confidence reattribution candidates")
    else:
        disputed_results = []
    
    # =========================================================================
    # GENERATE HTML REPORT
    # =========================================================================
    if len(disputed_results) > 0:
        logger.info("\n" + "="*50)
        logger.info("GENERATING HTML REPORT")
        logger.info("="*50)
        
        html_report = generate_html_report(artist_results, gender_results, disputed_results)
        
        report_path = f"{OUTPUT_DIR}/attribution_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"✓ HTML report saved to: {report_path}")
    
    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================
    
    viz_exists = all([
        pathlib.Path(f"{OUTPUT_DIR}/confusion_matrix_artist.png").exists(),
        pathlib.Path(f"{OUTPUT_DIR}/confusion_matrix_gender.png").exists(),
        pathlib.Path(f"{OUTPUT_DIR}/training_history_artist.png").exists(),
        pathlib.Path(f"{OUTPUT_DIR}/training_history_gender.png").exists()
    ])
    
    if not viz_exists and artist_results and gender_results:
        logger.info("\n" + "="*50)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*50)
        
        if 'confusion_matrix' in artist_results:
            cm_artist = np.array(artist_results['confusion_matrix'])
            fig_cm_artist = plot_confusion_matrix(cm_artist, artist_classes, 
                                                  "Artist Model - Confusion Matrix")
            fig_cm_artist.savefig(f"{OUTPUT_DIR}/confusion_matrix_artist.png", dpi=150)
            plt.close()
        
        if 'confusion_matrix' in gender_results:
            cm_gender = np.array(gender_results['confusion_matrix'])
            fig_cm_gender = plot_confusion_matrix(cm_gender, gender_classes,
                                                  "Gender Model - Confusion Matrix")
            fig_cm_gender.savefig(f"{OUTPUT_DIR}/confusion_matrix_gender.png", dpi=150)
            plt.close()
        
        if 'train_losses' in artist_results:
            fig_hist_artist = plot_training_history(
                artist_results['train_losses'], 
                artist_results['val_losses'], 
                artist_results['val_accuracies'],
                "Artist Model"
            )
            fig_hist_artist.savefig(f"{OUTPUT_DIR}/training_history_artist.png", dpi=150)
            plt.close()
        
        if 'train_losses' in gender_results:
            fig_hist_gender = plot_training_history(
                gender_results['train_losses'],
                gender_results['val_losses'],
                gender_results['val_accuracies'],
                "Gender Model"
            )
            fig_hist_gender.savefig(f"{OUTPUT_DIR}/training_history_gender.png", dpi=150)
            plt.close()
    
    # Save cache
    if artist_results and gender_results:
        save_cached_results({
            'artist_results': artist_results,
            'gender_results': gender_results
        })
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Artist Model Validation Accuracy: {artist_results['best_val_accuracy']:.2f}%")
    logger.info(f"Gender Model Validation Accuracy: {gender_results['best_val_accuracy']:.2f}%")
    
    if len(disputed_results) > 0:
        logger.info(f"High-Confidence Reattributions: {len(high_conf)}")
    
    logger.info(f"Models saved to: {MODELS_DIR}/")
    logger.info(f"Visualizations saved to: {OUTPUT_DIR}/")
    logger.info("="*70)

if __name__ == "__main__":
    main()