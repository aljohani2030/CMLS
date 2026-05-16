import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb  # optional
from config import Config
from models.cmls_model import CMLSModel
from data.dataset import get_data_loaders

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_mutual_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            # Forward pass
            logits, branch_features = model(images, return_features=True)
            
            # Compute collaborative loss
            loss, class_loss, mutual_loss = model.compute_loss(
                logits, branch_features, targets
            )
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_mutual_loss += mutual_loss.item()
        
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100 * correct / total
        })
    
    return {
        'loss': total_loss / len(loader),
        'class_loss': total_class_loss / len(loader),
        'mutual_loss': total_mutual_loss / len(loader),
        'accuracy': 100 * correct / total
    }


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)
            
            logits, branch_features = model(images, return_features=True)
            loss, _, _ = model.compute_loss(logits, branch_features, targets)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': 100 * correct / total,
        'precision': precision_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'auc': roc_auc_score(all_targets, all_preds)
    }


def main():
    config = Config()
    device = config.DEVICE
    
    # Initialize model
    model = CMLSModel(config)
    
    # Data parallel if multiple GPUs
    if config.NUM_GPUS > 1:
        model = torch.nn.Data