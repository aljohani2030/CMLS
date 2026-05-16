import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from PIL import Image

class DeepfakeDataset(Dataset):
    """Dataset loader for deepfake detection"""
    def __init__(self, root_dir, split='train', transform=None, frame_limit=50):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.frame_limit = frame_limit
        
        # Load file paths and labels
        self.video_paths = []
        self.labels = []
        
        # Real videos
        real_dir = os.path.join(root_dir, split, 'real')
        if os.path.exists(real_dir):
            for video in os.listdir(real_dir):
                self.video_paths.append(os.path.join(real_dir, video))
                self.labels.append(0)  # real
        
        # Fake videos
        fake_dir = os.path.join(root_dir, split, 'fake')
        if os.path.exists(fake_dir):
            for video in os.listdir(fake_dir):
                self.video_paths.append(os.path.join(fake_dir, video))
                self.labels.append(1)  # fake
        
        print(f"Loaded {len(self.video_paths)} {split} samples")
    
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.frame_limit:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            # Return blank image if no frames
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
        # Sample central frame for image-based detection
        # (For temporal detection, you would sample multiple frames)
        center_frame = frames[len(frames) // 2]
        
        # Apply transforms
        if self.transform:
            image = self.transform(Image.fromarray(center_frame))
        else:
            image = torch.from_numpy(center_frame).permute(2, 0, 1).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(config):
    """Create data loaders for training and validation"""
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DeepfakeDataset(
        root_dir=config.CELEB_DF_PATH,  # or FFPP_PATH
        split='train',
        transform=train_transform
    )
    
    val_dataset = DeepfakeDataset(
        root_dir=config.CELEB_DF_PATH,
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader