import torch

class Config:
    # Data settings
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Training settings
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Mutual learning settings
    MUTUAL_LOSS_WEIGHT = 0.5  # lambda in paper
    MUTUAL_LAYER_POSITIONS = ['early', 'middle', 'late']  # all positions
    
    # Model settings - CNN branch
    CNN_BACKBONE = 'resnet50'  # or 'efficientnet_b3'
    CNN_PRETRAINED = True
    
    # Model settings - Transformer branch
    TRANSFORMER_MODEL = 'vit_base_patch16_224'
    TRANSFORMER_PRETRAINED = True
    
    # Model settings - Mamba branch
    MAMBA_D_MODEL = 512
    MAMBA_N_LAYERS = 4
    MAMBA_D_STATE = 16
    
    # Fusion settings
    FUSION_HIDDEN_DIM = 512
    FUSION_DROPOUT = 0.2
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_GPUS = 2 if torch.cuda.device_count() > 1 else 1
    
    # Dataset paths (update these)
    CELEB_DF_PATH = './data/Celeb-DF'
    FFPP_PATH = './data/FaceForensics++'
    
    # Output
    SAVE_DIR = './checkpoints'
    LOG_DIR = './logs'
    VISUALIZATION_DIR = './visualizations'