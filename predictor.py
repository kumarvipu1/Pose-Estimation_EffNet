import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torchvision import transforms, utils
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_rmse,
)

model_15 = EfficientNet.from_pretrained("efficientnet-b0")
model_15._fc = nn.Linear(1280, 42)
model_15 = model_15.to(config.DEVICE)
optimizer = optim.Adam(model_15.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
    load_checkpoint(torch.load("checkpoint/b0_4_v2.pth.tar"), model_15, optimizer, config.LEARNING_RATE)

image = "C:/Users/44745/Desktop/Pose-Estimation/0010.jpg"
preds_15 = torch.clip(model_15(image).squeeze(0), 0.0, 224)
