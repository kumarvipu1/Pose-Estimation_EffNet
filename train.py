import torch
from dataset import LoadData
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
    get_submission
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    num_examples = 0
    for batch_idx, sample_batched in enumerate(loop):
        data = sample_batched['image']
        data = data.double()
        data = data.to(device=device)
        targets = sample_batched['keypoints']
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss average over epoch: {(sum(losses)/num_examples)**0.5}")


def main():
    train_ds = LoadData(
        root_dir= 'C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/train_data/images',
        annotation_file='C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/train_data/annotation_train.pickle',
        transform=transforms.Compose([
                                       config.Rescale(256),
                                       config.RandomCrop(224),
                                       config.ToTensor()
                                   ]))
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    val_ds = LoadData(
        root_dir='C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/val_data/images',
        transform=transforms.Compose([
            config.Rescale(256),
            config.RandomCrop(224),
            config.ToTensor()
        ]),
        annotation_file='C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/val_data/annotation_val.pickle',
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    test_ds = LoadData(
        root_dir='C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/val_data/images',
        annotation_file='C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/val_data/annotation_val.pickle',
        transform=transforms.Compose([
            config.Rescale(256),
            config.RandomCrop(224),
            config.ToTensor()
        ]),
        train=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    loss_fn = nn.MSELoss(reduction="sum")
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(1280, 42)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    '''
    model_4 = EfficientNet.from_pretrained("efficientnet-b0")
    model_4._fc = nn.Linear(1280, 42)
    model_15 = EfficientNet.from_pretrained("efficientnet-b0")
    model_15._fc = nn.Linear(1280, 42)
    model_4 = model_4.to(config.DEVICE)
    model_15 = model_15.to(config.DEVICE)

    
    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)
        load_checkpoint(torch.load("b0_4.pth.tar"), model_4, optimizer, config.LEARNING_RATE)
        load_checkpoint(torch.load("b0_15.pth.tar"), model_15, optimizer, config.LEARNING_RATE)

    get_submission(test_loader, test_ds, model_15, model_4)
    '''

    for epoch in range(config.NUM_EPOCHS):
        #get_rmse(val_loader, model, loss_fn, config.DEVICE)
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # get on validation
        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

if __name__ == "__main__":
    main()
