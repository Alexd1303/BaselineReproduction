import argparse
from pathlib import Path
from os import cpu_count
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from torchvision.datasets import ImageNet
from torchvision.transforms import v2

from torchmetrics import Accuracy

from Models import MARNetImageNetWrapper
from Utils import Checkpoint

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--checkpoints_dir", type=Path, required=True, help="Path to save checkpoints")
    args = parser.parse_args()
    
    checkpoints_dir = args.checkpoints_dir / "MARNet_ImageNet"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_Acc = 0.0
    if Path(checkpoints_dir / "MARNet_ImageNet_best.pt").is_file():
        best_checkpoint = Checkpoint()
        best_checkpoint.load_from_file(checkpoints_dir / "MARNet_ImageNet_best.pt")
        best_val_Acc = best_checkpoint.val_mAcc if best_checkpoint.val_mAcc is not None else 0.0
    
    transform = v2.Compose([
        v2.RandomResizedCrop((224, 224)),
        v2.RandomHorizontalFlip(),
        
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = v2.Compose([
        v2.CenterCrop((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageNet(root=args.dataset_dir, split="train", transform=transform)
    val_dataset = ImageNet(root=args.dataset_dir, split="val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count(), pin_memory=True, persistent_workers=True) # type: ignore
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=True, persistent_workers=True) # type: ignore
    
    model = MARNetImageNetWrapper().to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_accuracy = Accuracy(task="multiclass", num_classes=1000).to(device)
    val_accuracy = Accuracy(task="multiclass", num_classes=1000).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        total_iters=5
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - 5,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[5]
    )
    
    scaler = torch.amp.grad_scaler.GradScaler()
    
    for epoch in range(args.epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()
            train_accuracy.update(outputs, labels)
        
        print(f"Epoch {epoch + 1}/{args.epochs} - Train Accuracy: {train_accuracy.compute().item() * 100:.2f}%")
        train_accuracy.reset()

        model.eval()
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            val_accuracy.update(outputs, labels)
            
        print(f"Epoch {epoch + 1}/{args.epochs} - Validation Accuracy: {val_accuracy.compute().item() * 100:.2f}%")
        val_accuracy.reset()
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = checkpoints_dir / f"MARNet_ImageNet_{epoch % 2}.pt"
        
        checkpoint = Checkpoint()
        checkpoint.state_dict = model.state_dict()
        checkpoint.optimizer_state_dict = optimizer.state_dict()
        checkpoint.scheduler_state_dict = scheduler.state_dict()
        checkpoint.epoch = epoch
        checkpoint.name = "MARNet_ImageNet"
        checkpoint.val_mAcc = val_accuracy.compute().item()

        torch.save(checkpoint.asdict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        
        # Save best checkpoint based on mAcc
        if val_accuracy.compute().item() > best_val_Acc:
            best_val_Acc = val_accuracy.compute().item()
            best_checkpoint_path = checkpoints_dir / "MARNet_ImageNet_best.pt"
            torch.save(checkpoint.asdict(), best_checkpoint_path)
            print(f"Best checkpoint updated at {best_checkpoint_path}")
