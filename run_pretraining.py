import argparse
from pathlib import Path
from os import cpu_count
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from torchvision.datasets import ImageNet, UCF101
from torchvision.transforms import v2

from torchmetrics import Accuracy

from Models import MARNetPretrainingWrapper
from Utils import Checkpoint

def init_ImageNet(dataset_dir: Path, annotation_dir: Path, batch_size: int, num_workers: int, device: torch.device):
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

    train_dataset = ImageNet(root=dataset_dir, split="train", transform=transform)
    val_dataset = ImageNet(root=dataset_dir, split="val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = MARNetPretrainingWrapper(input_channels=3, num_classes=1000).to(device)
    
    def preprocess_batch(batch):
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return images, labels
    
    return train_loader, val_loader, model, preprocess_batch

def ucf101_collate(batch):
        videos, _, labels = zip(*batch)

        videos = torch.stack(videos, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return videos, labels

def init_UCF101(dataset_dir: Path, annotation_path: Path, batch_size: int, num_workers: int, device: torch.device):
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

    train_dataset = UCF101(root=dataset_dir / "train", annotation_path=str(annotation_path), train=True, transform=transform, frames_per_clip=16, step_between_clips=1, frame_rate=16, output_format="TCHW", num_workers=num_workers)
    val_dataset = UCF101(root=dataset_dir / "val", annotation_path=str(annotation_path), train=False, transform=val_transform, frames_per_clip=16, step_between_clips=1, frame_rate=16, output_format="TCHW", num_workers=num_workers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=ucf101_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=ucf101_collate)

    model = MARNetPretrainingWrapper(input_channels=48, num_classes=101).to(device)
    
    def preprocess_batch(batch):
        frames, labels = batch
        
        # change shape from (B, T, C, H, W) to (B, T * C, H, W) for MARNet
        B, T, C, H, W = frames.shape
        frames = frames.reshape(B, T * C, H, W)
        
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return frames, labels

    return train_loader, val_loader, model, preprocess_batch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    available_datasets = {"ImageNet": init_ImageNet, "UCF101": init_UCF101}
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to the data")
    parser.add_argument("--annotation_dir", type=Path, required=True, help="Path to the annotations")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--checkpoints_dir", type=Path, required=True, help="Path to save checkpoints")
    parser.add_argument("--dataset", type=str, choices=available_datasets.keys(), required=True, help="Dataset to use")
    args = parser.parse_args()
    
    checkpoints_dir = args.checkpoints_dir / f"MARNet_{args.dataset}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_Acc = 0.0
    if Path(checkpoints_dir / f"MARNet_{args.dataset}_best.pt").is_file():
        best_checkpoint = Checkpoint()
        best_checkpoint.load_from_file(checkpoints_dir / f"MARNet_{args.dataset}_best.pt")
        best_val_Acc = best_checkpoint.val_mAcc if best_checkpoint.val_mAcc is not None else 0.0
    
    train_loader, val_loader, model, preprocess_batch = available_datasets[args.dataset](args.dataset_dir, args.annotation_dir, args.batch_size, cpu_count(), device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_accuracy = Accuracy(task="multiclass", num_classes=model.num_classes).to(device)
    val_accuracy = Accuracy(task="multiclass", num_classes=model.num_classes).to(device)
    
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
    
    scaler = GradScaler()
    
    for epoch in range(args.epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, labels = preprocess_batch((images, labels))

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
            images, labels = preprocess_batch((images, labels))

            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            val_accuracy.update(outputs, labels)
            
        print(f"Epoch {epoch + 1}/{args.epochs} - Validation Accuracy: {val_accuracy.compute().item() * 100:.2f}%")
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = checkpoints_dir / f"MARNet_{args.dataset}_{epoch % 2}.pt"
        
        checkpoint = Checkpoint()
        checkpoint.state_dict = model.state_dict()
        checkpoint.optimizer_state_dict = optimizer.state_dict()
        checkpoint.scheduler_state_dict = scheduler.state_dict()
        checkpoint.epoch = epoch
        checkpoint.name = f"MARNet_{args.dataset}"
        checkpoint.val_mAcc = val_accuracy.compute().item()

        torch.save(checkpoint.asdict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        
        # Save best checkpoint based on mAcc
        if val_accuracy.compute().item() > best_val_Acc:
            best_val_Acc = val_accuracy.compute().item()
            best_checkpoint_path = checkpoints_dir / f"MARNet_{args.dataset}_best.pt"
            torch.save(checkpoint.asdict(), best_checkpoint_path)
            print(f"Best checkpoint updated at {best_checkpoint_path}")
        
        val_accuracy.reset()
