import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from tqdm import tqdm

from Data import CarDataset
from BaselineNet import BaselineNet

from Utils import Checkpoint

def train(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: ReduceLROnPlateau, train_dataset: CarDataset, val_dataset: CarDataset, checkpoints_dir: Path, name: str,start_epoch: int=0, num_epochs: int=10, batch_size: int=24, device: torch.device = torch.device("cpu")):
    
    checkpoints_dir = checkpoints_dir / name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    accuracy1 = Accuracy(task="multiclass", num_classes=5).to(device)
    accuracy2 = Accuracy(task="multiclass", num_classes=7).to(device)
    accuracy3 = Accuracy(task="multiclass", num_classes=3).to(device)
    accuracy4 = Accuracy(task="multiclass", num_classes=5).to(device)
    
    best_val_mAcc = 0.0
    if Path(checkpoints_dir / f"{name}_best.pt").is_file():
        best_checkpoint = Checkpoint()
        best_checkpoint.load_from_file(checkpoints_dir / f"{name}_best.pt")
        best_val_mAcc = best_checkpoint.val_mAcc if best_checkpoint.val_mAcc is not None else 0.0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    
    model.to(device)
    lossDER = nn.CrossEntropyLoss()
    lossDBR = nn.CrossEntropyLoss()
    lossTCR = nn.CrossEntropyLoss()
    lossVBR = nn.CrossEntropyLoss()
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for img1,img2,img3,img4,face,body, posture, gesture,emotion_label, behavior_label, context_label, vehicle_label in tqdm(train_loader, desc=f"Training"):
            img1 = img1.view(-1, 48, 224, 224).to(device, non_blocking=True)
            img2 = img2.view(-1, 48, 224, 224).to(device, non_blocking=True)
            img3 = img3.view(-1, 48, 224, 224).to(device, non_blocking=True)
            img4 = img4.view(-1, 48, 224, 224).to(device, non_blocking=True)
            face = face.view(-1, 48, 224, 224).to(device, non_blocking=True)
            body = body.view(-1, 48, 224, 224).to(device, non_blocking=True)
            gesture = gesture.view(batch_size, 3, 16, 42, 1).to(device, non_blocking=True)
            posture = posture.view(batch_size, 3, 16, 26, 1).to(device, non_blocking=True)
            emotion_label = emotion_label.to(device, non_blocking=True)
            behavior_label = behavior_label.to(device, non_blocking=True)
            context_label = context_label.to(device, non_blocking=True)
            vehicle_label = vehicle_label.to(device, non_blocking=True)

            model.zero_grad()
            out1, out2, out3, out4 = model(img1, img2, img3, img4, face, body, gesture, posture)
            
            accuracy1.update(out1, emotion_label)
            accuracy2.update(out2, behavior_label)
            accuracy3.update(out3, context_label)
            accuracy4.update(out4, vehicle_label)
            
            loss1 = lossDER(out1, emotion_label)
            loss2 = lossDBR(out2, behavior_label)
            loss3 = lossTCR(out3, context_label)
            loss4 = lossVBR(out4, vehicle_label)
            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} - Accuracy 1: {accuracy1.compute():.4f}, Accuracy 2: {accuracy2.compute():.4f}, Accuracy 3: {accuracy3.compute():.4f}, Accuracy 4: {accuracy4.compute():.4f}, mAcc: {(accuracy1.compute() + accuracy2.compute() + accuracy3.compute() + accuracy4.compute()) / 4:.4f}")
        accuracy1.reset()
        accuracy2.reset()
        accuracy3.reset()
        accuracy4.reset()
        
        val_loss1 = []
        val_loss2 = []
        val_loss3 = []
        val_loss4 = []
        val_mLoss = []
        
        model.eval()
        with torch.no_grad():
            for img1,img2,img3,img4,face,body, posture, gesture,emotion_label, behavior_label, context_label, vehicle_label in tqdm(val_loader, desc=f"Validation"):
                try:
                    img1 = img1.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    img2 = img2.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    img3 = img3.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    img4 = img4.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    face = face.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    body = body.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    gesture = gesture.view(batch_size, 3, 16, 42, 1).to(device, non_blocking=True)
                    posture = posture.view(batch_size, 3, 16, 26, 1).to(device, non_blocking=True)
                    emotion_label = emotion_label.to(device, non_blocking=True)
                    behavior_label = behavior_label.to(device, non_blocking=True)
                    context_label = context_label.to(device, non_blocking=True)
                    vehicle_label = vehicle_label.to(device, non_blocking=True)

                    out1_val, out2_val, out3_val, out4_val = model(img1, img2, img3, img4, face, body, gesture ,posture)

                    accuracy1.update(out1_val, emotion_label)
                    accuracy2.update(out2_val, behavior_label)
                    accuracy3.update(out3_val, context_label)
                    accuracy4.update(out4_val, vehicle_label)
                    
                    val_loss1.append(lossDER(out1_val, emotion_label).item())
                    val_loss2.append(lossDBR(out2_val, behavior_label).item())
                    val_loss3.append(lossTCR(out3_val, context_label).item())
                    val_loss4.append(lossVBR(out4_val, vehicle_label).item())
                    val_mLoss.append((lossDER(out1_val, emotion_label).item() + lossDBR(out2_val, behavior_label).item() + lossTCR(out3_val, context_label).item() + lossVBR(out4_val, vehicle_label).item()) / 4)
                except Exception as e:
                    pass
        
        mAcc = (accuracy1.compute() + accuracy2.compute() + accuracy3.compute() + accuracy4.compute()) / 4
        print(f"Validation - Accuracy 1: {accuracy1.compute():.4f}, Accuracy 2: {accuracy2.compute():.4f}, Accuracy 3: {accuracy3.compute():.4f}, Accuracy 4: {accuracy4.compute():.4f}, mAcc: {mAcc:.4f}")
        
        scheduler.step(sum(val_mLoss) / len(val_mLoss))

        # Save checkpoint
        checkpoint_path = checkpoints_dir / f"{name}_{epoch % 2}.pt"
        
        checkpoint = Checkpoint()
        checkpoint.state_dict = model.state_dict()
        checkpoint.optimizer_state_dict = optimizer.state_dict()
        checkpoint.scheduler_state_dict = scheduler.state_dict()
        checkpoint.epoch = epoch
        checkpoint.name = name
        checkpoint.val_accuracy1 = accuracy1.compute().item()
        checkpoint.val_accuracy2 = accuracy2.compute().item()
        checkpoint.val_accuracy3 = accuracy3.compute().item()
        checkpoint.val_accuracy4 = accuracy4.compute().item()
        checkpoint.val_mAcc = mAcc.item()
        checkpoint.val_mLoss = sum(val_mLoss) / len(val_mLoss)

        torch.save(checkpoint.asdict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        
        # Save best checkpoint based on mAcc
        if mAcc > best_val_mAcc:
            best_val_mAcc = mAcc
            best_checkpoint_path = checkpoints_dir / f"{name}_best.pt"
            torch.save(checkpoint.asdict(), best_checkpoint_path)
            print(f"Best checkpoint updated at {best_checkpoint_path}")
        
        accuracy1.reset()
        accuracy2.reset()
        accuracy3.reset()
        accuracy4.reset()

def test(model: nn.Module, test_dataset: CarDataset, checkpoints_dir: Path, name: str, batch_size: int=24, device: torch.device = torch.device("cpu")):
    
    checkpoints_dir = checkpoints_dir / name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    
    accuracy1 = Accuracy(task="multiclass", num_classes=5).to(device)
    accuracy2 = Accuracy(task="multiclass", num_classes=7).to(device)
    accuracy3 = Accuracy(task="multiclass", num_classes=3).to(device)
    accuracy4 = Accuracy(task="multiclass", num_classes=5).to(device)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for img1,img2,img3,img4,face,body, posture, gesture,emotion_label, behavior_label, context_label, vehicle_label in tqdm(test_loader, desc=f"Testing"):
            img1 = img1.view(-1, 48, 224, 224).to(device, non_blocking=True)
            img2 = img2.view(-1, 48, 224, 224).to(device, non_blocking=True)
            img3 = img3.view(-1, 48, 224, 224).to(device, non_blocking=True)
            img4 = img4.view(-1, 48, 224, 224).to(device, non_blocking=True)
            face = face.view(-1, 48, 224, 224).to(device, non_blocking=True)
            body = body.view(-1, 48, 224, 224).to(device, non_blocking=True)
            gesture = gesture.view(batch_size, 3, 16, 42, 1).to(device, non_blocking=True)
            posture = posture.view(batch_size, 3, 16, 26, 1).to(device, non_blocking=True)
            emotion_label = emotion_label.to(device, non_blocking=True)
            behavior_label = behavior_label.to(device, non_blocking=True)
            context_label = context_label.to(device, non_blocking=True)
            vehicle_label = vehicle_label.to(device, non_blocking=True)

            out1_test , out2_test , out3_test , out4_test = model(img1,img2,img3,img4 ,face ,body , gesture ,posture)
            
            accuracy1.update(out1_test, emotion_label)
            accuracy2.update(out2_test, behavior_label)
            accuracy3.update(out3_test, context_label)
            accuracy4.update(out4_test, vehicle_label)
            
    mAcc = (accuracy1.compute() + accuracy2.compute() + accuracy3.compute() + accuracy4.compute()) / 4
    print(f"Test - Accuracy 1: {accuracy1.compute():.4f}, Accuracy 2: {accuracy2.compute():.4f}, Accuracy 3: {accuracy3.compute():.4f}, Accuracy 4: {accuracy4.compute():.4f}, mAcc: {mAcc:.4f}")
    
    with open(checkpoints_dir / f"{name}_test_results.txt", "w") as f:
        f.write(f"Test - Accuracy 1: {accuracy1.compute():.4f}, Accuracy 2: {accuracy2.compute():.4f}, Accuracy 3: {accuracy3.compute():.4f}, Accuracy 4: {accuracy4.compute():.4f}, mAcc: {mAcc:.4f}\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=Path, default="./checkpoints", help="Path to the checkpoints directory")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to the dataset directory")
    parser.add_argument("--split_dir", type=Path, required=True, help="Path to the split directory")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to be used")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--start_from_checkpoint", type=Path, default=None, help="Path to a checkpoint to resume training from")

    args = parser.parse_args()

    if not args.split_dir.exists():
        print(f"Split directory {args.split_dir} does not exist. Please provide a valid path.")
        exit(1)

    if not args.dataset_dir.exists():
        print(f"Dataset directory {args.dataset_dir} does not exist. Please provide a valid path.")
        exit(1)
        
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    
    train_dataset = CarDataset(csv_file=args.split_dir / "training.csv", dataset_root=args.dataset_dir, horizontal_flip_prob=0.5, vertical_flip_prob=0.5)
    val_dataset = CarDataset(csv_file=args.split_dir / "validation.csv", dataset_root=args.dataset_dir, horizontal_flip_prob=0.0, vertical_flip_prob=0.0)
    
    model = BaselineNet()
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    if args.start_from_checkpoint is not None:
        checkpoint = torch.load(args.start_from_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
        
    train(model, optimizer, scheduler, train_dataset, val_dataset, args.checkpoints_dir, name=args.model_name, start_epoch=start_epoch, num_epochs=args.num_epochs, batch_size=args.batch_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    test_dataset = CarDataset(csv_file=args.split_dir / "testing.csv", dataset_root=args.dataset_dir, horizontal_flip_prob=0.0, vertical_flip_prob=0.0)
    
    best_checkpoint_path = args.checkpoints_dir / args.model_name / f"{args.model_name}_best.pt"
    
    best_checkpoint = Checkpoint()
    best_checkpoint.load_from_file(best_checkpoint_path)
    
    if best_checkpoint.state_dict is None:
        raise ValueError(f"The best checkpoint at {best_checkpoint_path} does not contain a valid state_dict.")
    
    model.load_state_dict(best_checkpoint.state_dict)
    
    test(model, test_dataset, args.checkpoints_dir, name=args.model_name, batch_size=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))