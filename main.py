import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from torchmetrics import Accuracy, ConfusionMatrix

from Data import CarDataset, EMOTION_LABEL, DRIVER_BEHAVIOR_LABEL, SCENE_CENTRIC_CONTEXT_LABEL, VEHICLE_BASED_CONTEXT_LABEL

from Utils import Checkpoint, EarlyStopper, plot_confusion_matrix

def train(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: SequentialLR, train_dataset: CarDataset, val_dataset: CarDataset, checkpoints_dir: Path, name: str,start_epoch: int=0, num_epochs: int=10, batch_size: int=24, device: torch.device = torch.device("cpu"), num_workers: int=4):
    
    checkpoints_dir = checkpoints_dir / name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    accuracy1 = Accuracy(task="multiclass", num_classes=5).to(device)
    accuracy2 = Accuracy(task="multiclass", num_classes=7).to(device)
    accuracy3 = Accuracy(task="multiclass", num_classes=3).to(device)
    accuracy4 = Accuracy(task="multiclass", num_classes=5).to(device)
    
    early_stopper = EarlyStopper(patience=25, min_delta=0.001)
    
    best_val_mAcc = 0.0
    if Path(checkpoints_dir / f"{name}_best.pt").is_file():
        best_checkpoint = Checkpoint()
        best_checkpoint.load_from_file(checkpoints_dir / f"{name}_best.pt")
        best_val_mAcc = best_checkpoint.val_mAcc if best_checkpoint.val_mAcc is not None else 0.0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    model.to(device)
    lossDER = nn.CrossEntropyLoss(label_smoothing=0.1)
    lossDBR = nn.CrossEntropyLoss(label_smoothing=0.1)
    lossTCR = nn.CrossEntropyLoss(label_smoothing=0.1)
    lossVBR = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    grad_norms = []
    val_mAccs = []
    
    for epoch in range(start_epoch, num_epochs):
        train_mLoss = []
        
        model.train()
        for inside, front, left, right, face, body, posture, gesture, emotion_label, behavior_label, context_label, vehicle_label in tqdm(train_loader, desc=f"Training"):
            inside = inside.view(-1, 48, 224, 224).to(device, non_blocking=True)
            front = front.view(-1, 48, 224, 224).to(device, non_blocking=True)
            left = left.view(-1, 48, 224, 224).to(device, non_blocking=True)
            right = right.view(-1, 48, 224, 224).to(device, non_blocking=True)
            face = face.view(-1, 48, 224, 224).to(device, non_blocking=True)
            body = body.view(-1, 48, 224, 224).to(device, non_blocking=True)
            #gesture = gesture.view(batch_size, 3, 16, 42, 1).to(device, non_blocking=True)
            #posture = posture.view(batch_size, 3, 16, 26, 1).to(device, non_blocking=True)
            gesture = gesture.permute(0, 3, 1, 2).unsqueeze(-1).contiguous().to(device, non_blocking=True)
            posture = posture.permute(0, 3, 1, 2).unsqueeze(-1).contiguous().to(device, non_blocking=True)
            emotion_label = emotion_label.to(device, non_blocking=True)
            behavior_label = behavior_label.to(device, non_blocking=True)
            context_label = context_label.to(device, non_blocking=True)
            vehicle_label = vehicle_label.to(device, non_blocking=True)

            model.zero_grad()
            out1, out2, out3, out4 = model(inside, front, left, right, face, body, gesture, posture)
            
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
            grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True))
            optimizer.step()
            
            train_mLoss.append(loss.item())

        mAcc = (accuracy1.compute() + accuracy2.compute() + accuracy3.compute() + accuracy4.compute()) / 4
        train_mLoss = sum(train_mLoss) / len(train_mLoss) if train_mLoss else float('inf')
        log_string = f"Epoch {epoch + 1}/{num_epochs} - Accuracy 1: {accuracy1.compute():.4f}, Accuracy 2: {accuracy2.compute():.4f}, Accuracy 3: {accuracy3.compute():.4f}, Accuracy 4: {accuracy4.compute():.4f}, mAcc: {mAcc:.4f}, train_loss: {train_mLoss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}, grad_norm: {torch.stack(grad_norms).mean():.4f}"
        print(log_string)
        with open(checkpoints_dir / f"{name}_train_log.txt", "a") as f:
            f.write(log_string + "\n")
        accuracy1.reset()
        accuracy2.reset()
        accuracy3.reset()
        accuracy4.reset()
        
        val_mLoss = []
        
        model.eval()
        with torch.no_grad():
            for inside, front, left, right, face, body, posture, gesture, emotion_label, behavior_label, context_label, vehicle_label in tqdm(val_loader, desc=f"Validation"):
                try:
                    inside = inside.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    front = front.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    left = left.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    right = right.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    face = face.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    body = body.view(-1, 48, 224, 224).to(device, non_blocking=True)
                    
                    #gesture = gesture.view(batch_size, 3, 16, 42, 1).to(device, non_blocking=True)
                    #posture = posture.view(batch_size, 3, 16, 26, 1).to(device, non_blocking=True)
                    gesture = gesture.permute(0, 3, 1, 2).unsqueeze(-1).contiguous().to(device, non_blocking=True)
                    posture = posture.permute(0, 3, 1, 2).unsqueeze(-1).contiguous().to(device, non_blocking=True)
                    
                    emotion_label = emotion_label.to(device, non_blocking=True)
                    behavior_label = behavior_label.to(device, non_blocking=True)
                    context_label = context_label.to(device, non_blocking=True)
                    vehicle_label = vehicle_label.to(device, non_blocking=True)

                    out1_val, out2_val, out3_val, out4_val = model(inside, front, left, right, face, body, gesture ,posture)

                    accuracy1.update(out1_val, emotion_label)
                    accuracy2.update(out2_val, behavior_label)
                    accuracy3.update(out3_val, context_label)
                    accuracy4.update(out4_val, vehicle_label)
                    
                    val_mLoss.append((lossDER(out1_val, emotion_label).item() + lossDBR(out2_val, behavior_label).item() + lossTCR(out3_val, context_label).item() + lossVBR(out4_val, vehicle_label).item()) / 4)
                    
                    val_mAccs.append((accuracy1.compute() + accuracy2.compute() + accuracy3.compute() + accuracy4.compute()) / 4)
                except Exception as e:
                    pass
        
        mAcc = (accuracy1.compute() + accuracy2.compute() + accuracy3.compute() + accuracy4.compute()) / 4
        val_mLoss = sum(val_mLoss) / len(val_mLoss) if val_mLoss else float('inf')
        log_string = f"Validation {epoch + 1}/{num_epochs} - Accuracy 1: {accuracy1.compute():.4f}, Accuracy 2: {accuracy2.compute():.4f}, Accuracy 3: {accuracy3.compute():.4f}, Accuracy 4: {accuracy4.compute():.4f}, mAcc: {mAcc:.4f}, val_loss: {val_mLoss:.4f}"
        print(log_string)
        with open(checkpoints_dir / f"{name}_val_log.txt", "a") as f:
            f.write(log_string + "\n")

        scheduler.step()

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
        checkpoint.val_mLoss = val_mLoss

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
        
        if epoch > warmup_epochs and early_stopper.early_stop(checkpoint, val_mLoss):
            print("Early stopping triggered.")
            break

def write_misclassified_frame(frames, index, cnt, task_name, angle, predicted_labels, confidence, true_labels, label_names: list[str], missclassified_dir: Path, frame_n: int, resize_size=(512, 512), position=(2, 408), font=None, color=(255, 0, 255)):
    font = ImageFont.load_default(size=28)
    
    inside_misclassified = frames[index].cpu().numpy().transpose(1, 2, 0)[:,:,frame_n: frame_n - 3 if frame_n - 3 > 0 else None :-1]
    img = Image.fromarray((inside_misclassified * 255).astype(np.uint8)).convert("RGB").resize(resize_size)
    img_d = ImageDraw.Draw(img)
    img_d.text(position, f"Inside\nTrue: {label_names[true_labels[index].item()]}\nPredicted: {label_names[predicted_labels[index].item()]} ({confidence[index].item():.2%})", fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    img.save(missclassified_dir / task_name / str(cnt) / angle / f"{angle}_{cnt}_{index}_{frame_n}.png")
        

def save_misclassified_images(misclassified_indices, missclassified_dir: Path, cnt: int, task_name: str, predicted_labels, confidence, true_labels, label_names: list[str], inside=None, front=None, left=None, right=None, face=None, body=None):
    missclassified_dir.mkdir(parents=True, exist_ok=True)
    (missclassified_dir / task_name).mkdir(parents=True, exist_ok=True)
    
    if misclassified_indices.numel():
        (missclassified_dir / task_name / str(cnt)).mkdir(parents=True, exist_ok=True)
    
    resize_size = (512, 512)
    position = (2, 408) # bottom-left corner
    font = ImageFont.load_default(size=28)
    color = (255, 0, 255)
    
    for i in misclassified_indices:
        if inside is not None:
            (missclassified_dir / task_name / str(cnt) / "inside").mkdir(parents=True, exist_ok=True)
            for frame_n in range(47, 0, -3):
                write_misclassified_frame(inside, i, cnt, task_name, "inside", predicted_labels, confidence, true_labels, label_names, missclassified_dir, frame_n, resize_size=resize_size, position=position, font=font, color=color)
        if front is not None: 
            (missclassified_dir / task_name / str(cnt) / "front").mkdir(parents=True, exist_ok=True)
            for frame_n in range(47, 0, -3):
                write_misclassified_frame(front, i, cnt, task_name, "front", predicted_labels, confidence, true_labels, label_names, missclassified_dir, frame_n, resize_size=resize_size, position=position, font=font, color=color)
        if left is not None:
            (missclassified_dir / task_name / str(cnt) / "left").mkdir(parents=True, exist_ok=True)
            for frame_n in range(47, 0, -3):
                write_misclassified_frame(left, i, cnt, task_name, "left", predicted_labels, confidence, true_labels, label_names, missclassified_dir, frame_n, resize_size=resize_size, position=position, font=font, color=color)
        if right is not None: 
            (missclassified_dir / task_name / str(cnt) / "right").mkdir(parents=True, exist_ok=True)
            for frame_n in range(47, 0, -3):
                write_misclassified_frame(right, i, cnt, task_name, "right", predicted_labels, confidence, true_labels, label_names, missclassified_dir, frame_n, resize_size=resize_size, position=position, font=font, color=color)
        if face is not None: 
            (missclassified_dir / task_name / str(cnt) / "face").mkdir(parents=True, exist_ok=True)
            for frame_n in range(47, 0, -3):
                write_misclassified_frame(face, i, cnt, task_name, "face", predicted_labels, confidence, true_labels, label_names, missclassified_dir, frame_n, resize_size=resize_size, position=position, font=font, color=color)
        if body is not None: 
            (missclassified_dir / task_name / str(cnt) / "body").mkdir(parents=True, exist_ok=True)
            for frame_n in range(47, 0, -3):
                write_misclassified_frame(body, i, cnt, task_name, "body", predicted_labels, confidence, true_labels, label_names, missclassified_dir, frame_n, resize_size=resize_size, position=position, font=font, color=color)

def test(model: nn.Module, test_dataset: CarDataset, checkpoints_dir: Path, name: str, batch_size: int=24, device: torch.device = torch.device("cpu"), num_workers: int=4, save_misclassified: bool=False):
    
    checkpoints_dir = checkpoints_dir / name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
     
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    accuracy1 = Accuracy(task="multiclass", num_classes=5).to(device)
    accuracy2 = Accuracy(task="multiclass", num_classes=7).to(device)
    accuracy3 = Accuracy(task="multiclass", num_classes=3).to(device)
    accuracy4 = Accuracy(task="multiclass", num_classes=5).to(device)
    confusion_matrix1 = ConfusionMatrix(task="multiclass", num_classes=5).to(device)
    confusion_matrix2 = ConfusionMatrix(task="multiclass", num_classes=7).to(device)
    confusion_matrix3 = ConfusionMatrix(task="multiclass", num_classes=3).to(device)
    confusion_matrix4 = ConfusionMatrix(task="multiclass", num_classes=5).to(device)
    
    model.to(device)
    model.eval()
    
    missclassified_dir = checkpoints_dir / f"{name}_misclassified"
    
    with torch.no_grad():
        for cnt, (inside, front, left, right, face, body, posture, gesture, emotion_label, behavior_label, context_label, vehicle_label) in tqdm(enumerate(test_loader), desc=f"Testing"):
            inside = inside.view(-1, 48, 224, 224).to(device, non_blocking=True)
            front = front.view(-1, 48, 224, 224).to(device, non_blocking=True)
            left = left.view(-1, 48, 224, 224).to(device, non_blocking=True)
            right = right.view(-1, 48, 224, 224).to(device, non_blocking=True)
            face = face.view(-1, 48, 224, 224).to(device, non_blocking=True)
            body = body.view(-1, 48, 224, 224).to(device, non_blocking=True)
            #gesture = gesture.view(batch_size, 3, 16, 42, 1).to(device, non_blocking=True)
            #posture = posture.view(batch_size, 3, 16, 26, 1).to(device, non_blocking=True)
            gesture = gesture.permute(0, 3, 1, 2).unsqueeze(-1).contiguous().to(device, non_blocking=True)
            posture = posture.permute(0, 3, 1, 2).unsqueeze(-1).contiguous().to(device, non_blocking=True)
            emotion_label = emotion_label.to(device, non_blocking=True)
            behavior_label = behavior_label.to(device, non_blocking=True)
            context_label = context_label.to(device, non_blocking=True)
            vehicle_label = vehicle_label.to(device, non_blocking=True)

            out1_test, out2_test, out3_test, out4_test = model(inside, front, left, right, face, body, gesture, posture)
            
            if save_misclassified:
                save_misclassified_images((out1_test.argmax(dim=1) != emotion_label).nonzero(as_tuple=True)[0], missclassified_dir, cnt, "DER", out1_test.argmax(dim=1), out1_test.softmax(dim=1).max(dim=1)[0], emotion_label, EMOTION_LABEL,inside=inside, front=front, left=left, right=right, face=face, body=body)
                save_misclassified_images((out2_test.argmax(dim=1) != behavior_label).nonzero(as_tuple=True)[0], missclassified_dir, cnt, "DBR", out2_test.argmax(dim=1), out2_test.softmax(dim=1).max(dim=1)[0], behavior_label, DRIVER_BEHAVIOR_LABEL, inside=inside, front=front, left=left, right=right, face=face, body=body)
                save_misclassified_images((out3_test.argmax(dim=1) != context_label).nonzero(as_tuple=True)[0], missclassified_dir, cnt, "TCR", out3_test.argmax(dim=1), out3_test.softmax(dim=1).max(dim=1)[0], context_label, SCENE_CENTRIC_CONTEXT_LABEL, inside=inside, front=front, left=left, right=right, face=face, body=body)
                save_misclassified_images((out4_test.argmax(dim=1) != vehicle_label).nonzero(as_tuple=True)[0], missclassified_dir, cnt, "VBR", out4_test.argmax(dim=1), out4_test.softmax(dim=1).max(dim=1)[0], vehicle_label, VEHICLE_BASED_CONTEXT_LABEL, inside=inside, front=front, left=left, right=right, face=face, body=body)


            
            accuracy1.update(out1_test, emotion_label)
            accuracy2.update(out2_test, behavior_label)
            accuracy3.update(out3_test, context_label)
            accuracy4.update(out4_test, vehicle_label)
            confusion_matrix1.update(out1_test, emotion_label)
            confusion_matrix2.update(out2_test, behavior_label)
            confusion_matrix3.update(out3_test, context_label)
            confusion_matrix4.update(out4_test, vehicle_label)
            
    mAcc = (accuracy1.compute() + accuracy2.compute() + accuracy3.compute() + accuracy4.compute()) / 4
    print(f"Test - Accuracy 1: {accuracy1.compute():.4f}, Accuracy 2: {accuracy2.compute():.4f}, Accuracy 3: {accuracy3.compute():.4f}, Accuracy 4: {accuracy4.compute():.4f}, mAcc: {mAcc:.4f}")
    print(f"Test - Confusion Matrix 1:\n{confusion_matrix1.compute()}")
    print(f"Test - Confusion Matrix 2:\n{confusion_matrix2.compute()}")
    print(f"Test - Confusion Matrix 3:\n{confusion_matrix3.compute()}")
    print(f"Test - Confusion Matrix 4:\n{confusion_matrix4.compute()}")

    with open(checkpoints_dir / f"{name}_test_results.txt", "w") as f:
        f.write(f"Test - Accuracy 1: {accuracy1.compute():.4f}, Accuracy 2: {accuracy2.compute():.4f}, Accuracy 3: {accuracy3.compute():.4f}, Accuracy 4: {accuracy4.compute():.4f}, mAcc: {mAcc:.4f}\n")
        f.write(f"Test - Confusion Matrix 1:\n{confusion_matrix1.compute()}\n")
        f.write(f"Test - Confusion Matrix 2:\n{confusion_matrix2.compute()}\n")
        f.write(f"Test - Confusion Matrix 3:\n{confusion_matrix3.compute()}\n")
        f.write(f"Test - Confusion Matrix 4:\n{confusion_matrix4.compute()}\n")
    
    plot_confusion_matrix(confusion_matrix1.compute(), EMOTION_LABEL, "Confusion Matrix for DER", checkpoints_dir / f"{name}_confusion_matrix_DER.png")
    plot_confusion_matrix(confusion_matrix2.compute(), DRIVER_BEHAVIOR_LABEL, "Confusion Matrix for DBR", checkpoints_dir / f"{name}_confusion_matrix_DBR.png")
    plot_confusion_matrix(confusion_matrix3.compute(), SCENE_CENTRIC_CONTEXT_LABEL, "Confusion Matrix for TCR", checkpoints_dir / f"{name}_confusion_matrix_TCR.png")
    plot_confusion_matrix(confusion_matrix4.compute(), VEHICLE_BASED_CONTEXT_LABEL, "Confusion Matrix for VBR", checkpoints_dir / f"{name}_confusion_matrix_VBR.png")

def instantiate_MARNetOnly():
    from Models import MARNetOnly
    return MARNetOnly()

def instantiate_DFMEFusion():
    from Models import DBMEFusion
    return DBMEFusion()

def instantiate_BaselineNet():
    from Models import BaselineNet
    return BaselineNet()

def instantiate_CausalNet():
    from Models import CausalNet
    return CausalNet(context_dim=512)

def instantiate_Resnet18EncoderNet(pretrained=True, freeze=True):
    from Models import Resnet18EncoderNet
    return Resnet18EncoderNet(pretrained=pretrained, freeze=freeze)

def instantiate_MARNetPretrainingWrapper():
    from Models import MARNetPretrainingWrapper
    return MARNetPretrainingWrapper()

if __name__ == "__main__":
    
    available_models = {
        "MARNetOnly": instantiate_MARNetOnly,
        "MARNetPretrainingWrapper": instantiate_MARNetPretrainingWrapper,
        "fusion": instantiate_DFMEFusion,
        "baseline": instantiate_BaselineNet,
        "causalnet": instantiate_CausalNet,
        "resnet18": lambda: instantiate_Resnet18EncoderNet(pretrained=True, freeze=True),
        "resnet18scratch": lambda: instantiate_Resnet18EncoderNet(pretrained=False, freeze=False)
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=available_models.keys(), required=True, help="Type of model to train: 'baseline' or 'fusion'")
    parser.add_argument("--checkpoints_dir", type=Path, default="./checkpoints", help="Path to the checkpoints directory")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to the dataset directory")
    parser.add_argument("--split_dir", type=Path, required=True, help="Path to the split directory")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to be used")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--start_from_checkpoint", type=Path, default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lr_drop_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced")
    parser.add_argument("--lr_drop_patience", type=int, default=5, help="Number of epochs with no improvement after which learning rate will be reduced")
    parser.add_argument("--save_misclassified", action="store_true", help="Flag to save misclassified images during testing")
    parser.add_argument("--use_experimental_augmentation", action="store_true", help="Flag to use experimental augmentation techniques during training")
    parser.add_argument("--synchronized_augmentation", action="store_true", help="Flag to synchronize augmentation across all camera views during training")

    args = parser.parse_args()

    if not args.split_dir.exists():
        print(f"Split directory {args.split_dir} does not exist. Please provide a valid path.")
        exit(1)

    if not args.dataset_dir.exists():
        print(f"Dataset directory {args.dataset_dir} does not exist. Please provide a valid path.")
        exit(1)
        
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    
    train_dataset = CarDataset(csv_file=args.split_dir / "training.csv", dataset_root=args.dataset_dir, horizontal_flip_prob=0.5, vertical_flip_prob=0.5, use_experimental_augmentation=args.use_experimental_augmentation, synchronize_transforms=args.synchronized_augmentation)
    val_dataset = CarDataset(csv_file=args.split_dir / "validation.csv", dataset_root=args.dataset_dir, horizontal_flip_prob=0.0, vertical_flip_prob=0.0)
    
    if args.model_type in available_models:
        model = available_models[args.model_type]().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        raise ValueError(f"Invalid model type. Please choose one of: {', '.join(available_models.keys())}.")

    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    
    warmup_epochs = 10

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs - warmup_epochs,
        eta_min=1e-5,
    )

    
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
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
        
    train(model, optimizer, scheduler, train_dataset, val_dataset, args.checkpoints_dir, name=args.model_name, start_epoch=start_epoch, num_epochs=args.num_epochs, batch_size=args.batch_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_workers=args.num_workers)
    
    test_dataset = CarDataset(csv_file=args.split_dir / "testing.csv", dataset_root=args.dataset_dir, horizontal_flip_prob=0.0, vertical_flip_prob=0.0)
    
    test(model, test_dataset, args.checkpoints_dir, name=args.model_name, batch_size=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_workers=args.num_workers, save_misclassified=args.save_misclassified)
    
    best_checkpoint_path = args.checkpoints_dir / args.model_name / f"{args.model_name}_best.pt"
    
    best_checkpoint = Checkpoint()
    best_checkpoint.load_from_file(best_checkpoint_path)
    
    if best_checkpoint.state_dict is None:
        raise ValueError(f"The best checkpoint at {best_checkpoint_path} does not contain a valid state_dict.")
    
    model.load_state_dict(best_checkpoint.state_dict)
    
    test(model, test_dataset, args.checkpoints_dir, name=args.model_name, batch_size=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_workers=args.num_workers)