import torch
import sys
from pathlib import Path

from Utils import Checkpoint

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise ValueError("Please provide the path to the checkpoint file as a command line argument.")
    
    checkpoint_path = Path(sys.argv[1])
    
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"The provided path {checkpoint_path} is not a valid file.")
    
    checkpoint = Checkpoint()
    checkpoint.load_from_file(checkpoint_path)
    
    if checkpoint.name is not None:
            print(f"Checkpoint name: {checkpoint.name}")
    if checkpoint.epoch is not None:
        print(f"Last train/val epoch: {checkpoint.epoch}")
    if checkpoint.val_accuracy1 is not None:
        print(f"Validation Accuracy 1: {checkpoint.val_accuracy1:.4f}")
    if checkpoint.val_accuracy2 is not None:
        print(f"Validation Accuracy 2: {checkpoint.val_accuracy2:.4f}")
    if checkpoint.val_accuracy3 is not None:
        print(f"Validation Accuracy 3: {checkpoint.val_accuracy3:.4f}")
    if checkpoint.val_accuracy4 is not None:
        print(f"Validation Accuracy 4: {checkpoint.val_accuracy4:.4f}")
    if checkpoint.val_accuracy1 is not None and checkpoint.val_accuracy2 is not None and checkpoint.val_accuracy3 is not None and checkpoint.val_accuracy4 is not None:
        mAcc = (checkpoint.val_accuracy1 + checkpoint.val_accuracy2 + checkpoint.val_accuracy3 + checkpoint.val_accuracy4) / 4
        print(f"Validation mAcc: {mAcc:.4f}")
    if checkpoint.val_mLoss is not None:
        print(f"Validation mLoss: {checkpoint.val_mLoss:.4f}")