import argparse
from pathlib import Path
import torch
import torch.nn as nn

from Data import CarDataset
from Utils import Checkpoint

from main import test

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description="Test the model on the test dataset.")
    parser.add_argument("--model_type", type=str, required=True, choices=["MARNetOnly", "fusion", "baseline", "causalnet", "resnet18", "resnet18scratch"], help="Type of model to test: 'MARNetOnly', 'fusion', 'baseline', or 'causalnet'.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Directory where the dataset is located.")
    parser.add_argument("--split_dir", type=Path, required=True, help="Directory where the dataset split files are located.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--save_misclassified", action="store_true", help="Flag to save misclassified images during testing.")
    args = parser.parse_args()
    
    if args.model_type == "MARNetOnly":
        from Models import MARNetOnly
        model = MARNetOnly()
    elif args.model_type == "fusion":
        from Models import DBMEFusion
        model = DBMEFusion()
    elif args.model_type == "baseline":
        from Models import BaselineNet
        model = BaselineNet()
    elif args.model_type == "causalnet":
        from Models import CausalNet
        model = CausalNet(context_dim=512, alpha_blend=False)
    elif args.model_type == "resnet18":
        from Models import Resnet18EncoderNet
        model = Resnet18EncoderNet(pretrained=True, freeze=True)
    elif args.model_type == "resnet18scratch":
        from Models import Resnet18EncoderNet
        model = Resnet18EncoderNet(pretrained=False, freeze=False)
    else:
        raise ValueError("Invalid model type. Please choose 'baseline' or 'fusion'.")
    
    test_dataset = CarDataset(csv_file=args.split_dir / "testing.csv", dataset_root=args.dataset_dir, horizontal_flip_prob=0.0, vertical_flip_prob=0.0)
        
    best_checkpoint_path = args.checkpoint
    best_checkpoint_name = Path(best_checkpoint_path).stem
    best_checkpoint_dir = Path(best_checkpoint_path).parent
    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"The specified checkpoint file does not exist: {best_checkpoint_path}")
    
    best_checkpoint = Checkpoint()
    best_checkpoint.load_from_file(best_checkpoint_path)
    
    if best_checkpoint.state_dict is None:
        raise ValueError(f"The best checkpoint at {best_checkpoint_path} does not contain a valid state_dict.")
    
    model.load_state_dict(best_checkpoint.state_dict)
    
    test(model, test_dataset, best_checkpoint_dir, name=best_checkpoint_name, batch_size=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_workers=args.num_workers, save_misclassified=args.save_misclassified)
    