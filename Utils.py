import torch
from pathlib import Path
from typing import Any, Mapping, Optional

        
class Checkpoint:
    def __init__(self):
        self.state_dict: Optional[Mapping[str, Any]] = None
        self.optimizer_state_dict: Optional[Mapping[str, Any]] = None
        self.scheduler_state_dict: Optional[Mapping[str, Any]] = None
        self.epoch: Optional[int] = None
        self.name: Optional[str] = None
        self.val_accuracy1: Optional[float] = None
        self.val_accuracy2: Optional[float] = None
        self.val_accuracy3: Optional[float] = None
        self.val_accuracy4: Optional[float] = None
        self.val_mAcc: Optional[float] = None
        self.val_mLoss: Optional[float] = None
    
    def load_from_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            raise FileNotFoundError(f"The provided path {file_path} is not a valid file.")
        data = torch.load(file_path, map_location=torch.device('cpu'))
        self.state_dict = data.get('state_dict')
        self.optimizer_state_dict = data.get('optimizer_state_dict')
        self.scheduler_state_dict = data.get('scheduler_state_dict')
        self.epoch = data.get('epoch')
        self.name = data.get('name')
        self.val_accuracy1 = data.get('val_accuracy1')
        self.val_accuracy2 = data.get('val_accuracy2')
        self.val_accuracy3 = data.get('val_accuracy3')
        self.val_accuracy4 = data.get('val_accuracy4')
        self.val_mAcc = data.get('val_mAcc')
        self.val_mLoss = data.get('val_mLoss')

    def asdict(self) -> Mapping[str, Any]:
        return {
            'state_dict': self.state_dict,
            'optimizer_state_dict': self.optimizer_state_dict,
            'scheduler_state_dict': self.scheduler_state_dict,
            'epoch': self.epoch,
            'name': self.name,
            'val_accuracy1': self.val_accuracy1,
            'val_accuracy2': self.val_accuracy2,
            'val_accuracy3': self.val_accuracy3,
            'val_accuracy4': self.val_accuracy4,
            'val_mAcc': self.val_mAcc,
            'val_mLoss': self.val_mLoss
        }
        
class EarlyStopper:
    """
    Source: https://stackoverflow.com/a/73704579/14171143
    """
    
    def __init__(self, patience: int=1, min_delta: float=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model_checkpoint: Checkpoint = Checkpoint()

    def early_stop(self, checkpoint: Checkpoint, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_checkpoint = checkpoint
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def plot_confusion_matrix(confusion_matrix, labels, title, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(confusion_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title(title)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    checkpoint = Checkpoint()
    checkpoint.load_from_file(Path("./checkpoint/MARNetOnly_b6/MARNetOnly_b6_0.pt"))
    print(tuple(checkpoint.asdict().keys()))