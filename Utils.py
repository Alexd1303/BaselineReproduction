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


if __name__ == "__main__":
    checkpoint = Checkpoint()
    checkpoint.load_from_file(Path("./checkpoint/MARNetOnly_b6/MARNetOnly_b6_0.pt"))
    print(tuple(checkpoint.asdict().keys()))