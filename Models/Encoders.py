import torch
from torch import nn


from abc import ABC, abstractmethod
from typing import Optional, override, Type


class Encoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        ...
    
    @abstractmethod
    def get_output_dim(self) -> int:
        ...
    
    @abstractmethod
    def get_input_dim(self) -> int:
        ...
    
    def __init__(self, *args, **kwargs):
        super().__init__()


class Resnet18Encoder(Encoder):
    def __init__(self, pretrained: bool = False, freeze: bool = False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        
        self.freeze = freeze
        
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-3])
        
        self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def encode_stream(self, x):
        B, C, H, W = x.shape
        T = C // 3

        x = x.reshape(B * T, 3, H, W)

        # x must already be in [0, 1]
        x = self.normalize(x)

        with torch.no_grad():
            h = self.encoder(x)

        h = h.reshape(B, T, h.shape[1], h.shape[2], h.shape[3])

        h = h.mean(dim=1)
        h = self.conv1x1(h)

        return h

    def train(self, mode=True):
        super().train(mode)
        # This accounts for the batch normalization layers in the encoder, which should be frozen during training.
        if self.freeze:
            self.encoder.eval()
        return self

    @override
    def forward(self, x):
        return self.encode_stream(x)
    
    @override
    def get_output_dim(self):
        return self.encoder[-1].out_channels
    
    @override
    def get_input_dim(self):
        return self.encoder[0].in_channels

class MARNetEncoder(Encoder):
    def __init__(self, pretrained: bool = False, freeze: bool = False):
        super().__init__()
        from Models.Baseline.MARNet import MARNet
        
        self.encoder = MARNet()
    
    @override
    def forward(self, x):
        return self.encoder(x)
    
    @override
    def get_output_dim(self):
        return list(self.encoder.children())[-1].out_channels
    
    @override
    def get_input_dim(self):
        return list(self.encoder.children())[0].in_channels

class MobileNetV3Encoder(Encoder):
    def __init__(self, pretrained: bool = False, freeze: bool = False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        
        self.freeze = freeze
        
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.conv1x1 = nn.Conv2d(576, 512, kernel_size=1, stride=1, padding=0)
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    @override
    def forward(self, x):
        B, C, H, W = x.shape
        T = C // 3

        x = x.reshape(B * T, 3, H, W)

        # x must already be in [0, 1]
        x = self.normalize(x)

        with torch.no_grad():
            h = self.encoder(x)

        h = h.reshape(B, T, h.shape[1], h.shape[2], h.shape[3])

        h = h.mean(dim=1)
        h = self.conv1x1(h)
        print("MobileNetV3Encoder output shape:", h.shape)

        return h
    
    @override
    def get_output_dim(self):
        return list(self.encoder.children())[-1].out_channels
    
    @override
    def get_input_dim(self):
        return list(self.encoder.children())[0].in_channels

    def train(self, mode=True):
        super().train(mode)
        # This accounts for the batch normalization layers in the encoder, which should be frozen during training.
        if self.freeze:
            self.encoder.eval()
        return self
