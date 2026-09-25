from typing import Any

import torch
import torch.nn as nn
import math

from BaselineNet import ConvNet3D

class VGG(torch.nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        
        self.convBlock1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.convBlock2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.convBlock3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.convBlock4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: Any) -> Any:
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        return x

class TaskSpecificBranch(nn.Module):
    """
    Input:  [B, 8, 512]
    Output: [B, 4096]
    """
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.conv_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv_k = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv_v = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        x = x.transpose(1, 2)

        q = self.conv_q(x).transpose(1, 2)
        k = self.conv_k(x).transpose(1, 2)
        v = self.conv_v(x).transpose(1, 2)

        B, N, C = q.shape

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).contiguous().view(B, N, C)

        weights = torch.sigmoid(x)

        x = residual * weights

        return x.flatten(1)

    
class ChannelGate(nn.Module):
    """
    Input:  [B, C, H, W]
    Output: [B, C, 1, 1]
    """
    def __init__(self):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv_q = nn.Conv1d(1, 1, 3, padding=1)
        self.conv_k = nn.Conv1d(1, 1, 3, padding=1)
        self.conv_v = nn.Conv1d(1, 1, 3, padding=1)

    def forward(self, x):
        x = self.gap(x).flatten(1)
        x = x.unsqueeze(1)

        q = self.conv_q(x).transpose(1, 2)
        k = self.conv_k(x).transpose(1, 2)
        v = self.conv_v(x).transpose(1, 2)

        attention = q @ k.transpose(-1, -2)
        attention = torch.softmax(attention, dim=-1)

        x = attention @ v

        weights = torch.sigmoid(x).unsqueeze(-1)

        return weights

class TaskSharedBranch(nn.Module):
    """
    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(self, channels=512):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gate = ChannelGate()

    def forward(self, t1, t2):
        x = t1 + t2
        x = self.conv1(x) + self.conv3(x)
        w = self.gate(x)

        return t2 * w + t1 * (1 - w)


class DBMEFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.VGG1 = VGG(in_channels=48)
        self.VGG2 = VGG(in_channels=48)
        self.VGG3 = VGG(in_channels=48)
        self.VGG4 = VGG(in_channels=48)
        self.VGG5 = VGG(in_channels=48)
        self.VGG6 = VGG(in_channels=48)

        self.conv3d_gesture = ConvNet3D(num_keypoints=42)
        self.conv3d_posture = ConvNet3D(num_keypoints=26)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.driver_projection = nn.Conv2d(1536, 512, kernel_size=1)

        self.scene_projection = nn.Conv2d(1536, 512, kernel_size=1)
        
        self.joint_projection = nn.Conv2d(1024, 512, kernel_size=1)

        self.specific1 = TaskSpecificBranch()
        self.specific2 = TaskSpecificBranch()
        self.specific3 = TaskSpecificBranch()
        self.specific4 = TaskSpecificBranch()

        self.shared1 = TaskSharedBranch(512)
        self.shared2 = TaskSharedBranch(512)
        
        self.fc1 = nn.Linear(4096, 5)
        self.fc2 = nn.Linear(4096, 7)
        self.fc3 = nn.Linear(4096, 3)
        self.fc4 = nn.Linear(4096, 5)
        
        self.fcs1 = nn.Linear(512, 5)
        self.fcs2 = nn.Linear(512, 7)
        self.fcs3 = nn.Linear(512, 3)
        self.fcs4 = nn.Linear(512, 5)
        
        self.weight1 = nn.Parameter(torch.tensor(0.0))
        self.weight2 = nn.Parameter(torch.tensor(0.0))
        self.weight3 = nn.Parameter(torch.tensor(0.0))
        self.weight4 = nn.Parameter(torch.tensor(0.0))


    def forward(self, img1, img2, img3, img4, face, body, gesture, posture):
        h1 = self.VGG1(img1)
        h2 = self.VGG2(img2)
        h3 = self.VGG3(img3)
        h4 = self.VGG4(img4)
        h_face = self.VGG5(face)
        h_body = self.VGG6(body)

        h_gesture = self.conv3d_gesture(gesture)
        h_posture = self.conv3d_posture(posture)

        h1_g = self.gap(h1).flatten(1)
        h2_g = self.gap(h2).flatten(1)
        h3_g = self.gap(h3).flatten(1)
        h4_g = self.gap(h4).flatten(1)
        h_face_g = self.gap(h_face).flatten(1)
        h_body_g = self.gap(h_body).flatten(1)

        modality_tokens = torch.stack([h1_g, h2_g, h3_g, h4_g, h_face_g, h_body_g, h_gesture, h_posture], dim=1)

        f_sp1 = self.specific1(modality_tokens)
        f_sp2 = self.specific2(modality_tokens)
        f_sp3 = self.specific3(modality_tokens)
        f_sp4 = self.specific4(modality_tokens)

        f_dr = torch.cat([h1, h_face, h_body], dim=1)

        f_sc = torch.cat([h2, h3, h4], dim=1)

        f_dr = self.driver_projection(f_dr)
        f_sc = self.scene_projection(f_sc)

        f_ps = self.shared1(f_dr, f_sc)

        H, W = f_ps.shape[-2:]

        gesture_map = h_gesture[:, :, None, None].expand(-1, -1, H, W)

        posture_map = h_posture[:, :, None, None].expand(-1, -1, H, W)

        f_jo = torch.cat([gesture_map, posture_map],dim=1)

        f_jo = self.joint_projection(f_jo)
        f_sh = self.shared2(f_jo, f_ps)

        f_sh = self.gap(f_sh).flatten(1)
    
        out1 = self.fc1(f_sp1) * torch.sigmoid(self.weight1) + self.fcs1(f_sh) * (1 - torch.sigmoid(self.weight1))
        out2 = self.fc2(f_sp2) * torch.sigmoid(self.weight2) + self.fcs2(f_sh) * (1 - torch.sigmoid(self.weight2))
        out3 = self.fc3(f_sp3) * torch.sigmoid(self.weight3) + self.fcs3(f_sh) * (1 - torch.sigmoid(self.weight3))
        out4 = self.fc4(f_sp4) * torch.sigmoid(self.weight4) + self.fcs4(f_sh) * (1 - torch.sigmoid(self.weight4))
        
        return out1, out2, out3, out4


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #model = VGG(in_channels=48).to(device)
    #
    #x = torch.randn(24, 48, 224, 224).to(device, dtype=torch.float32)
    #
    #y = model(x)
    #
    #print(f"Input shape: {x.shape}")
    #print(f"Output shape: {y.shape}")
    #
    #model = TaskSpecificBranch().to(device)
    #
    #x = torch.randn(24, 512, 14, 14).to(device, dtype=torch.float32)
    #
    #y = model(x)
    #
    #print(f"Input shape: {x.shape}")
    #print(f"Output shape: {y.shape}")
    #
    #model = TaskSharedBranch().to(device)
    #
    #x1 = torch.randn(24, 512, 14, 14).to(device, dtype=torch.float32)
    #x2 = torch.randn(24, 512, 14, 14).to(device, dtype=torch.float32)
    #
    #y = model(x1, x2)
    #
    #print(f"Input shape: {x1.shape}")
    #print(f"Output shape: {y.shape}")
    
    model = DBMEFusion().to(device)
    
    img1 = torch.randn(6, 48, 224, 224).to(device)
    img2 = torch.randn(6, 48, 224, 224).to(device)
    img3 = torch.randn(6, 48, 224, 224).to(device)
    img4 = torch.randn(6, 48, 224, 224).to(device)
    face = torch.randn(6, 48, 224, 224).to(device)
    body = torch.randn(6, 48, 224, 224).to(device)
    gesture = torch.randn(6, 3, 16, 42, 1).to(device)
    posture = torch.randn(6, 3, 16, 26, 1).to(device)
    
    out1, out2, out3, out4 = model(img1, img2, img3, img4, face, body, gesture, posture)
    
    print("Output shapes:", out1.shape, out2.shape, out3.shape, out4.shape)