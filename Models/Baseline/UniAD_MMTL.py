from torch import nn
import torch
from torchvision.models import resnet18, ResNet18_Weights

from Models.Baseline.Fusion import TaskSharedBranch, TaskSpecificBranch
from Models.Baseline.MARNet import MARNet
from Models.Baseline.MARNetOnly import ConvNet3D


class Resnet18EncoderNet(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        self.pretrained = pretrained
        self.freeze = freeze
        
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None)
        
        self.encoder = nn.Sequential(*list(backbone.children())[:-3])
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        self.conv1x1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))

        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.conv3d_gesture = ConvNet3D(num_keypoints=42)
        self.conv3d_posture = ConvNet3D(num_keypoints=26)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        #TODO: Increase the size of the projection layers.
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
        
        self.fc1 = nn.Linear(4096, 5)
        self.fc2 = nn.Linear(4096, 7)
        self.fc3 = nn.Linear(4096, 3)
        self.fc4 = nn.Linear(4096, 5)
        
        self.dropout = nn.Dropout(p=0.5)
    
    def encode_stream(self, x):
        B, C, H, W = x.shape
        T = C // 3

        x = x.reshape(B * T, 3, H, W)

        # x must already be in [0, 1]
        x = self.normalize(x)

        with torch.no_grad():
            h = self.encoder(x)

        h = h.reshape(
            B, T,
            h.shape[1],
            h.shape[2],
            h.shape[3]
        )

        h = h.mean(dim=1)
        h = self.conv1x1(h)

        return h
    
    def forward(self, img1, img2, img3, img4, face, body, gesture, posture):
        h1 = self.encode_stream(img1)
        h2 = self.encode_stream(img2)
        h3 = self.encode_stream(img3)
        h4 = self.encode_stream(img4)
        h_face = self.encode_stream(face)
        h_body = self.encode_stream(body)

        h_gesture = self.conv3d_gesture(gesture)
        h_posture = self.conv3d_posture(posture)
        
        h1_g = self.avg_pool(h1).flatten(1)
        h2_g = self.avg_pool(h2).flatten(1)
        h3_g = self.avg_pool(h3).flatten(1)
        h4_g = self.avg_pool(h4).flatten(1)
        h_face_g = self.avg_pool(h_face).flatten(1)
        h_body_g = self.avg_pool(h_body).flatten(1)
        modality_tokens = torch.stack([h1_g, h2_g, h3_g, h4_g, h_face_g, h_body_g, h_gesture, h_posture], dim=1)
        
        f_sp1 = self.dropout(self.specific1(modality_tokens))
        f_sp2 = self.dropout(self.specific2(modality_tokens))
        f_sp3 = self.dropout(self.specific3(modality_tokens))
        f_sp4 = self.dropout(self.specific4(modality_tokens))

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

        f_sh = self.dropout(self.avg_pool(f_sh).flatten(1))
    
        out1 = self.fc1(f_sp1) * torch.sigmoid(self.weight1) + self.fcs1(f_sh) * (1 - torch.sigmoid(self.weight1))
        out2 = self.fc2(f_sp2) * torch.sigmoid(self.weight2) + self.fcs2(f_sh) * (1 - torch.sigmoid(self.weight2))
        out3 = self.fc3(f_sp3) * torch.sigmoid(self.weight3) + self.fcs3(f_sh) * (1 - torch.sigmoid(self.weight3))
        out4 = self.fc4(f_sp4) * torch.sigmoid(self.weight4) + self.fcs4(f_sh) * (1 - torch.sigmoid(self.weight4))
        
        return out1, out2, out3, out4
    
    def train(self, mode=True):
        super().train(mode)
        # This accounts for the batch normalization layers in the encoder, which should be frozen during training.
        if self.freeze:
            self.encoder.eval()
        return self
    
    def normalize(self, x):
        return (x - self.mean) / self.std
        
        

class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.marnet1 = MARNet()
        self.marnet2 = MARNet()
        self.marnet3 = MARNet()
        self.marnet4 = MARNet()
        self.marnet5 = MARNet()
        self.marnet6 = MARNet()

        self.conv3d_gesture = ConvNet3D(num_keypoints=42)
        self.conv3d_posture = ConvNet3D(num_keypoints=26)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        #TODO: Increase the size of the projection layers.
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
        
        self.fc1 = nn.Linear(4096, 5)
        self.fc2 = nn.Linear(4096, 7)
        self.fc3 = nn.Linear(4096, 3)
        self.fc4 = nn.Linear(4096, 5)
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img1, img2, img3, img4, face, body, gesture, posture):

        h1 = self.marnet1(img1)
        h2 = self.marnet2(img2)
        h3 = self.marnet3(img3)
        h4 = self.marnet4(img4)
        h_face = self.marnet5(face)
        h_body = self.marnet6(body)

        h_gesture = self.conv3d_gesture(gesture)
        h_posture = self.conv3d_posture(posture)
        
        h1_g = self.avg_pool(h1).flatten(1)
        h2_g = self.avg_pool(h2).flatten(1)
        h3_g = self.avg_pool(h3).flatten(1)
        h4_g = self.avg_pool(h4).flatten(1)
        h_face_g = self.avg_pool(h_face).flatten(1)
        h_body_g = self.avg_pool(h_body).flatten(1)
        modality_tokens = torch.stack([h1_g, h2_g, h3_g, h4_g, h_face_g, h_body_g, h_gesture, h_posture], dim=1)
        
        f_sp1 = self.dropout(self.specific1(modality_tokens))
        f_sp2 = self.dropout(self.specific2(modality_tokens))
        f_sp3 = self.dropout(self.specific3(modality_tokens))
        f_sp4 = self.dropout(self.specific4(modality_tokens))

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

        f_sh = self.dropout(self.avg_pool(f_sh).flatten(1))
    
        out1 = self.fc1(f_sp1) * torch.sigmoid(self.weight1) + self.fcs1(f_sh) * (1 - torch.sigmoid(self.weight1))
        out2 = self.fc2(f_sp2) * torch.sigmoid(self.weight2) + self.fcs2(f_sh) * (1 - torch.sigmoid(self.weight2))
        out3 = self.fc3(f_sp3) * torch.sigmoid(self.weight3) + self.fcs3(f_sh) * (1 - torch.sigmoid(self.weight3))
        out4 = self.fc4(f_sp4) * torch.sigmoid(self.weight4) + self.fcs4(f_sh) * (1 - torch.sigmoid(self.weight4))
        
        return out1, out2, out3, out4
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        gestureCnn = ConvNet3D(num_keypoints=42).to(device).eval()
        postureCnn = ConvNet3D(num_keypoints=26).to(device).eval()
        
        gesture = torch.randn(24, 16, 42, 3).to(device)
        posture = torch.randn(24, 16, 26, 3).to(device)
        
        gesture = gesture.permute(0, 3, 1, 2).unsqueeze(-1).contiguous().to(device)
        posture = posture.permute(0, 3, 1, 2).unsqueeze(-1).contiguous().to(device)
        
        gesture_output = gestureCnn(gesture)
        posture_output = postureCnn(posture)
        
        print("Gesture output shape:", gesture_output.shape)
        print("Posture output shape:", posture_output.shape)


        model = BaselineNet().to(device).eval()
        img1 = torch.randn(24, 48, 224, 224).to(device)
        img2 = torch.randn(24, 48, 224, 224).to(device)
        img3 = torch.randn(24, 48, 224, 224).to(device)
        img4 = torch.randn(24, 48, 224, 224).to(device)
        face = torch.randn(24, 48, 224, 224).to(device)
        body = torch.randn(24, 48, 224, 224).to(device)
        gesture = torch.randn(24, 3, 16, 42, 1).to(device)
        posture = torch.randn(24, 3, 16, 26, 1).to(device)
        
        out1, out2, out3, out4 = model(img1, img2, img3, img4, face, body, gesture, posture)
        print("Output shapes:", out1.shape, out2.shape, out3.shape, out4.shape)
        
        # print parameter counts for BaselineNet
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in BaselineNet: {total_params}")
        