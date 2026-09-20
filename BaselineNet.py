from torch import nn
import torch

from MARNet import MARNet

class ConvNet3D(nn.Module):
    def __init__(self, num_classes=512, num_keypoints=42):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64 * 16 * (num_keypoints) * 1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
        
        self.fc1 = nn.Linear(4096, 5)
        self.fc2 = nn.Linear(4096, 7)
        self.fc3 = nn.Linear(4096, 3)
        self.fc4 = nn.Linear(4096, 5)

    def forward(self, img1, img2, img3, img4, face, body, gesture, posture):

        # Original feature extraction
        h1 = self.marnet1(img1)
        h2 = self.marnet2(img2)
        h3 = self.marnet3(img3)
        h4 = self.marnet4(img4)
        h_face = self.marnet5(face)
        h_body = self.marnet6(body)

        h_gesture = self.conv3d_gesture(gesture)
        h_posture = self.conv3d_posture(posture)
        
        h1 = self.avg_pool(h1).squeeze(-1).squeeze(-1)
        h2 = self.avg_pool(h2).squeeze(-1).squeeze(-1)
        h3 = self.avg_pool(h3).squeeze(-1).squeeze(-1)
        h4 = self.avg_pool(h4).squeeze(-1).squeeze(-1)
        h_face = self.avg_pool(h_face).squeeze(-1).squeeze(-1)
        h_body = self.avg_pool(h_body).squeeze(-1).squeeze(-1)
        
        fc = torch.cat([h1, h2, h3, h4, h_face, h_body, h_gesture, h_posture], dim=1)
        
        out1 = self.fc1(fc)
        out2 = self.fc2(fc)
        out3 = self.fc3(fc)
        out4 = self.fc4(fc)

        return out1, out2, out3, out4
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
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