import torch
import math
from torch import nn
# from Fusionlist.SeBlock import *
# from EcaBlock import *
from torch.nn import init

class DBME(nn.Module):
    def __init__(self, channels=512,r=4):
        super(DBME, self).__init__()
        inter_channels = int(channels // r) 
        # kernel_size = int(abs((math.log(channels, 2) + 1) / 2))
        # print(kernel_size)
        # print((kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # self.conv = nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias = False)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(channels)

        # self.local_att2 = nn.Sequential(
        #     nn.Conv1d(1, 1, kernel_size = 3, padding =1, bias = False),
        #     nn.BatchNorm2d(channels),
        # )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )


        # self.attention1 = ECABlock(channels, gamma = 2, b = 1)
 
        # self.attention2 = SEBlock(channels, 16)
        self.attention2 = ECAAttention(channels,8)
 
        self.sigmoid = nn.Sigmoid()
 
 
    def forward(self, x, residual):
        xa = x + residual

        xz1 = self.local_att1(xa)
        # xg1 = self.attention1(xz1)
        

        # xz2 = self.conv(xa.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        xz2 = self.conv(xa)
        
        xz2 = self.bn(xz2)
        # xz2 = self.bn(xz2)
        # xg2 = self.attention2(xz2)

        xlg1 = xz2 + xz1
        xlg1 = self.attention2(xlg1)
        wei = self.sigmoid(xlg1)
 
 
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
    

class ECAAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv3=nn.Conv1d(1,1,kernel_size=3,padding=(3-1)//2)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x) # bs, c, 1, 1
        y = y.squeeze(-1).permute(0, 2, 1) # bs, 1, c
        y3=self.conv3(y) #bs,1,c

        attn_output, _ = self.attention(y3, y3, y3) # attn_output: bs, 1,  c
        attn_output = attn_output.permute(0, 2, 1).unsqueeze(-1) # bs, c, 1, 1
        attn_output = self.sigmoid(attn_output) # bs, c, 1, 1
        return x * attn_output.expand_as(x)
    
if __name__ == "__main__":

    model = DBME(channels)#.cuda()
    print("Model loaded.")
    x1 = torch.rand(2, 512,1,1)#.cuda()
    x2 = torch.rand(2, 512,1,1)#.cuda()
    print("x1 and x2 loaded.")

	# Run a feedforward and check shape
    c = model(x1, x2)
    print(image.shape)
    print(c.shape)
