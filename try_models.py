import torch
import torch.nn as nn

from Models import Resnet18EncoderNet, BaselineNet, MARNetPretrainingWrapper

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        img1 = torch.randn(24, 48, 224, 224).to(device)
        img2 = torch.randn(24, 48, 224, 224).to(device)
        img3 = torch.randn(24, 48, 224, 224).to(device)
        img4 = torch.randn(24, 48, 224, 224).to(device)
        face = torch.randn(24, 48, 224, 224).to(device)
        body = torch.randn(24, 48, 224, 224).to(device)
        gesture = torch.randn(24, 3, 16, 42, 1).to(device)
        posture = torch.randn(24, 3, 16, 26, 1).to(device)
        
        #model = Resnet18EncoderNet().to(device).eval()
        #
        #out1, out2, out3, out4 = model(img1, img2, img3, img4, face, body, gesture, posture)
        #
        #print("Output shapes:", out1.shape, out2.shape, out3.shape, out4.shape)
        #print("Parameter count:", sum(p.numel() for p in model.parameters()))
        #
        #model = BaselineNet().to(device).eval()
        #
        #out1, out2, out3, out4 = model(img1, img2, img3, img4, face, body, gesture, posture)
        #
        #print("Output shapes:", out1.shape, out2.shape, out3.shape, out4.shape)
        #print("Parameter count:", sum(p.numel() for p in model.parameters()))
        
        model = MARNetPretrainingWrapper().to(device).eval()
        imagenet_input = torch.randn(2, 3, 224, 224).to(device)
        out = model(imagenet_input)
        
        print(model)
        
        print("Output shape:", out.shape)
        print("Parameter count:", sum(p.numel() for p in model.parameters()))
    