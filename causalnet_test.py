import torch
from torch import nn

from Models import TCR_Module, DER_Module, DBR_Module, VBR_Module, CausalNet, BaselineNet
 
if __name__ == "__main__":    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    context_size = 1250
        
    with torch.no_grad():
        """        
        tcr_module = TCR_Module(context_dim=context_size).eval().to(device, non_blocking=True)
        
        tcr_combined_input = torch.randn(24, 1536).to(device, non_blocking=True)
        
        output, context = tcr_module(tcr_combined_input)
        
        print("TCR_Module output shape:", output.shape)
        print("TCR_Module combined features shape:", context.shape)
        print("TCR_Module parameters:", sum(p.numel() for p in tcr_module.parameters() if p.requires_grad))
        
        der_module = DER_Module(context_dim=context_size).eval().to(device, non_blocking=True)
        
        der_combined_input = torch.randn(24, 2048).to(device, non_blocking=True)

        output, context = der_module(der_combined_input, context_embedding=context)

        print("DER_Module output shape:", output.shape)
        print("DER_Module combined features shape:", context.shape)
        print("DER_Module parameters:", sum(p.numel() for p in der_module.parameters() if p.requires_grad))
        dbr_module = DBR_Module(context_dim=context_size).eval().to(device, non_blocking=True)
        
        dbr_combined_input = torch.randn(24, 2048).to(device, non_blocking=True)
        
        output, context = dbr_module(dbr_combined_input, context_embedding=context)
        
        print("DBR_Module output shape:", output.shape)
        print("DBR_Module combined features shape:", context.shape)
        print("DBR_Module parameters:", sum(p.numel() for p in dbr_module.parameters() if p.requires_grad))

        vbr_module = VBR_Module(context_dim=context_size).eval().to(device, non_blocking=True)
        
        vbr_combined_input = torch.randn(24, 3584).to(device, non_blocking=True)
        
        output, context = vbr_module(vbr_combined_input, context_embedding=context)
        
        print("VBR_Module output shape:", output.shape)
        print("VBR_Module combined features shape:", context.shape)
        print("VBR_Module parameters:", sum(p.numel() for p in vbr_module.parameters() if p.requires_grad))
        """
        
        causalNet_model = CausalNet(context_dim=context_size).eval().to(device, non_blocking=True)
        
        left = torch.randn(24, 48, 224, 224).to(device)
        front = torch.randn(24, 48, 224, 224).to(device)
        right = torch.randn(24, 48, 224, 224).to(device)
        inside = torch.randn(24, 48, 224, 224).to(device)
        body = torch.randn(24, 48, 224, 224).to(device)
        face = torch.randn(24, 48, 224, 224).to(device)
        gesture = torch.randn(24, 3, 16, 42, 1).to(device)
        posture = torch.randn(24, 3, 16, 26, 1).to(device)

        tcr_output, der_output, dbr_output, vbr_output = causalNet_model(left, front, right, inside, body, face, gesture, posture)

        print("CausalNet TCR output shape:", tcr_output.shape)
        print("CausalNet DER output shape:", der_output.shape)
        print("CausalNet DBR output shape:", dbr_output.shape)
        print("CausalNet VBR output shape:", vbr_output.shape)
        
        baseline_model = BaselineNet().eval().to(device, non_blocking=True)
        
        print(f"CausalNet total parameters:{sum(p.numel() for p in causalNet_model.parameters() if p.requires_grad):,}")
        #print(f"CausalNet total parameters:{sum(p.numel() for p in tcr_module.parameters() if p.requires_grad) + sum(p.numel() for p in der_module.parameters() if p.requires_grad) + sum(p.numel() for p in dbr_module.parameters() if p.requires_grad) + sum(p.numel() for p in vbr_module.parameters() if p.requires_grad):,}")
        print(f"BaselineNet parameters:{sum(p.numel() for p in baseline_model.parameters() if p.requires_grad):,}")