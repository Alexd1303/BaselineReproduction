from typing import Optional

import torch
from torch import nn

from Models.Baseline.MARNet import MARNet
from Models.Baseline.MARNetOnly import ConvNet3D

class ContextCombinationModule(nn.Module):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, context_dim)
        self.relu = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, embedding):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x * embedding
        

class TCR_Module(nn.Module):
    def __init__(self, context_dim: int):
        super().__init__()
        self.context_module = ContextCombinationModule(input_dim=1536, context_dim=context_dim)
        self.context_projection = nn.Linear(1536, context_dim)
        
        self.fc = nn.Linear(1536 + context_dim, 3)
    
    def forward(self, combined_features, context_embedding: Optional[torch.Tensor] = None):
        if context_embedding is not None:
            context = self.context_module(combined_features, context_embedding)
        else:
            context = self.context_projection(combined_features)
        
        output = self.fc(torch.cat([combined_features, context], dim=1))

        return output, context


class DER_Module(nn.Module):
    def __init__(self, context_dim: int):
        super().__init__()
        self.context_module = ContextCombinationModule(input_dim=2560, context_dim=context_dim)
        self.context_projection = nn.Linear(2560, context_dim)
        
        self.fc = nn.Linear(2560 + context_dim, 5)

    def forward(self, combined_features, context_embedding=None):
        if context_embedding is not None:
            context = self.context_module(combined_features, context_embedding)
        else:
            context = self.context_projection(combined_features)
        
        output = self.fc(torch.cat([combined_features, context], dim=1))

        return output, context

class DBR_Module(nn.Module):
    def __init__(self, context_dim: int):
            super().__init__()
            self.context_module = ContextCombinationModule(input_dim=2048, context_dim=context_dim)
            self.context_projection = nn.Linear(2048, context_dim)
            
            self.fc = nn.Linear(2048 + context_dim, 7)
    
    def forward(self, combined_features, context_embedding=None):
        if context_embedding is not None:
            context = self.context_module(combined_features, context_embedding)
        else:
            context = self.context_projection(combined_features)
        
        output = self.fc(torch.cat([combined_features, context], dim=1))

        return output, context

class VBR_Module(nn.Module):
    def __init__(self, context_dim: int):
        super().__init__()
        self.context_module = ContextCombinationModule(input_dim=4096, context_dim=context_dim)
        self.context_projection = nn.Linear(4096, context_dim)
        
        self.fc = nn.Linear(4096 + context_dim, 5)
        
    def forward(self, combined_features, context_embedding=None):
        
        if context_embedding is not None:
            context = self.context_module(combined_features, context_embedding)
        else:
            context = self.context_projection(combined_features)
        
        output = self.fc(torch.cat([combined_features, context], dim=1))

        return output, context
        

class CausalNet(nn.Module):
    def __init__(self, context_dim: int = 4096):
        super().__init__()
        self.marnet_left = MARNet()
        self.marnet_front = MARNet()
        self.marnet_right = MARNet()
        self.marnet_inside = MARNet()
        self.marnet_body = MARNet()
        self.marnet_face = MARNet()
        
        self.gesture_conv3d = ConvNet3D(num_keypoints=42)
        self.posture_conv3d = ConvNet3D(num_keypoints=26)
        
        self.TCR_module = TCR_Module(context_dim=context_dim)
        self.DER_module = DER_Module(context_dim=context_dim)
        self.DBR_module = DBR_Module(context_dim=context_dim)
        self.VBR_module = VBR_Module(context_dim=context_dim)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, left, front, right, inside, body, face, gesture, posture):
        left_features = self.marnet_left(left)
        front_features = self.marnet_front(front)
        right_features = self.marnet_right(right)
        inside_features = self.marnet_inside(inside)
        body_features = self.marnet_body(body)
        face_features = self.marnet_face(face)
        
        gesture_features = self.gesture_conv3d(gesture)
        posture_features = self.posture_conv3d(posture)

        left_g = self.avg_pool(left_features).flatten(1)
        front_g = self.avg_pool(front_features).flatten(1)
        right_g = self.avg_pool(right_features).flatten(1)
        inside_g = self.avg_pool(inside_features).flatten(1)
        body_g = self.avg_pool(body_features).flatten(1)
        face_g = self.avg_pool(face_features).flatten(1)

        tcr_combined = torch.cat([left_g, front_g, right_g], dim=1)
        der_combined = torch.cat([inside_g, body_g, face_g, gesture_features, posture_features], dim=1)
        dbr_combined = torch.cat([body_g, face_g, gesture_features, posture_features], dim=1)
        vbr_combined = torch.cat([left_g, front_g, right_g, inside_g, body_g, face_g, gesture_features, posture_features], dim=1)
        
        tcr_output, tcr_context = self.TCR_module(tcr_combined)
        der_output, der_context = self.DER_module(der_combined, context_embedding=tcr_context)
        dbr_output, dbr_context = self.DBR_module(dbr_combined, context_embedding=der_context)
        vbr_output, vbr_context = self.VBR_module(vbr_combined, context_embedding=dbr_context)
        
        return der_output, dbr_output, tcr_output, vbr_output