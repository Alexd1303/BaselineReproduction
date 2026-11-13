from abc import ABC, abstractmethod
from typing import Optional, override, Type, Any, Tuple

import torch
from torch import nn

from Models.Baseline.MARNet import MARNet
from Models.Baseline.MARNetOnly import ConvNet3D

from Models.Encoders import Encoder, MobileNetV3Encoder, Resnet18Encoder, MARNetEncoder


class ContextCombinationModule(nn.Module, ABC):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim

    @abstractmethod
    def forward(self, x, embedding) -> Any:
        ...


class FiLMContextCombination(ContextCombinationModule):
    """
    FiLM (Feature-wIse Linear Modulation) context combination module.
    This module applies feature-wise linear modulation to the input features based on the context embedding.
    
    Args:
        input_dim (int): Dimension of the input features (d_B).
        context_dim (int): Dimension of the context embedding (d_A).
        hidden (int, optional): Dimension of the hidden layer. If None, it will be set to the same as context_dim.
        gamma_scale (float): Scale factor for the gamma parameter in FiLM.
    """
    def __init__(self, input_dim, context_dim, hidden=None, gamma_scale=1.0):
        super().__init__(input_dim, context_dim)
        hidden = hidden or input_dim
        self.net = nn.Sequential(
            #nn.Linear(context_dim, hidden),
            #nn.GELU(),
            nn.Linear(context_dim, 2 * input_dim),
        )
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.gamma_scale = gamma_scale

        # Identity initialization: at the beginning, the input is not altered by the context
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
        self.projection = nn.Linear(input_dim, context_dim)

    @override
    def forward(self, x, embedding):
        gb = self.net(embedding)                       # [B, 2*d_B]
        gamma, beta = gb.chunk(2, dim=-1)        # [B, d_B] each
        gamma = self.gamma_scale * torch.tanh(gamma)   # limits the modulation

        if x.dim() == 3:                       # sequence: broadcast over tokens
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)

        x = self.norm(x)                     # normalize before modulating
        out = self.projection((1 + gamma) * x + beta)
        return out


class CrossAttentionContextCombination(ContextCombinationModule):
    def __init__(self, input_dim, context_dim, num_heads=4, feture_dim=512):
        super().__init__(input_dim, context_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=feture_dim, num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(feture_dim, context_dim)

    @override
    def forward(self, x, embedding):
        N = self.input_dim // 512
        x = x.reshape(x.size(0), N, 512)

        # embedding = self.context_projection(embedding)
        embedding = embedding.unsqueeze(1)  # Add sequence dimension for attention

        attn_output, _ = self.cross_attention(query=embedding, key=x, value=x, need_weights=False)

        attn_output = attn_output.squeeze(1)  # Remove sequence dimension

        out = self.projection(attn_output)
        return out


class DFEContextCombination(ContextCombinationModule):
    def __init__(self, input_dim, context_dim):
        super().__init__(input_dim, context_dim)
        self.fc1 = nn.Linear(input_dim, context_dim)
        self.relu = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()

    @override
    def forward(self, x, embedding):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x * embedding
    
class GRUContextCombination(ContextCombinationModule):
    def __init__(self, input_dim, context_dim):
        super().__init__(input_dim, context_dim)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=context_dim, batch_first=True)
    
    @override
    def forward(self, x, embedding):
        x = x.unsqueeze(1)
        
        embedding = embedding.unsqueeze(0)
        
        output, _ = self.gru(x, embedding)
        
        output = output.squeeze(1)
        
        return output


class LSTMContextCombination(ContextCombinationModule):
    def __init__(self, input_dim, context_dim):
        super().__init__(input_dim, context_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=context_dim, batch_first=True)
    
    @override
    def forward(self, x, embedding):
        x = x.unsqueeze(1)
        
        embedding = embedding.unsqueeze(0)
        
        output, _ = self.lstm(x, (embedding, torch.zeros_like(embedding)))
        
        output = output.squeeze(1)
        
        return output    

class Causal_Module(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, context_dim: int, alpha_blend: bool = False, alpha_projection_dim: int = -1, contextCombinationModule: Type[ContextCombinationModule] = GRUContextCombination):
        super().__init__()
        self.context_module = contextCombinationModule(input_dim=input_dim, context_dim=context_dim)
        self.context_projection = nn.Linear(input_dim, context_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc: nn.Linear
        
        self.alpha = None
        if alpha_blend:
            assert alpha_projection_dim > 0, "alpha_projection_dim must be greater than 0 when alpha_blend is True"
            self.features_alpha_projection = nn.Linear(input_dim, alpha_projection_dim)
            self.context_alpha_projection = nn.Linear(context_dim, alpha_projection_dim)
            self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.normalization = nn.LayerNorm(alpha_projection_dim)
            self.fc = nn.Linear(alpha_projection_dim, output_dim)
        else:
            self.fc = nn.Linear(input_dim + context_dim, output_dim)
    
    def forward(self, combined_features, context_embedding: Optional[torch.Tensor] = None):
        if context_embedding is not None:
            context = self.context_module(combined_features, context_embedding)
        else:
            context = self.context_projection(combined_features)
        
        context = self.dropout(context)
        combined_features = self.dropout(combined_features)
        
        if self.alpha is None:
            output = self.fc(torch.cat([combined_features, context], dim=1))
        else:
            alpha = torch.sigmoid(self.alpha)
            output = self.fc(alpha * self.normalization(self.features_alpha_projection(combined_features)) + (1 - alpha) * self.normalization(self.context_alpha_projection(context)))

        return output, context
        

class CausalNet(nn.Module):
    def __init__(self, context_dim: int, use_joints_data: bool = True, alpha_blend: bool = False, encoder: Type[Encoder] = MARNetEncoder, pretrained_encoder: bool = False, freeze_encoder: bool = False, contextCombinationModule: Type[ContextCombinationModule] = GRUContextCombination):
        super().__init__()        
        self.encoder = encoder(pretrained=pretrained_encoder, freeze=freeze_encoder)
        
        self.freeze_encoder = freeze_encoder
        
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.use_joints_data = use_joints_data
        if self.use_joints_data:
            self.gesture_conv3d = ConvNet3D(num_keypoints=42)
            self.posture_conv3d = ConvNet3D(num_keypoints=26)
            
        
        self.TCR_module = Causal_Module(input_dim=1536, output_dim=3, context_dim=context_dim, alpha_blend=alpha_blend, alpha_projection_dim=512, contextCombinationModule=contextCombinationModule)
        
        if self.use_joints_data:
            self.DER_module = Causal_Module(input_dim=2560, output_dim=5, context_dim=context_dim, alpha_blend=alpha_blend, alpha_projection_dim=512, contextCombinationModule=contextCombinationModule)
            self.DBR_module = Causal_Module(input_dim=2048, output_dim=7, context_dim=context_dim, alpha_blend=alpha_blend, alpha_projection_dim=1024, contextCombinationModule=contextCombinationModule)
            self.VBR_module = Causal_Module(input_dim=4096, output_dim=5, context_dim=context_dim, alpha_blend=alpha_blend, alpha_projection_dim=1024, contextCombinationModule=contextCombinationModule)
        else:
            self.DER_module = Causal_Module(input_dim=1536, output_dim=5, context_dim=context_dim, alpha_blend=alpha_blend, alpha_projection_dim=512, contextCombinationModule=contextCombinationModule)
            self.DBR_module = Causal_Module(input_dim=1024, output_dim=7, context_dim=context_dim, alpha_blend=alpha_blend, alpha_projection_dim=1024, contextCombinationModule=contextCombinationModule)
            self.VBR_module = Causal_Module(input_dim=3072, output_dim=5, context_dim=context_dim, alpha_blend=alpha_blend, alpha_projection_dim=1024, contextCombinationModule=contextCombinationModule)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inside, front, left, right, face, body, gesture, posture):
        left_features = self.encoder(left)
        front_features = self.encoder(front)
        right_features = self.encoder(right)
        inside_features = self.encoder(inside)
        body_features = self.encoder(body)
        face_features = self.encoder(face)

        left_g = self.avg_pool(left_features).flatten(1)
        front_g = self.avg_pool(front_features).flatten(1)
        right_g = self.avg_pool(right_features).flatten(1)
        inside_g = self.avg_pool(inside_features).flatten(1)
        body_g = self.avg_pool(body_features).flatten(1)
        face_g = self.avg_pool(face_features).flatten(1)
        
        if self.use_joints_data:
            gesture_features = self.gesture_conv3d(gesture)
            posture_features = self.posture_conv3d(posture)
            tcr_combined = torch.cat([left_g, front_g, right_g], dim=1)
            der_combined = torch.cat([inside_g, body_g, face_g, gesture_features, posture_features], dim=1)
            dbr_combined = torch.cat([body_g, face_g, gesture_features, posture_features], dim=1)
            vbr_combined = torch.cat([left_g, front_g, right_g, inside_g, body_g, face_g, gesture_features, posture_features], dim=1)
        else:
            tcr_combined = torch.cat([left_g, front_g, right_g], dim=1)
            der_combined = torch.cat([inside_g, body_g, face_g], dim=1)
            dbr_combined = torch.cat([body_g, face_g], dim=1)
            vbr_combined = torch.cat([left_g, front_g, right_g, inside_g, body_g, face_g], dim=1)
        
        tcr_output, tcr_context = self.TCR_module(tcr_combined)
        der_output, der_context = self.DER_module(der_combined, context_embedding=tcr_context)
        dbr_output, dbr_context = self.DBR_module(dbr_combined, context_embedding=der_context)
        vbr_output, vbr_context = self.VBR_module(vbr_combined, context_embedding=dbr_context)
        
        return der_output, dbr_output, tcr_output, vbr_output
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self


class CausalNetLateFusion(CausalNet):
    def __init__(self, context_dim: int, use_joints_data: bool = True, alpha_blend: bool = False, encoder: Type[Encoder] = MARNetEncoder, pretrained_encoder: bool = False, freeze_encoder: bool = False, contextCombinationModule: Type[ContextCombinationModule] = GRUContextCombination):
        super().__init__(context_dim, use_joints_data, alpha_blend, encoder, pretrained_encoder, freeze_encoder, contextCombinationModule)

    def forward(self, inside, front, left, right, face, body, gesture, posture):
        der_outputs = []
        dbr_outputs = []
        tcr_outputs = []
        vbr_outputs = []

        for frame_n in range(0, 48, 3):
            inside_frame = inside[:, frame_n: frame_n + 3]
            front_frame = front[:, frame_n: frame_n + 3]
            left_frame = left[:, frame_n: frame_n + 3]
            right_frame = right[:, frame_n: frame_n + 3]
            face_frame = face[:, frame_n: frame_n + 3]
            body_frame = body[:, frame_n: frame_n + 3]
            gesture_frame = gesture[:, frame_n: frame_n + 3] if self.use_joints_data else None
            posture_frame = posture[:, frame_n: frame_n + 3] if self.use_joints_data else None

            der_output, dbr_output, tcr_output, vbr_output = super().forward(inside_frame, front_frame, left_frame, right_frame, face_frame, body_frame, gesture_frame, posture_frame)
            der_outputs.append(der_output)
            dbr_outputs.append(dbr_output)
            tcr_outputs.append(tcr_output)
            vbr_outputs.append(vbr_output)

        der_output = torch.stack(der_outputs, dim=1).mean(dim=1)
        dbr_output = torch.stack(dbr_outputs, dim=1).mean(dim=1)
        tcr_output = torch.stack(tcr_outputs, dim=1).mean(dim=1)
        vbr_output = torch.stack(vbr_outputs, dim=1).mean(dim=1)

        return der_output, dbr_output, tcr_output, vbr_output

class CausalNetLearnedChain(CausalNet):
    def __init__(self, context_dim: int, use_joints_data: bool = True, alpha_blend: bool = False, encoder: Type[Encoder] = MARNetEncoder, pretrained_encoder: bool = False, freeze_encoder: bool = False, contextCombinationModule: Type[ContextCombinationModule] = GRUContextCombination):
        super().__init__(context_dim, use_joints_data, alpha_blend, encoder, pretrained_encoder, freeze_encoder, contextCombinationModule)

        self.multi_task_context_combination = contextCombinationModule(input_dim=context_dim, context_dim=context_dim)

    def forward(self, inside, front, left, right, face, body, gesture, posture):
        left_features = self.encoder(left)
        front_features = self.encoder(front)
        right_features = self.encoder(right)
        inside_features = self.encoder(inside)
        body_features = self.encoder(body)
        face_features = self.encoder(face)

        left_g = self.avg_pool(left_features).flatten(1)
        front_g = self.avg_pool(front_features).flatten(1)
        right_g = self.avg_pool(right_features).flatten(1)
        inside_g = self.avg_pool(inside_features).flatten(1)
        body_g = self.avg_pool(body_features).flatten(1)
        face_g = self.avg_pool(face_features).flatten(1)
        
        if self.use_joints_data:
            gesture_features = self.gesture_conv3d(gesture)
            posture_features = self.posture_conv3d(posture)
            tcr_combined = torch.cat([left_g, front_g, right_g], dim=1)
            der_combined = torch.cat([inside_g, body_g, face_g, gesture_features, posture_features], dim=1)
            dbr_combined = torch.cat([body_g, face_g, gesture_features, posture_features], dim=1)
            vbr_combined = torch.cat([left_g, front_g, right_g, inside_g, body_g, face_g, gesture_features, posture_features], dim=1)
        else:
            tcr_combined = torch.cat([left_g, front_g, right_g], dim=1)
            der_combined = torch.cat([inside_g, body_g, face_g], dim=1)
            dbr_combined = torch.cat([body_g, face_g], dim=1)
            vbr_combined = torch.cat([left_g, front_g, right_g, inside_g, body_g, face_g], dim=1)

        vbr_output, vbr_context = self.VBR_module(vbr_combined)
        dbr_output, dbr_context = self.DBR_module(dbr_combined, context_embedding=vbr_context)
        der_output, der_context = self.DER_module(der_combined, context_embedding=dbr_context)

        vbr_dbr_context = self.multi_task_context_combination(dbr_context, embedding=vbr_context)

        tcr_output, tcr_context = self.TCR_module(tcr_combined, context_embedding=vbr_dbr_context)
        
        return der_output, dbr_output, tcr_output, vbr_output

class TaskSpecificNet(nn.Module):
    def __init__(self, input_dim: int, class_num: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, class_num)

    def forward(self, x):
        return self.fc(x)

class FlowNet(nn.Module):
    def __init__(self, encoder: Type[Encoder] = MobileNetV3Encoder, pretrained_encoder: bool = False, freeze_encoder: bool = False):
        super().__init__()

        self.encoder = encoder(pretrained=pretrained_encoder, freeze=freeze_encoder)
                        
        self.freeze_encoder = freeze_encoder

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.tcr_net = TaskSpecificNet(input_dim=3072, class_num=3)
        self.der_net = TaskSpecificNet(input_dim=3072, class_num=5)
        self.dbr_net = TaskSpecificNet(input_dim=3072, class_num=7)
        self.vbr_net = TaskSpecificNet(input_dim=3072, class_num=5)

    def forward(self, inside, front, left, right, face, body, gesture, posture):
        left_features = self.encoder(left)
        front_features = self.encoder(front)
        right_features = self.encoder(right)
        inside_features = self.encoder(inside)
        body_features = self.encoder(body)
        face_features = self.encoder(face)

        left_g = self.avg_pool(left_features).flatten(1)
        front_g = self.avg_pool(front_features).flatten(1)
        right_g = self.avg_pool(right_features).flatten(1)
        inside_g = self.avg_pool(inside_features).flatten(1)
        body_g = self.avg_pool(body_features).flatten(1)
        face_g = self.avg_pool(face_features).flatten(1)

        tcr_combined = torch.cat([left_g, front_g, right_g], dim=1)
        der_combined = torch.cat([inside_g, body_g, face_g], dim=1)
        dbr_combined = torch.cat([body_g, face_g], dim=1)
        vbr_combined = torch.cat([left_g, front_g, right_g, inside_g, body_g, face_g], dim=1)

        der_output = self.der_net(vbr_combined)
        dbr_output = self.dbr_net(vbr_combined)
        tcr_output = self.tcr_net(vbr_combined)
        vbr_output = self.vbr_net(vbr_combined)

        return der_output, dbr_output, tcr_output, vbr_output