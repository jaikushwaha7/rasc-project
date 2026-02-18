"""
Relationship prediction models
Includes MLP and Neural Motifs architectures
"""

import torch
import torch.nn as nn


# ============================================================
# MLP MODEL (UNCHANGED)
# ============================================================

class RelationshipMLP(nn.Module):
    """Simple MLP for relationship prediction"""
    
    def __init__(
        self,
        num_classes: int,
        num_relations: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed = nn.Embedding(num_classes, emb_dim)

        input_dim = emb_dim * 2 + 8  # 2 embeddings + 2 boxes

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_relations)
        )
    
    def forward(self, s_cls, s_box, o_cls, o_box):
        s_emb = self.embed(s_cls)
        o_emb = self.embed(o_cls)
        x = torch.cat([s_emb, o_emb, s_box, o_box], dim=1)
        return self.fc(x)


# ============================================================
# ✅ FIXED NEURAL MOTIFS (PAIRWISE VERSION)
# ============================================================

class NeuralMotifs(nn.Module):
    """
    Simplified pairwise Neural Motifs-style model
    Compatible with (s_cls, s_box, o_cls, o_box)
    """

    def __init__(
        self,
        num_classes: int,
        num_relations: int,
        emb_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()

        self.obj_embed = nn.Embedding(num_classes, emb_dim)

        lstm_hidden = emb_dim

        # Context encoder over subject-object pair (sequence length = 2)
        self.context_rnn = nn.LSTM(
            emb_dim + 4,
            lstm_hidden,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden

        # Predicate classifier
        self.rel_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_relations)
        )

    # ========================================================
    # ✅ CHANGED FORWARD SIGNATURE
    # OLD:
    # def forward(self, obj_classes, obj_boxes, pair_indices)
    #
    # NEW:
    # def forward(self, s_cls, s_box, o_cls, o_box)
    # ========================================================

    def forward(self, s_cls, s_box, o_cls, o_box):

        # Embed subject and object
        s_emb = self.obj_embed(s_cls)
        o_emb = self.obj_embed(o_cls)

        # Concatenate embedding + bbox
        s_feat = torch.cat([s_emb, s_box], dim=1)
        o_feat = torch.cat([o_emb, o_box], dim=1)

        # Stack as sequence of length 2
        pair_seq = torch.stack([s_feat, o_feat], dim=1)
        # Shape: [B, 2, emb_dim+4]

        # Context encoding
        context, _ = self.context_rnn(pair_seq)
        # context shape: [B, 2, lstm_output_dim]

        subj_ctx = context[:, 0, :]
        obj_ctx = context[:, 1, :]

        pair_feat = torch.cat([subj_ctx, obj_ctx], dim=1)

        logits = self.rel_classifier(pair_feat)

        return logits


# ============================================================
# FACTORY
# ============================================================

def create_model(
    model_type: str,
    num_classes: int,
    num_relations: int,
    **kwargs
) -> nn.Module:

    if model_type.lower() == 'mlp':
        return RelationshipMLP(num_classes, num_relations, **kwargs)

    elif model_type.lower() in ['neural_motifs', 'motifs']:
        return NeuralMotifs(num_classes, num_relations, **kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

# """
# Relationship prediction models
# Includes MLP and Neural Motifs architectures
# """

# import torch
# import torch.nn as nn
# from typing import Tuple


# class RelationshipMLP(nn.Module):
#     """Simple MLP for relationship prediction"""
    
#     def __init__(
#         self,
#         num_classes: int,
#         num_relations: int,
#         emb_dim: int = 64,
#         hidden_dim: int = 256,
#         dropout: float = 0.1
#     ):
#         """
#         Initialize MLP model
        
#         Args:
#             num_classes: Number of object classes
#             num_relations: Number of relationship types
#             emb_dim: Embedding dimension
#             hidden_dim: Hidden layer dimension
#             dropout: Dropout probability
#         """
#         super().__init__()
        
#         self.num_classes = num_classes
#         self.num_relations = num_relations
#         self.emb_dim = emb_dim
        
#         # Object class embeddings
#         self.embed = nn.Embedding(num_classes, emb_dim)
        
#         # MLP classifier
#         # Input: subject embedding + object embedding + subject bbox + object bbox
#         input_dim = emb_dim * 2 + 8  # 2 bboxes, 4 coords each
        
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, num_relations)
#         )
    
#     def forward(
#         self,
#         s_cls: torch.Tensor,
#         s_box: torch.Tensor,
#         o_cls: torch.Tensor,
#         o_box: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Forward pass
        
#         Args:
#             s_cls: Subject class indices [B]
#             s_box: Subject bounding boxes [B, 4]
#             o_cls: Object class indices [B]
#             o_box: Object bounding boxes [B, 4]
            
#         Returns:
#             Relationship logits [B, num_relations]
#         """
#         s_emb = self.embed(s_cls)
#         o_emb = self.embed(o_cls)
#         x = torch.cat([s_emb, o_emb, s_box, o_box], dim=1)
#         return self.fc(x)


# class NeuralMotifs(nn.Module):
#     """
#     Neural Motifs model for relationship prediction
#     Based on "Neural Motifs: Scene Graph Parsing with Global Context" (Zellers et al., CVPR 2018)
#     """
    
#     def __init__(
#         self,
#         num_classes: int,
#         num_relations: int,
#         emb_dim: int = 128,
#         hidden_dim: int = 512,
#         dropout: float = 0.1,
#         bidirectional: bool = True
#     ):
#         """
#         Initialize Neural Motifs model
        
#         Args:
#             num_classes: Number of object classes
#             num_relations: Number of relationship types
#             emb_dim: Embedding dimension
#             hidden_dim: Hidden dimension for classifier
#             dropout: Dropout probability
#             bidirectional: Use bidirectional LSTM
#         """
#         super().__init__()
        
#         self.num_classes = num_classes
#         self.num_relations = num_relations
#         self.emb_dim = emb_dim
#         self.bidirectional = bidirectional
        
#         # Object class embeddings
#         self.obj_embed = nn.Embedding(num_classes, emb_dim)
        
#         # Context encoder (LSTM over objects)
#         # Input: object embedding + bbox (4 coords)
#         lstm_hidden = emb_dim
#         self.context_rnn = nn.LSTM(
#             emb_dim + 4,
#             lstm_hidden,
#             batch_first=True,
#             bidirectional=bidirectional
#         )
        
#         # Output dimension after LSTM
#         lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        
#         # Predicate classifier
#         # Input: concatenated subject and object contexts
#         self.rel_classifier = nn.Sequential(
#             nn.Linear(lstm_output_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, num_relations)
#         )
#     def forward(self, s_cls, s_box, o_cls, o_box):

#         obj_classes = torch.cat([s_cls, o_cls], dim=0)
#         obj_boxes = torch.cat([s_box, o_box], dim=0)

#         B = s_cls.size(0)

#         pair_indices = torch.stack([
#             torch.arange(B, device=s_cls.device),
#             torch.arange(B, device=s_cls.device) + B
#         ], dim=1)

#         # ---- original logic ----
#         obj_emb = self.obj_embed(obj_classes)
#         obj_feat = torch.cat([obj_emb, obj_boxes], dim=1)
#         obj_feat = obj_feat.unsqueeze(0)

#         context, _ = self.context_rnn(obj_feat)
#         context = context.squeeze(0)

#         subj_ctx = context[pair_indices[:, 0]]
#         obj_ctx = context[pair_indices[:, 1]]

#         pair_feat = torch.cat([subj_ctx, obj_ctx], dim=1)
#         logits = self.rel_classifier(pair_feat)

#         return logits
#     # def forward(
#     #     self,
#     #     obj_classes: torch.Tensor,
#     #     obj_boxes: torch.Tensor,
#     #     pair_indices: torch.Tensor
#     # ) -> torch.Tensor:
#     #     """
#     #     Forward pass
        
#     #     Args:
#     #         obj_classes: Object class indices [N]
#     #         obj_boxes: Object bounding boxes [N, 4]
#     #         pair_indices: Pairs of object indices [M, 2] where M is number of pairs
            
#     #     Returns:
#     #         Relationship logits [M, num_relations]
#     #     """
#     #     # Get object embeddings and concatenate with bounding boxes
#     #     obj_emb = self.obj_embed(obj_classes)  # [N, emb_dim]
#     #     obj_feat = torch.cat([obj_emb, obj_boxes], dim=1)  # [N, emb_dim + 4]
        
#     #     # Add batch dimension for LSTM
#     #     obj_feat = obj_feat.unsqueeze(0)  # [1, N, emb_dim + 4]
        
#     #     # Run through LSTM to get contextualized features
#     #     context, _ = self.context_rnn(obj_feat)  # [1, N, lstm_output_dim]
#     #     context = context.squeeze(0)  # [N, lstm_output_dim]
        
#     #     # Get subject and object contexts for each pair
#     #     subj_ctx = context[pair_indices[:, 0]]  # [M, lstm_output_dim]
#     #     obj_ctx = context[pair_indices[:, 1]]   # [M, lstm_output_dim]
        
#     #     # Concatenate and classify
#     #     pair_feat = torch.cat([subj_ctx, obj_ctx], dim=1)  # [M, lstm_output_dim * 2]
#     #     logits = self.rel_classifier(pair_feat)  # [M, num_relations]
        
#     #     return logits


# def create_model(
#     model_type: str,
#     num_classes: int,
#     num_relations: int,
#     **kwargs
# ) -> nn.Module:
#     """
#     Factory function to create relationship models
    
#     Args:
#         model_type: Type of model ('mlp' or 'neural_motifs')
#         num_classes: Number of object classes
#         num_relations: Number of relationship types
#         **kwargs: Additional model-specific arguments
        
#     Returns:
#         Initialized model
#     """
#     if model_type.lower() == 'mlp':
#         return RelationshipMLP(num_classes, num_relations, **kwargs)
#     elif model_type.lower() in ['neural_motifs', 'motifs']:
#         return NeuralMotifs(num_classes, num_relations, **kwargs)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")
