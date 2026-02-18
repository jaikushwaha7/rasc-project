import torch
import torch.nn as nn

class RelationshipMLP(nn.Module):
    def __init__(self, num_classes, num_relations, emb_dim=64):
        super().__init__()

        self.embed = nn.Embedding(num_classes, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2 + 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_relations)
        )

    def forward(self, s_cls, s_box, o_cls, o_box):
        s_emb = self.embed(s_cls)
        o_emb = self.embed(o_cls)
        x = torch.cat([s_emb, o_emb, s_box, o_box], dim=1)
        return self.fc(x)
