from torch import nn
import torch


class fusion(nn.Module):
    def __init__(self, vdim, tdim, D):
        super(fusion, self).__init__()

        self.vfc = nn.Linear(vdim, D)
        self.tfc = nn.Linear(tdim, D)
        self.output = nn.Linear(2 * D, 3)

        self.bn = nn.BatchNorm1d(D)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, v_emb, t_emb):
        v_emb = self.drop(self.relu(self.bn(self.vfc(v_emb))))
        t_emb = self.drop(self.relu(self.bn(self.tfc(t_emb))))
        fusion = torch.cat((v_emb, t_emb), axis=1)

        return self.output(fusion)

class single(nn.Module):
    def __init__(self, dim, D):
        super(single, self).__init__()

        self.fc = nn.Linear(dim, D)
        self.output = nn.Linear(D, 3)

        self.bn = nn.BatchNorm1d(D)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, emb):

        out = self.drop(self.relu(self.bn(self.fc(emb))))
        return self.output(out)
