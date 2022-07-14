from torch import nn
import torch


class fusion(nn.Module):
    def __init__(self, vdim, tdim, D):
        super(fusion, self).__init__()

        self.vfc = nn.Linear(vdim, D)
        self.tfc = nn.Linear(tdim, D)
        self.output = nn.Linear(2 * D, 3)

        self.bn1 = nn.BatchNorm1d(D)
        self.bn2 = nn.BatchNorm1d(D)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, v_emb, t_emb):
        v_emb = self.drop1(self.relu(self.bn1(self.vfc(v_emb))))
        t_emb = self.drop2(self.relu(self.bn2(self.tfc(t_emb))))
        fusion = torch.cat((v_emb, t_emb), axis=1)

        return self.output(fusion)


class fusionWithCrossAttention(nn.Module):
    def __init__(self, vdim, tdim, D):
        super(fusionWithCrossAttention, self).__init__()
        self.D = D

        self.vfc = nn.Linear(vdim, D)
        self.tfc = nn.Linear(tdim, D)
        self.output = nn.Linear(2 * D, 3)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=D, num_heads=8, batch_first=True)
        self.tk = nn.Linear(D, D)
        self.tv = nn.Linear(D, D)
        self.tq = nn.Linear(D, D)
        self.vk = nn.Linear(D, D)
        self.vv = nn.Linear(D, D)
        self.vq = nn.Linear(D, D)

        self.flat = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(D)
        self.bn2 = nn.BatchNorm1d(D)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, v_emb, t_emb):
        v_emb = self.drop1(self.relu(self.bn1(self.vfc(v_emb))))
        t_emb = self.drop2(self.relu(self.bn2(self.tfc(t_emb))))

        # v_emb = self.vfc(v_emb)
        # t_emb = self.tfc(t_emb)

        tk = self.tk(t_emb)
        tv = self.tv(t_emb)
        tq = self.tq(t_emb)
        vk = self.vk(v_emb)
        vv = self.vv(v_emb)
        vq = self.vq(v_emb)
        key = torch.stack((tk, vk), dim=1)
        value = torch.stack((tv, vv), dim=1)
        query = torch.stack((tq, vq), dim=1)
        fusion, _ = self.multihead_attn(query, key, value)
        fusion = fusion.reshape(-1, 2 * self.D)

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
