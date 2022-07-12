from torch import nn
import torch
from torchvision import models
from transformers import RobertaModel


class fusion(nn.Module):
    def __init__(self, vdim, tdim, D):
        super(fusion, self).__init__()
        self.resnet = models.__dict__['resnet50'](pretrained=True)
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.vfc = nn.Linear(vdim, D)
        self.tfc = nn.Linear(tdim, D)
        self.output = nn.Linear(2 * D, 3)

        self.bn = nn.BatchNorm1d(D)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, image, input_ids, attention_mask):
        v_emb = self.resnet(image)
        t_emb = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        t_emb = t_emb.pooler_output

        v_emb = self.drop(self.relu(self.bn(self.vfc(v_emb))))
        t_emb = self.drop(self.relu(self.bn(self.tfc(t_emb))))

        fusion = torch.cat((v_emb, t_emb), axis=1)

        return self.output(fusion)


class single(nn.Module):
    def __init__(self, dim, D):
        super(single, self).__init__()
        self.resnet = models.__dict__['resnet50'](pretrained=True)
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.fc = nn.Linear(dim, D)
        self.output = nn.Linear(D, 3)

        self.bn = nn.BatchNorm1d(D)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, ablation, emb1, emb2 = None):
        if(ablation == 1):
            emb = self.roberta(
                input_ids=emb1,
                attention_mask=emb2
            )
            emb = emb.pooler_output
        else:
            emb = self.resnet(emb1)

        out = self.drop(self.relu(self.bn(self.fc(emb))))
        return self.output(out)
