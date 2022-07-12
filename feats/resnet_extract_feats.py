import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

img_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class preDataset(Dataset):
    def __init__(self, dir, img_transform=None):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.img_transform = img_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])
        img = Image.open(os.path.join('data/pics_texts_pair', fname + '.jpg'))

        if self.img_transform:
            img = self.img_transform(img)

        return img, fname


def get_resnet_feats(dloc):
    feats = []

    model = models.__dict__['resnet50'](pretrained=True)
    model.eval().to(device)

    dataset = preDataset(dloc, img_transform=img_transforms)
    dt_loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=8)

    for i, batch in enumerate(dt_loader):
        print("processing:\t %d / %d " % (i + 1, len(dt_loader)))
        img_inputs = batch[0].to(device)

        with torch.no_grad():
            outputs = model(img_inputs)

        feats.extend(outputs.view(-1, outputs.shape[1]).data.cpu().numpy().tolist())

    return feats


if __name__ == '__main__':
    dloc = 'data/train.txt'
    image_feats = get_resnet_feats(dloc)
    json.dump({'img_feats': image_feats},
              open('saved/saved_feats/resnet_train.json', 'w'))

    dloc = 'data/test_without_label.txt'
    image_feats = get_resnet_feats(dloc)
    json.dump({'img_feats': image_feats},
              open('saved/saved_feats/resnet_test.json', 'w'))
