import argparse
import json
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_head import *

# 检测是否使用nvidia的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MVSA_DS(Dataset):
    def __init__(self, vfeats, tfeats):
        self.vfeats = vfeats
        self.tfeats = tfeats

    def __len__(self):
        return len(self.vfeats)

    def __getitem__(self, idx):
        vfeat = self.vfeats[idx]
        tfeat = self.tfeats[idx]
        return torch.FloatTensor(vfeat), torch.FloatTensor(tfeat)


def evaluate(hyper_dict, test_loader):
    # 显示我们需要的超参数
    print(hyper_dict)
    print("using {} device.".format(device))
    net = torch.load(hyper_dict.model_loc)

    # 验证
    net.eval()
    all_preds = []

    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)

        for val_data in val_bar:
            val_images, val_text = val_data
            outputs = net(val_text.to(device), val_text.to(device))
            preds = torch.argmax(outputs.data, 1)
            all_preds.extend(preds.cpu().numpy().flatten())

    return all_preds

def mapfunc(x):
    if x == 0: return "negative"
    elif x == 1: return "neutral"
    else: return "positive"

if __name__ == '__main__':

    # 读取参数设置
    parser = argparse.ArgumentParser(description="inference phase")
    parser.add_argument('--vtype', type=str, default='clip', help='resnet-50 | clip')
    parser.add_argument('--ttype', type=str, default='clip', help='roberta | clip')
    parser.add_argument('--bs', type=int, default=1024, help='128 512 1024')
    parser.add_argument('--model_loc', type=str, default='saved/saved_models/clip.pth',
                        help='saved models\' location')
    args = parser.parse_args()

    # 读取数据集与特征信息
    id = pd.read_csv("data/train_label.csv", header=None)[0].to_numpy().flatten()

    feats_text = json.load(open('saved/saved_feats/%s_test.json' % args.ttype, 'r'))
    feats_img = json.load(open('saved/saved_feats/%s_test.json' % args.vtype, 'r'))

    if args.ttype == 'clip':
        feats_text = feats_text['txt_feats']

    if args.vtype == 'clip':
        feats_img = feats_img['img_feats']

    feats_text = np.array(feats_text)
    feats_img = np.array(feats_img)

    # print(len(feats_text), len(feats_img))

    # 构建数据集与dataloader
    test_data = MVSA_DS(feats_img, feats_text)
    test_loader = DataLoader(dataset=test_data, batch_size=args.bs, num_workers=2)

    # 推理开始
    res = evaluate(args, test_loader)
    # print(res)

    # 给出答案
    df = pd.read_csv("data/test_without_label.txt")
    df["tag"] = res
    df["tag"].map(mapfunc)
    df.to_csv("data/answer.csv", index=False)