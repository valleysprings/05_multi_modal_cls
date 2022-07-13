import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_head import *

# 写入一些日志数据，方便可视化训练情况
writer = SummaryWriter(comment='model_training_for_MM_cls')

# 检测是否使用nvidia的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 混合精度是否启用
use_amp = True

# 可复现性设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

RANDOM_SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)


class MVSA_DS(Dataset):
    def __init__(self, vfeats, tfeats, labels):
        self.vfeats = vfeats
        self.tfeats = tfeats
        self.labels = np.array(labels).astype(np.int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vfeat = self.vfeats[idx]
        tfeat = self.tfeats[idx]
        label = self.labels[idx]
        return torch.FloatTensor(vfeat), torch.FloatTensor(tfeat), torch.tensor(label)


def train(hyper_dict, tr_loader, vl_loader, save_path = "saved/saved_models/"):
    # 显示我们需要的超参数
    vdim, tdim = 0, 0
    print(hyper_dict)
    print("using {} device.".format(device))

    # 模型定义
    if hyper_dict.vtype == "clip":
        vdim = 512
    elif hyper_dict.vtype == "resnet":
        vdim = 1000

    if hyper_dict.ttype == "clip":
        tdim = 512
    elif hyper_dict.ttype == "roberta":
        tdim = 768

    D = hyper_dict.D

    if hyper_dict.ablation == 1:
        net = single(tdim, D).to(device)
    elif hyper_dict.ablation == 2:
        net = single(vdim, D).to(device)
    else:
        net = fusion(vdim, tdim, D).to(device)

    # 选择优化器
    optimizer = optim.Adam(net.parameters(), lr=hyper_dict.lr)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 定义训练与验证过程
    epochs = hyper_dict.ep
    best_acc = 0.0
    train_steps = len(tr_loader)

    # fp16设置
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):

        # 训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(tr_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, texts, labels = data
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                if hyper_dict.ablation == 1:
                    outputs = net(texts.to(device))
                elif hyper_dict.ablation == 2:
                    outputs = net(images.to(device))
                else:
                    outputs = net(images.to(device), texts.to(device))

                loss = loss_function(outputs, labels.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            # 记录每iteration的损失
            writer.add_scalar('training loss', loss,
                              epoch * train_steps + step)

        # 验证
        net.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_bar = tqdm(vl_loader, file=sys.stdout)

            for val_data in val_bar:
                val_images, val_text, val_labels = val_data

                if hyper_dict.ablation == 1:
                    outputs = net(val_text.to(device))
                elif hyper_dict.ablation == 2:
                    outputs = net(val_images.to(device))
                else:
                    outputs = net(val_images.to(device), val_text.to(device))

                preds = torch.argmax(outputs.data, 1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(val_labels.cpu().numpy().flatten())

        val_accurate = metrics.accuracy_score(all_labels, all_preds)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate

        if hyper_dict.ep == epoch + 1 and hyper_dict.ablation == 0:
            save_path += hyper_dict.vtype + '.pth'
            torch.save(net, save_path)

    print('Finished Training')
    print('Best Accuracy:\t %.3f' % (best_acc))

def mapfunc(x):
    if x == "negative": return 0
    elif x == "neutral": return 1
    else: return 2

if __name__ == '__main__':

    # 读取参数设置
    parser = argparse.ArgumentParser(description="training phase")
    parser.add_argument('--vtype', type=str, default='clip', help='resnet | clip')
    parser.add_argument('--ttype', type=str, default='clip', help='roberta | clip')
    parser.add_argument('--bs', type=int, default=4096, help='32, 128, 512')
    parser.add_argument('--lr', type=float, default='1e-4', help='1e-4, 5e-5, 2e-5')
    parser.add_argument('--ep', type=int, default=50, help='50, 100')
    parser.add_argument('--D', type=int, default=256, help='128, 256')
    parser.add_argument('--ablation', type=int, default=0, help='1: no image, 2: no text')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(RANDOM_SEED)

    # 读取数据集与特征信息

    labels = pd.read_csv("data/train.txt")["tag"].map(mapfunc).to_numpy().flatten()
    feats_text = json.load(open('saved/saved_feats/%s_train.json' % args.ttype, 'r'))
    feats_img = json.load(open('saved/saved_feats/%s_train.json' % args.vtype, 'r'))


    feats_text = feats_text['txt_feats']
    feats_img = feats_img['img_feats']
    feats_text = np.array(feats_text)
    feats_img = np.array(feats_img)

    print(len(feats_text), len(feats_img))

    # 获取训练集/验证集/测试集分割
    lab_train, lab_val, ft_tr_txt,  ft_vl_txt, ft_tr_img, ft_vl_img = \
        train_test_split(labels, feats_text, feats_img, test_size=0.1, random_state=RANDOM_SEED)
    lab_train = list(map(int, lab_train))
    lab_val = list(map(int, lab_val))

    # 构建数据集与dataloader
    tr_data = MVSA_DS(ft_tr_img, ft_tr_txt, lab_train)
    vl_data = MVSA_DS(ft_vl_img, ft_vl_txt, lab_val)

    tr_loader = DataLoader(dataset=tr_data, batch_size=args.bs, num_workers=2, shuffle=True)
    vl_loader = DataLoader(dataset=vl_data, batch_size=args.bs, num_workers=2)

    # 训练开始
    train(args, tr_loader, vl_loader)
