import argparse
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch.optim as optim
from PIL import Image
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from transformers import RobertaTokenizer

from model_fine_tuning import *

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

img_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)


def standard_txt_transform(tweet):
    proc_tweet = text_processor.pre_process_doc(tweet)
    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]
    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)

def mapfunc(x):
    if x == "negative":
        return 0
    elif x == "neutral":
        return 1
    else:
        return 2

class MVSA_DS(Dataset):
    def __init__(self, dir, img_transform, txt_transform, tokenizer, labels):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.img_transform = img_transform
        self.tokenizer = tokenizer
        self.txt_transform = txt_transform
        self.labels = np.array(labels).astype(np.int)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])
        img = Image.open(os.path.join('data/pics_texts_pair', fname + '.jpg'))
        text = open(os.path.join('data/pics_texts_pair', fname + '.txt'), 'r', encoding='utf-8',
                    errors='ignore').read().strip()

        if self.img_transform:
            img = self.img_transform(img)

        if self.txt_transform:
            text = self.txt_transform(text)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        label = self.labels[idx]

        return img, encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), torch.tensor(label)


def train(hyper_dict, dloc, save_path="saved/saved_models/"):
    # 显示我们需要的超参数
    vdim, tdim = 0, 0
    print(hyper_dict)
    print("using {} device.".format(device))

    # 读取数据集与特征信息
    labels = pd.read_csv("data/train.txt")["tag"].map(mapfunc).to_numpy().flatten()

    # 构建数据集与dataloader
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tr_data = MVSA_DS(dloc, img_transforms, standard_txt_transform, tokenizer, labels)

    tr_size = int(0.9 * len(tr_data))
    vl_size = len(tr_data) - tr_size

    tr_data, vl_data = random_split(tr_data, [tr_size, vl_size],
                                    generator=torch.Generator().manual_seed(RANDOM_SEED))

    tr_loader = DataLoader(dataset=tr_data, batch_size=args.bs, num_workers=2, shuffle=True)
    vl_loader = DataLoader(dataset=vl_data, batch_size=args.bs, num_workers=2)

    # 模型定义
    if hyper_dict.vtype == "resnet":
        vdim = 1000

    if hyper_dict.ttype == "roberta":
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
            images, input_ids, attention_mask, labels = data
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # print(images.shape, input_ids.shape)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                if hyper_dict.ablation == 1:
                    outputs = net(1, input_ids, attention_mask)
                elif hyper_dict.ablation == 2:
                    outputs = net(2, images)
                else:
                    outputs = net(images, input_ids, attention_mask)

                loss = loss_function(outputs, labels)

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
                val_images, val_input_ids, val_attention_mask, val_labels = val_data

                val_images = val_images.to(device)
                val_input_ids = val_input_ids.to(device)
                val_attention_mask = val_attention_mask.to(device)
                val_labels = val_labels.to(device)

                if hyper_dict.ablation == 1:
                    outputs = net(1, val_input_ids, val_attention_mask)
                elif hyper_dict.ablation == 2:
                    outputs = net(2, val_images)
                else:
                    outputs = net(val_images, val_input_ids, val_attention_mask)

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


if __name__ == '__main__':
    # 读取参数设置
    parser = argparse.ArgumentParser(description="training phase")
    parser.add_argument('--vtype', type=str, default='resnet', help='resnet')
    parser.add_argument('--ttype', type=str, default='roberta', help='roberta')
    parser.add_argument('--bs', type=int, default=64, help='32, 64, 128')
    parser.add_argument('--lr', type=float, default='2e-5', help='1e-4, 5e-5, 2e-5')
    parser.add_argument('--ep', type=int, default=50, help='10, 25, 50, 100')
    parser.add_argument('--D', type=int, default=256, help='128, 256')
    parser.add_argument('--ablation', type=int, default=0, help='1: no image, 2: no text')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(RANDOM_SEED)

    # 训练开始
    train(args, dloc='data/train.txt')
