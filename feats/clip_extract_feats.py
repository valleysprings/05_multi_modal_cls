import os, re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import json
import clip
from PIL import Image

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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


class preDataset(Dataset):
    def __init__(self, dir, img_transform=None, txt_transform=None):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.img_transform = img_transform
        self.txt_transform = txt_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])
        img = Image.open(os.path.join('data/pics_texts_pair', fname + '.jpg'))
        text = open(os.path.join('data/pics_texts_pair', fname + '.txt'), 'r', encoding='utf-8',
                    errors='ignore').read().strip()
        tag = str(self.file_names.iloc[idx, 1])

        if self.img_transform:
            img = self.img_transform(img)

        if self.txt_transform:
            text = self.txt_transform(text)

        return img, text, tag


def standard_txt_transform(tweet):
    proc_tweet = text_processor.pre_process_doc(tweet)
    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]
    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)


def get_clip_feats(dloc, process_tweet=None):
    img_feats, txt_feats, tags = [], [], []

    model, img_preprocess = clip.load('ViT-B/16', device=device)
    # model, img_preprocess = clip.load('ViT-L/14', device=device)
    model.eval()

    dataset = preDataset(dloc, img_transform=img_preprocess, txt_transform=process_tweet)
    dt_loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=8)

    for i, batch in enumerate(dt_loader):
        print("processing:\t %d / %d " % (i + 1, len(dt_loader)))
        img_emb, txt_emb, tag = batch[0].to(device), batch[1], batch[2]
        tags.append(str(tag[0]))

        txt_emb = clip.tokenize(txt_emb).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img_emb)
            text_features = model.encode_text(txt_emb)

            img_feats.extend(image_features.cpu().numpy().tolist())
            txt_feats.extend(text_features.cpu().numpy().tolist())

    return img_feats, txt_feats, tags


if __name__ == '__main__':

    dloc = 'data/train.txt'
    img_feats, text_feats, tags = get_clip_feats(dloc, standard_txt_transform)
    json.dump({'img_feats': img_feats, 'txt_feats': text_feats, 'tags': tags},
              open('saved/saved_feats/clip_train.json', 'w'))

    dloc = 'data/test_without_label.txt'
    img_feats, text_feats, _ = get_clip_feats(dloc, standard_txt_transform)
    json.dump({'img_feats': img_feats, 'txt_feats': text_feats},
              open('saved/saved_feats/clip_test.json', 'w'))
