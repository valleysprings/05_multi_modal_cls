import json
import os
import re

import pandas as pd
import torch
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, AutoModel

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
    def __init__(self, dir, txt_transform=None):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.txt_transform = txt_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])

        text = open(os.path.join('data/pics_texts_pair', fname + '.txt'), 'r', encoding='utf-8',
                    errors='ignore').read().strip()

        if self.txt_transform:
            text = self.txt_transform(text)

        return text, fname


def standard_txt_transform(tweet):
    proc_tweet = text_processor.pre_process_doc(tweet)
    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]
    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)


def get_bert_embeddings(dloc, process_tweet=None):
    txt_feats = []

    model_path = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
    model.to(device).eval()

    dataset = preDataset(dloc, txt_transform=process_tweet)
    dt_loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=8)

    for i, batch in enumerate(dt_loader):
        print("processing:\t %d / %d " % (i + 1, len(dt_loader)))
        txt_emb = batch[0]

        # txt_emb = torch.tensor([tokenizer.encode(txt_emb, add_special_tokens=True)]).to(device)
        txt_emb = tokenizer(txt_emb, return_tensors="pt").to(device)

        with torch.no_grad():
            # outputs = model(txt_emb)
            outputs = model(**txt_emb)
            text_features = outputs.pooler_output
            txt_feats.extend(text_features.cpu().numpy().tolist())

    return txt_feats

if __name__ == '__main__':
    dloc = 'data/train.txt'
    text_feats= get_bert_embeddings(dloc, standard_txt_transform)
    json.dump({'txt_feats': text_feats},
              open('saved/saved_feats/deberta_train.json', 'w'))

    dloc = 'data/test_without_label.txt'
    text_feats= get_bert_embeddings(dloc, standard_txt_transform)
    json.dump({'txt_feats': text_feats},
              open('saved/saved_feats/deberta_test.json', 'w'))
