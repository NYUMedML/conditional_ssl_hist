import sys
sys.path.append('../')
import argparse
from inception_v4 import InceptionV4
import numpy as np
import pickle
import torch
import torch.nn as nn
import os
import cv2
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from PIL import Image
import shelve

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)


def get_embeddings_bagging(feature_extractor, subtype_model, data_set):
    embedding_dict = defaultdict(list)
    outcomes_dict = defaultdict(list)
    feature_extractor.eval()
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=256, shuffle=False, num_workers=torch.cuda.device_count())
    with torch.no_grad():
        count = 0
        for batch in tqdm(data_loader, position=0, leave=True):
            count += 1
            img, _, bag_idx = batch
            feat = feature_extractor(img.to(device)).cpu()
            subtype_model.eval()
            subtype_prob = subtype_model(img)
            subtype_pred = torch.argmax(subtype_prob, dim=1)
            tumor_idx = (subtype_pred != 0)
            feat = feat[tumor_idx].numpy()
            bag_idx = bag_idx[tumor_idx]
            for i in range(len(bag_idx)):
                embedding_dict[bag_idx[i].item()].append(feat[i][np.newaxis,:])
                slide_id = data_set.idx2slide[bag_idx[i].item()]
                if "TCGA" in slide_id:
                    case_id = '-'.join(slide_id.split('-', 3)[:3])
                else:
                    case_id = slide_id.rsplit('-', 1)[0]
                outcomes_dict[bag_idx[i].item()] = annotations[case_id]
        for k in embedding_dict:
            embedding_dict[k] = np.concatenate(embedding_dict[k], axis=0)
    return embedding_dict, outcomes_dict

def load_pretrained(net, model_dir):

    print(model_dir)
    checkpoint = torch.load(model_dir)
    model_state_dict = {k.replace("module.encoder_q.", ""): v for k, v in checkpoint['state_dict'].items() if
                        "encoder_q" in k}
    net.load_state_dict(model_state_dict)
    net.last_linear = nn.Identity()

parser = argparse.ArgumentParser(description='Extract embeddings ')

parser.add_argument('--feature_extractor_dir', default='./pretrained/checkpoint.pth.tar', type=str)
parser.add_argument('--subtype_model_dir', default='./subtype_cls/checkpoint.pth.tar', type=str)
parser.add_argument('--root_dir', type=str)
parser.add_argument('--split_dir', type=str)
parser.add_argument('--out_dir', type=str)

args = parser.parse_args()

tcga_annotation = pickle.load(open('../TCGA/recurrence_annotation.pkl', 'rb'))
cptac_annotation = pickle.load(open('../CPTAC/recurrence_annotation.pkl', 'rb'))
annotations = {**tcga_annotation, **cptac_annotation}
feature_extractor = InceptionV4(num_classes=256)
load_pretrained(feature_extractor, args.feature_extractor_dir)
feature_extractor.to('cuda')
featurizer = nn.DataParallel(model, device_ids=device_ids)

subtype_model = InceptionV4(num_classes=2).to('cuda')
subtype_model.load_state_dict(torch.load(args.subtype_model_dir))
subtype_model = nn.DataParallel(subtype_model, device_ids=device_ids)


train_dataset = WSI_Dataset(args.root_dir, args.split_dir, 'train')
val_dataset = WSI_Dataset(args.root_dir, args.split_dir, 'val')
test_dataset = WSI_Dataset(args.root_dir, args.split_dir, 'test')


with torch.no_grad():
    names = ['train', 'val', 'test']
    for name, data_set in zip(names, [train_dataset, val_dataset, test_dataset]):
        print(name)
        embedding_dict, outcomes_dict = get_embeddings_bagging(feature_extractor, subtype_model, data_set)
        pickle.dump(embedding_dict, open("{}_{}_embedding.pkl".format(args.out_dir), 'wb'), protocol=4)
        pickle.dump((outcomes_dict), open("{}_{}_outcomes.pkl".format(args.out_dir), 'wb'), protocol=4)

