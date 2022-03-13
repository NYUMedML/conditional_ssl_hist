import os
import argparse
import pandas as pd
from utils import wsi_to_tiles

parser = argparse.ArgumentParser(description='Process TCGA')

parser.add_argument('--followup_path', default='./LSCC-clinicalTable.csv', type=str)
parser.add_argument('--wsi_path', default='../CPTAC_WSI', type=str)
parser.add_argument('--refer_img', default='./colorstandard.png', type=str)
parser.add_argument('--s', default=0.9, type=float, help='The proportion of tissues')
args = parser.parse_args()

clinicalTable = pd.read_csv(args.followup_path).set_index('case_id')
wsi_dir_dict = {}
wsi_list = os.popen("find {} -name '*.svs'".format(args.wsi_path)).read().strip('\n').replace(wsi_path,'').split("\n")
for slide_id in wsi_list:
    slide_id = slide_id.lstrip('/').rstrip('.svs')
    tile_path = os.path.join('./tiles', slide_id)
    if not os.path.exists(tile_path):
        os.mkdir(tile_path)

for idx, wsi in enumerate(wsi_list):
    wsi_to_tiles(idx, wsi, args.refer_img, args.s)

