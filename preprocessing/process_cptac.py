import os
import argparse
import pandas as pd
from utils import wsi_to_tiles
import pickle

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
    tile_path = os.path.join('../CPTAC/tiles', slide_id)
    if not os.path.exists(tile_path):
        os.mkdir(tile_path)

for idx, wsi in enumerate(wsi_list):
    wsi_to_tiles(idx, wsi, args.refer_img, args.s)

# Get annotation
annotation = {}
for case_id in clinicalTable.index:
    clinicalRow = clinicalTable.loc[case_id].to_dict()
    try:
        imageRow = imageTable.loc[case_id].to_dict(orient='list')
        slide_id = imageRow['Slide_ID']
    except:
        slide_id = []
    annotation[case_id] = {'recurrence': clinicalRow['Recurrence.status..1..yes..0..no.'],
                           'stage': stage_dict[clinicalRow['baseline.tumor_stage_pathological']],
                           'survival_days': clinicalRow['Overall.survival..days'],
                           'survival': clinicalRow['Survival.status..1..dead..0..alive.'],
                           'recurrence_free_days':clinicalRow['Recurrence.free.survival..days'],
                           'age':clinicalRow['consent.age'],
                           'gender':clinicalRow['consent.sex'],
                           'followup_days':clinicalRow['follow.up.number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_last_contact'],
                          'slide_id': slide_id}
pickle.dump(annotation, open('../CPTAC/recurrence_annotation.pkl', 'wb'))

