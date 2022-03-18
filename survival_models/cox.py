import sys
sys.path.append('../')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pickle
import  utils
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
import cox_utils
from sksurv.linear_model import CoxPHSurvivalAnalysis



parser = argparse.ArgumentParser(description='Cox-PH survival models')

parser.add_argument('--data_dir', default='.', type=str)
parser.add_argument('--cluster_name', default='gmm_50.pkl', type=str)
parser.add_argument('--normalize', default='mean', type=str)


args = parser.parse_args()

cluster = pickle.load(open(os.path.join(args.data_dir, args.cluster_name), 'rb'))
cluster_method = type(cluster).__name__
if cluster_method == 'GaussianMixture':
    n_clusters = len(cluster.weights_)
else:
    n_clusters = cluster.n_clusters

train_data, val_data, test_data = utils.load_data(args.data_dir, 
                os.path.join(args.data_dir, args.cluster_name), normalize=args.normalize)

train_df, val_df, test_df = utils.preprocess_data(train_data, val_data, test_data)
if data_source == 'TCGA':
    test_df = test_df.loc[test_df['tcga_flag']==1.0]
elif data_source =='CPTAC':
    test_df = test_df.loc[test_df['tcga_flag']==0.0]
train_df, val_df, test_df = train_df.drop(columns=['tcga_flag']), val_df.drop(columns=['tcga_flag']), test_df.drop(columns=['tcga_flag'])
y_train = np.array([tuple((bool(row[0]), row[1])) for row in zip(train_df['outcome'], train_df['day'])],
        dtype=[('outcome', 'bool'), ('day', '<f4')])
y_val = np.array([tuple((bool(row[0]), row[1])) for row in zip(val_df['outcome'], val_df['day'])],
        dtype=[('outcome', 'bool'), ('day', '<f4')])
y_test = np.array([tuple((bool(row[0]), row[1])) for row in zip(test_df['outcome'], test_df['day'])],
        dtype=[('outcome', 'bool'), ('day', '<f4')])
alpha_list = [10**i for i in np.linspace(-3,0,10)]
alpha = alpha_list[0]
val_results = []
for i, alpha in enumerate(alpha_list):
    est = CoxPHSurvivalAnalysis(alpha=alpha).fit(train_df.drop(columns=['outcome','day']), y_train)
    val_metrics = utils.get_metics(train_df, val_df, est)
    val_results.append(val_metrics['C-index'])

alpha = alpha_list[np.argmax(val_results)]
est = CoxPHSurvivalAnalysis(alpha=alpha).fit(train_df.drop(columns=['outcome','day']), y_train)
test_metrics = utils.get_metics(train_df, test_df, est)

pickle.dump({"setting": os.path.join(args.data_dir, args.cluster_name),
            "test_results": test_metrics},
            open(os.path.join(args.data_dir, "test_results.p", 'wb')))