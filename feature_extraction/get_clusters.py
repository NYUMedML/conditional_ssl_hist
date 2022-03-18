import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pickle
import numpy as np
import sys



parser = argparse.ArgumentParser(description='Get cluster features')

parser.add_argument('--data_dir', default='./', type=str)
parser.add_argument('--cluster_type', default='gmm', type=str)
parser.add_argument('--n_cluster', default=50, type=int)
parser.add_argument('--out_dir', default='./', type=str)


args = parser.parse_args()

train_features = pickle.load(open(args.data_dir + 'train_embedding.pkl', 'rb'))
train_features = np.concatenate(list(train_features.values()), axis=0)
if args.cluster_type == "kmeans":
    print('kmeans')
    cluster = KMeans(n_clusters=n_cluster).fit(train_features)
    pickle.dump(cluster, open(data_dir + 'kmeans_{}.pkl'.format(args.n_cluster), 'wb'))
else:
    print('gmm')
    cluster = GaussianMixture(n_components=n_cluster).fit(train_features)
    pickle.dump(cluster, open(data_dir + 'gmm_{}.pkl'.format(args.n_cluster), 'wb'))


