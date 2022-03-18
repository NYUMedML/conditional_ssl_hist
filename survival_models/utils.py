import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, concordance_index_censored, integrated_brier_score


def get_metics(train_df, test_df, est):
    metrics = {}
    y_train = np.array([tuple((bool(row[0]), row[1])) for row in zip(train_df['outcome'], train_df['day'])],
         dtype=[('outcome', 'bool'), ('day', '<f4')])
    y_test = np.array([tuple((bool(row[0]), row[1])) for row in zip(test_df['outcome'], test_df['day'])],
             dtype=[('outcome', 'bool'), ('day', '<f4')])
    survs = est.predict_survival_function(test_df.drop(columns=['outcome','day']))
    try:
        times = np.arange(4, 10)
        preds = np.asarray([[fn(t) for t in times] for fn in survs])
    except:
        print('no five year records')
    preds = [fn(4) for fn in survs]
    times, score = brier_score(y_train, y_test, preds, 4)
    metrics['Brier_2yr'] = score[0]
    try:
        preds = [fn(10) for fn in survs]
        times, score = brier_score(y_train, y_test, preds, 10)
        metrics['Brier_5yr'] = score[0]
    except:
        print('no five year records')
    preds = est.predict(test_df.drop(columns=['outcome','day']))
    metrics['C-index'] = concordance_index_censored(y_train, y_test, preds)[0]
    return metrics


def preprocess_data(train_data, val_data, test_data):
    if 'tcga' not in train_data:
        train_df = pd.DataFrame(np.concatenate([train_data['cluster'], np.stack([train_data['recur_day'], train_data['followup_day'], train_data['outcome']]).T], axis=1),columns=['c_{}'.format(i) for i in range(50)] + ['recur', 'followup', 'outcome'])
    else:
        train_df = pd.DataFrame(np.concatenate([train_data['cluster'], np.stack([train_data['recur_day'], train_data['followup_day'], train_data['outcome']]).T], axis=1),columns=['c_{}'.format(i) for i in range(50)] + ['recur', 'followup', 'outcome'])
    train_df = train_df[(train_df['recur'].notna() | train_df['followup'].notna())]
    train_df['day'] = train_df['recur']
    train_df.loc[train_df['recur'].isna(), 'day'] = train_df.loc[train_df['recur'].isna(), 'followup']
    train_df = train_df.drop(columns=['recur', 'followup'])
    train_df = train_df[train_df['day'] > 0]
    train_df['day'] = train_df['day'] // 180 + 1

    if 'tcga' not in val_data:
        val_df = pd.DataFrame(np.concatenate([val_data['cluster'], np.stack([val_data['recur_day'], val_data['followup_day'], val_data['outcome']]).T], axis=1),
                           columns=['c_{}'.format(i) for i in range(50)] + ['recur', 'followup', 'outcome'])
    else:
        val_df = pd.DataFrame(np.concatenate([val_data['cluster'], np.stack([val_data['recur_day'], val_data['followup_day'], val_data['outcome']]).T], axis=1),columns=['c_{}'.format(i) for i in range(50)] + ['recur', 'followup', 'outcome'])
    val_df = val_df[(val_df['recur'].notna() | val_df['followup'].notna())]
    val_df['day'] = val_df['recur']
    val_df.loc[val_df['recur'].isna(), 'day'] = val_df.loc[val_df['recur'].isna(), 'followup']
    val_df = val_df.drop(columns=['recur', 'followup'])
    val_df = val_df[val_df['day'] > 0]
    val_df['day'] = val_df['day'] // 180 + 1

    
    if 'tcga' not in test_data:
        test_df = pd.DataFrame(np.concatenate([test_data['cluster'], np.stack([test_data['recur_day'], test_data['followup_day'], test_data['outcome']]).T], axis=1),
                               columns=['c_{}'.format(i) for i in range(50)] + ['recur', 'followup', 'outcome'])
    else:
        test_df = pd.DataFrame(np.concatenate([test_data['cluster'], np.stack([test_data['recur_day'], test_data['followup_day'], test_data['outcome']]).T], axis=1),
                               columns=['c_{}'.format(i) for i in range(50)] + ['recur', 'followup', 'outcome'])

    test_df['day'] = test_df['recur']
    test_df = test_df[(test_df['recur'].notna() | test_df['followup'].notna())]

    test_df.loc[test_df['recur'].isna(), 'day'] = test_df.loc[test_df['recur'].isna(), 'followup']
    test_df = test_df.drop(columns=['recur', 'followup'])
    test_df = test_df[test_df['day'] > 0]
    test_df['day'] = test_df['day'] // 180 + 1
    return train_df, val_df, test_df
    

def label_cluster(feature, cluster):
    clusters = defaultdict()
    cluster_method = type(cluster).__name__
    for k in tqdm(list(feature.keys())):
        if cluster_method == 'GaussianMixture':
            clusters[k] = cluster.predict_proba(feature[int(k)])
        else:
            clusters[k] = cluster.predict(feature[int(k)])
    return clusters


def load_data(data_dir, cluster_dir, normalize='mean', cls=1):
    split_dir = data_dir.rsplit('/', 1)[0] + '/'
    cluster = pickle.load(open(cluster_dir, 'rb'))
    cluster_method = type(cluster).__name__
    if cluster_method == 'GaussianMixture':
        n_clusters = len(cluster.weights_)
    else:
        n_clusters = cluster.n_clusters
    train_features = pickle.load(open(data_dir + 'train_embedding.pkl', 'rb'))
    train_outcomes = pickle.load(open(data_dir + 'train_outcomes.pkl', 'rb'))
    val_features = pickle.load(open(data_dir + 'val_embedding.pkl', 'rb'))
    val_outcomes = pickle.load(open(data_dir + 'val_outcomes.pkl', 'rb'))
    test_features = pickle.load(open(data_dir + 'test_embedding.pkl', 'rb'))
    test_outcomes = pickle.load(open(data_dir + 'test_outcomes.pkl', 'rb'))

    val_tcga_flag = np.array([])
    test_tcga_flag = np.array([])
    

    train_cluster = label_cluster(train_features, cluster)
    val_cluster = label_cluster(val_features, cluster)
    test_cluster = label_cluster(test_features, cluster)

    train_data = transform(train_features, train_cluster, train_outcomes, train_tcga_flag,  n_clusters, normalize, demo=False)
    val_data = transform(val_features, val_cluster, val_outcomes, val_tcga_flag, n_clusters, normalize, demo=False)
    test_data = transform(test_features, test_cluster, test_outcomes, test_tcga_flag, n_clusters, normalize, demo=False)

    return train_data, val_data, test_data


def counter(arr, n):
    count = defaultdict(lambda: 0)
    for k, v in Counter(arr).items():
        count[k] = v
    return [count[i] for i in range(n)]


def transform(features, cluster, outcomes, tcga_flag, n_clusters, normalize='count', cls=1, weight=None, demo=None):
    count_list = []
    outcome_list = []
    recur_day_list = []
    followup_day_list = []
    raw_featuer_list = []
    tile_outcome = []
    demo_list = []
    tcga_flag_list = []
    cluster_method = type(cluster).__name__
    for k in cluster:
        for v in features[int(k)]:
            raw_featuer_list.append(v)
            tile_outcome.append(outcomes[int(k)])
        if normalize == 'mean':
            count_list.append(cluster[int(k)].mean(axis=0))
        else:
            count_list.append(counter(cluster[int(k)], n_clusters))
    
        outcome_list.append(outcomes[int(k)]['recurrence'])
        recur_day_list.append(outcomes[int(k)]['recurrence_free_days'])
        followup_day_list.append(outcomes[int(k)]['followup_days'])
    count_list, outcome_list = np.array(count_list), np.array(outcome_list)
    count_list = count_list + 1e-10
    
    if normalize == 'mean':
        cluster_features = (count_list.T/count_list.sum(axis=1)).T
    elif normalize == 'count':
        cluster_features = (count_list.T/count_list.sum(axis=1)).T
    elif normalize == 'onehot':
        cluster_features = (count_list > 1e-5)
    elif normalize == 'avg':
        cluster_features = count_list
    elif normalize == 'weight':
        cluster_features = count_list * weight
    elif normalize == 'sum':
        cluster_features = count_list
    return {'cluster': cluster_features,
            'tile_feat': np.array(raw_featuer_list),
            'tile_outcome': np.array(tile_outcome),
            'recur_day': np.array(recur_day_list),
            'followup_day': np.array(followup_day_list),
            'outcome': outcome_list,
            'demo': demo_list,
            'tcga': np.array(tcga_flag_list),
            }