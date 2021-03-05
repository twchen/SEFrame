import math
import random
import itertools
from collections import Counter

import pandas as pd
import numpy as np

import torch as th
import dgl


class AugmentedDataset:
    def __init__(self, df, read_sid=False, sort_by_length=True):
        if read_sid:
            df = df[['sessionId', 'userId', 'items']]
        else:
            df = df[['userId', 'items']]
        self.sessions = df.values
        session_lens = df['items'].apply(len)
        index = create_index(session_lens)
        if sort_by_length:
            # sort by labelIndex in descending order
            # it is to be used with BatchSampler to make data loading of RNN models faster
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        sidx, lidx = self.index[idx]
        sess = self.sessions[sidx]
        seq = sess[-1][:lidx]
        label = sess[-1][lidx]
        item = (*sess[:-1], seq, label)
        return item

    def __len__(self):
        return len(self.index)


class AnonymousAugmentedDataset:
    def __init__(self, df, sort_by_length=True):
        self.sessions = df['items'].values
        session_lens = np.fromiter(map(len, self.sessions), dtype=np.long)
        index = create_index(session_lens)
        if sort_by_length:
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        sidx, lidx = self.index[idx]
        sess = self.sessions[sidx]
        seq = sess[:lidx]
        label = sess[lidx]
        item = (0, seq, label)
        return item

    def __len__(self):
        return len(self.index)


class BatchSampler:
    """
    First, the sequences of the same length are grouped into the same batch
    Then, the remaining sequences of similar lengths are grouped into the same batch
    the sequences in a batch is sorted by length in desending order
    """
    def __init__(self, augmented_dataset, batch_size, drop_last=False, seed=None):
        df_index = pd.DataFrame(
            augmented_dataset.index, columns=['sessionId', 'labelIdx']
        )
        self.groups = [df for _, df in df_index.groupby('labelIdx')]
        self.groups.sort(key=lambda g: g.iloc[0].labelIdx, reverse=True)
        self.batch_size = batch_size
        num_batches = len(augmented_dataset) / batch_size
        if drop_last:
            self.num_batches = math.floor(num_batches)
        else:
            self.num_batches = math.ceil(num_batches)
        self.drop_last = drop_last
        self.seed = seed

    def _create_batch_indices(self):
        # shuffle sequences of the same length
        groups = [df.sample(frac=1, random_state=self.seed) for df in self.groups]
        df_index = pd.concat(groups)
        # shuffle batches
        batch_indices = [
            df.index
            for _, df in df_index.groupby(np.arange(len(df_index)) // self.batch_size)
        ][:self.num_batches]
        random.seed(self.seed)
        random.shuffle(batch_indices)
        self.batch_indices = batch_indices
        if self.seed is not None:
            self.seed += 1

    def __iter__(self):
        self._create_batch_indices()
        return iter(self.batch_indices)

    def __len__(self):
        return self.num_batches


def create_index(session_lens):
    num_sessions = len(session_lens)
    session_idx = np.repeat(np.arange(num_sessions), session_lens - 1)
    label_idx = map(lambda l: range(1, l), session_lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df['items'] = df['items'].apply(lambda x: [int(i) for i in x.split(',')])
    return df


def read_dataset(dataset_dir):
    stats = pd.read_csv(dataset_dir / 'stats.txt', sep='\t').iloc[0]
    df_train = read_sessions(dataset_dir / 'train.txt')
    df_valid = read_sessions(dataset_dir / 'valid.txt')
    df_test = read_sessions(dataset_dir / 'test.txt')
    return df_train, df_valid, df_test, stats


def read_social_network(csv_file):
    df = pd.read_csv(csv_file, sep='\t')
    g = dgl.graph((df.followee.values, df.follower.values))
    return g


def build_knowledge_graph(df_train, social_network, do_count_clipping=True):
    print('building heterogeneous knowledge graph...')
    followed_edges = social_network.edges()
    clicks = Counter()
    transits = Counter()
    for _, row in df_train.iterrows():
        uid = row['userId']
        seq = row['items']
        for iid in seq:
            clicks[(uid, iid)] += 1
        transits.update(zip(seq, seq[1:]))
    clicks_u, clicks_i = zip(*clicks.keys())
    prev_i, next_i = zip(*transits.keys())
    kg = dgl.heterograph({
        ('user', 'followedby', 'user'): followed_edges,
        ('user', 'clicks', 'item'): (clicks_u, clicks_i),
        ('item', 'clickedby', 'user'): (clicks_i, clicks_u),
        ('item', 'transitsto', 'item'): (prev_i, next_i),
    })
    click_cnts = np.array(list(clicks.values()))
    transit_cnts = np.array(list(transits.values()))
    if do_count_clipping:
        click_cnts = clip_counts(click_cnts)
        transit_cnts = clip_counts(transit_cnts)
    click_cnts = th.LongTensor(click_cnts) - 1
    transit_cnts = th.LongTensor(transit_cnts) - 1
    kg.edges['clicks'].data['cnt'] = click_cnts
    kg.edges['clickedby'].data['cnt'] = click_cnts
    kg.edges['transitsto'].data['cnt'] = transit_cnts
    return kg


def find_max_count(counts):
    max_cnt = np.max(counts)
    density = np.histogram(
        counts, bins=np.arange(1, max_cnt + 2), range=(1, max_cnt + 1), density=True
    )[0]
    cdf = np.cumsum(density)
    for i in range(max_cnt):
        if cdf[i] > 0.95:
            return i + 1
    return max_cnt


def clip_counts(counts):
    """
    Truncate the counts to the maximum value of the smallest 95% counts.
    This could avoid outliers and reduce the number of count embeddings.
    """
    max_cnt = find_max_count(counts)
    counts = np.minimum(counts, max_cnt)
    return counts


def compute_visible_time_list(in_neighbors, zero_hop_visible_time, num_layers):
    visible_time_list = [zero_hop_visible_time]
    num_nodes = len(zero_hop_visible_time)
    for n in range(1, num_layers + 1):
        prev_visible_time = visible_time_list[-1]
        n_hop_visible_time = []
        for node in range(num_nodes):
            if len(in_neighbors[node]) == 0:
                neigh_vis_time = float('inf')
            else:
                neigh_vis_time = min([
                    prev_visible_time[neigh] for neigh in in_neighbors[node]
                ])
            node_vis_time = max(neigh_vis_time, zero_hop_visible_time[node])
            n_hop_visible_time.append(node_vis_time)
        visible_time_list.append(n_hop_visible_time)
    return visible_time_list


def compute_visible_time_list_and_in_neighbors(df_train, dataset_dir, num_layers):
    df_edges = pd.read_csv(dataset_dir / 'edges.txt', sep='\t')
    num_nodes = df_edges.values.max() + 1
    in_neighbors = [[] for i in range(num_nodes)]
    pd_series = df_edges.groupby('follower').followee.apply(list)
    for follower, followees in pd_series.items():
        in_neighbors[follower] = followees

    visible_time = df_train.groupby('userId').sessionId.min().values
    visible_time_list = compute_visible_time_list(
        in_neighbors, visible_time, num_layers
    )
    return visible_time_list, in_neighbors


def filter_invalid_sessions(*dfs, L_hop_visible_time):
    print('filtering invalid sessions')
    dfs_filtered = []
    for df in dfs:
        sids_to_keep = []
        for _, row in df.iterrows():
            sid = row['sessionId']
            uid = row['userId']
            if L_hop_visible_time[uid] <= sid:
                sids_to_keep.append(sid)
        df_filtered = df[df.sessionId.isin(sids_to_keep)]
        dfs_filtered.append(df_filtered)
    return dfs_filtered
