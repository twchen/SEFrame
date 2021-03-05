import torch as th
import dgl
import numpy as np
from collections import Counter


def label_last(g, last_nid):
    is_last = th.zeros(g.number_of_nodes(), dtype=th.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g


def seq_to_unweighted_graph(seq):
    iid, seq_nid, cnt = np.unique(seq, return_inverse=True, return_counts=True)
    num_nodes = len(iid)

    if len(seq_nid) > 1:
        edges = zip(seq_nid, seq_nid[1:])
        counter = Counter(edges)
        unique_edges = counter.keys()
        src, dst = zip(*unique_edges)
    else:
        src = th.LongTensor([])
        dst = th.LongTensor([])

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = th.LongTensor(iid)
    g.ndata['cnt'] = th.FloatTensor(cnt)
    label_last(g, seq_nid[-1])
    return g


def seq_to_weighted_graph(seq):
    iid, seq_nid, cnt = np.unique(seq, return_inverse=True, return_counts=True)
    num_nodes = len(iid)

    if len(seq_nid) > 1:
        counter = Counter(zip(seq_nid, seq_nid[1:]))
        src, dst = zip(*counter.keys())
        weight = th.FloatTensor(list(counter.values()))
    else:
        src = th.LongTensor([])
        dst = th.LongTensor([])
        weight = th.FloatTensor([])

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = th.LongTensor(iid)
    g.ndata['cnt'] = th.FloatTensor(cnt)
    g.edata['w'] = weight.view(g.num_edges(), 1)
    label_last(g, seq_nid[-1])
    return g
