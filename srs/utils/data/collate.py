import random
from bisect import bisect_left
import numpy as np
import pandas as pd
import torch as th
import dgl


def collate_fn_for_rnn_cnn(samples):
    uids, seqs, labels = zip(*samples)
    uids = th.LongTensor(uids)
    labels = th.LongTensor(labels)

    seqs = list(map(lambda seq: th.LongTensor(seq), seqs))
    padded_seqs = th.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    lens = th.LongTensor(list(map(len, seqs)))
    inputs = uids, padded_seqs, lens
    return inputs, labels


def collate_fn_for_gnn_factory(*seq_to_graph_fns):
    def collate_fn(samples):
        uids, seqs, labels = zip(*samples)
        uids = th.LongTensor(uids)
        labels = th.LongTensor(labels)

        inputs = [uids]
        for seq_to_graph in seq_to_graph_fns:
            graphs = [seq_to_graph(seq) for seq in seqs]
            bg = dgl.batch(graphs)
            inputs.append(bg)
        return inputs, labels

    return collate_fn


def sample_blocks(g, uniq_uids, uniq_iids, fanouts, steps):
    seeds = {'user': th.LongTensor(uniq_uids), 'item': th.LongTensor(uniq_iids)}
    blocks = []
    for fanout in fanouts:
        if fanout <= 0:
            frontier = dgl.in_subgraph(g, seeds)
        else:
            frontier = dgl.sampling.sample_neighbors(
                g, seeds, fanout, copy_ndata=False, copy_edata=True
            )
        block = dgl.to_block(frontier, seeds)
        seeds = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes}
        blocks.insert(0, block)
    return blocks, seeds


class CollateFnRNNCNN:
    def __init__(self, knowledge_graph, num_layers, num_neighbors, **kwargs):
        self.knowledge_graph = knowledge_graph
        self.num_layers = num_layers
        # num_neighbors is a list of integers
        if len(num_neighbors) != num_layers:
            assert len(num_neighbors) == 1
            self.fanouts = num_neighbors * num_layers
        else:
            self.fanouts = num_neighbors

    def _collate_fn(self, samples, fanouts):
        uids, seqs, labels = zip(*samples)

        batch_size = len(seqs)
        lens = list(map(len, seqs))
        max_len = max(lens)

        iids = np.concatenate(seqs)
        new_iids, uniq_iids = pd.factorize(iids, sort=True)
        padded_seqs = np.zeros((batch_size, max_len), dtype=np.long)
        cur_idx = 0
        for i, seq in enumerate(seqs):
            padded_seqs[i, :len(seq)] = new_iids[cur_idx:cur_idx + len(seq)]
            cur_idx += len(seq)

        new_uids, uniq_uids = pd.factorize(uids, sort=True)

        extra_inputs = sample_blocks(
            self.knowledge_graph, uniq_uids, uniq_iids, fanouts, self.num_layers
        )

        new_uids = th.LongTensor(new_uids)
        padded_seqs = th.from_numpy(padded_seqs)
        lens = th.LongTensor(lens)
        labels = th.LongTensor(labels)
        inputs = new_uids, padded_seqs, lens
        return (inputs, extra_inputs), labels

    def collate_train(self, samples):
        return self._collate_fn(samples, self.fanouts)

    def collate_test(self, samples):
        inputs, labels = collate_fn_for_rnn_cnn(samples)
        return (inputs, ), labels

    def collate_test_otf(self, samples):
        return self._collate_fn(samples, [0] * self.num_layers)


class CollateFnGNN:
    def __init__(
        self, knowledge_graph, num_layers, num_neighbors, seq_to_graph_fns, **kwargs
    ):
        self.knowledge_graph = knowledge_graph
        self.num_layers = num_layers
        self.seq_to_graph_fns = seq_to_graph_fns
        # num_neighbors is a list of integers
        if len(num_neighbors) != num_layers:
            assert len(num_neighbors) == 1
            self.fanouts = num_neighbors * num_layers
        else:
            self.fanouts = num_neighbors

    def _collate_fn(self, samples, fanouts):
        uids, seqs, labels = zip(*samples)

        new_uids, uniq_uids = pd.factorize(uids, sort=True)
        new_uids = th.LongTensor(new_uids)
        labels = th.LongTensor(labels)

        iids = np.concatenate(seqs)
        new_iids, uniq_iids = pd.factorize(iids, sort=True)
        cur_idx = 0
        new_seqs = []
        for i, seq in enumerate(seqs):
            new_seq = new_iids[cur_idx:cur_idx + len(seq)]
            cur_idx += len(seq)
            new_seqs.append(new_seq)

        inputs = [new_uids]
        for seq_to_graph in self.seq_to_graph_fns:
            graphs = [seq_to_graph(seq) for seq in new_seqs]
            bg = dgl.batch(graphs)
            inputs.append(bg)

        extra_inputs = sample_blocks(
            self.knowledge_graph, uniq_uids, uniq_iids, fanouts, self.num_layers
        )
        return (inputs, extra_inputs), labels

    def collate_train(self, samples):
        return self._collate_fn(samples, self.fanouts)

    def collate_test(self, samples):
        uids, seqs, labels = zip(*samples)

        uids = th.LongTensor(uids)
        labels = th.LongTensor(labels)

        inputs = [uids]
        for seq_to_graph in self.seq_to_graph_fns:
            graphs = [seq_to_graph(seq) for seq in seqs]
            bg = dgl.batch(graphs)
            inputs.append(bg)

        return (inputs, ), labels

    def collate_test_otf(self, samples):
        return self._collate_fn(samples, [0] * self.num_layers)


class CollateFnDGRec:
    def __init__(
        self, visible_time_list, in_neighbors, uid2sessions, num_layers, num_neighbors,
        **kwargs
    ):
        """
        Args
        ----
        visible_time_list:
            `visible_time_list[l][i]` is the time t when user i has a l-hop neighbor
            such that every user along the path from the neighbor to user i has
            generated a session at or before time t.
        in_neighbors:
            `in_neighbors[i]` is the user ids of the incomping neighbors of user i.
        uid2sessions:
            `uid2sessions[i]` is a list of all training sessions generated by user i,
            sorted in ascending order by session id. Since a session with a smaller
            session id has an earlier end time, the sessions are also sorted in
            ascending order by end time.
        num_layers: int
            The number of graph attention layers
        num_neighbors: list[int]
            `len(num_neighbors)` should be either num_layers or 1
            `num_neighbors[l]` is the number of sampled neighbors at layer l.
            If `len(num_neighbors)` is 1, then the number of sampled neighbors is the
            same in all layers.
        """
        self.visible_time_list = visible_time_list[:-1]
        self.in_neighbors = in_neighbors
        self.uid2sessions = uid2sessions
        # num_neighbors is a list of integers
        if len(num_neighbors) != num_layers:
            assert len(num_neighbors) == 1
            self.fanouts = num_neighbors * num_layers
            # repeat num_neighbors[0] num_layers times.
        else:
            self.fanouts = num_neighbors

    def sample_sessions(self, sid, uid, seq, all_uids):
        """
        Args
        ----
        sid: int
            The session id of the current session
        uid: int
            The user id of the current session
        seq: list[int]
            The prefix of the current session
        all_uids: list[int]
            All user ids in the (sampled) L-hop neighborhood of the current
            user, including the current user id

        Returns
        -------
        sessions: list
            A list of sessions, where the i-th session is the latest session,
            i.e., the last session happened before the current session, of
            the i-th user in all_uids
        """
        sessions = []
        for neigh_uid in all_uids:
            if neigh_uid == uid:
                sessions.append((uid, seq))
            else:
                sids = self.uid2sessions[neigh_uid]['sids']
                idx = bisect_left(sids, sid)
                assert idx > 0
                session = self.uid2sessions[neigh_uid]['sessions'][idx - 1]
                sessions.append((neigh_uid, session))
        return sessions

    def sample_blocks(self, sid, uid):
        """
        Args
        ----
        sid : int
            The session id of the current session
        uid : int
            The user id of the current session

        Returns
        -------
        blocks : list[DGLGraph]
            A list of bipartite graphs. `blocks[-i]` is a graph from the sampled
            users at the i-th layer to the sampled users at the (i-1)-th layer,
            for 1 <= i <= L. (the sampled user at the 0-th layer contains uid only)
            The target nodes in `blocks[i]` are included at the beginning of
            the source nodes in `blocks[i]`.
        all_uids : list[int]
            All user ids in the (sampled) L-hop neighborhood of the current user.
            The first entry is `uid`.
        """
        blocks = []
        seeds = [uid]
        nid_map = {uid: 0}
        for fanout, visible_time in zip(self.fanouts, self.visible_time_list[::-1]):
            src = []
            dst = []
            for i, node in enumerate(seeds):
                candidates = [
                    neigh for neigh in self.in_neighbors[node]
                    if visible_time[neigh] <= sid
                ]
                assert len(candidates) > 0
                if len(candidates) <= fanout:
                    sampled_neighs = candidates
                else:
                    sampled_neighs = random.sample(candidates, fanout)
                for neigh in sampled_neighs:
                    if neigh not in nid_map:
                        nid_map[neigh] = len(nid_map)
                    src.append(nid_map[neigh])
                dst += [i] * len(sampled_neighs)
            block = dgl.heterograph(
                data_dict={('followee', 'followedby', 'follower'): (src, dst)},
                num_nodes_dict={
                    'followee': len(nid_map),
                    'follower': len(seeds)
                }
            )
            blocks.insert(0, block)
            seeds = list(nid_map.keys())
        return blocks, seeds

    def collate_train(self, samples):
        batch_blocks = []
        batch_sessions = []
        labels = []
        # the uid is the first src node of block.
        cur_sidx = [0]
        # cur_sidx[b]: the index (in batch_sessions) of the b-th ongoing session
        # in the current batch
        for sid, uid, seq, label in samples:
            blocks, all_uids = self.sample_blocks(sid, uid)
            sessions = self.sample_sessions(sid, uid, seq, all_uids)
            batch_blocks.append(blocks)
            batch_sessions += sessions
            labels.append(label)
            cur_sidx.append(cur_sidx[-1] + len(sessions))
        graphs = []
        # graphs[i]: a graph that batches the blocks of all samples at the i-th layer
        idx_maps = []
        # idx_maps[i]: the indices of the target nodes of graphs[i]
        # in the source nodes of graphs[i]
        for blocks in zip(*batch_blocks):
            graphs.append(dgl.batch(blocks))
            idx_map = []
            total_number_of_src_nodes = 0
            for block in blocks:
                idx = th.arange(block.number_of_dst_nodes()) + total_number_of_src_nodes
                idx_map.append(idx)
                total_number_of_src_nodes += block.number_of_src_nodes()
            idx_map = th.cat(idx_map)
            idx_maps.append(idx_map)
        uids, seqs = zip(*batch_sessions)
        tensor_seqs = list(map(lambda seq: th.LongTensor(seq) + 1, seqs))
        padded_seqs = th.nn.utils.rnn.pad_sequence(tensor_seqs, batch_first=True)
        lens = list(map(len, tensor_seqs))

        uids = th.LongTensor(uids)
        lens = th.LongTensor(lens)
        cur_sidx = th.LongTensor(cur_sidx[:-1])
        inputs = [graphs, idx_maps, uids, padded_seqs, lens, cur_sidx]
        labels = th.LongTensor(labels)
        return inputs, labels

    collate_test = collate_train
