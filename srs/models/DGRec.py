import torch as th
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import dgl.ops as F

from srs.utils.Dict import Dict
from srs.utils.data.collate import CollateFnDGRec
from srs.utils.prepare_batch import prepare_batch_factory_recursive


class GAT(nn.Module):
    def __init__(
        self,
        qry_feats,
        key_feats,
        val_feats,
        feat_drop=0.0,
        batch_norm=False,
    ):
        super().__init__()
        if batch_norm:
            self.batch_norm_q = nn.BatchNorm1d(qry_feats)
            self.batch_norm_k = nn.BatchNorm1d(key_feats)
        else:
            self.batch_norm_q = None
            self.batch_norm_k = None
        self.feat_drop = nn.Dropout(feat_drop)

        self.fc = nn.Linear(qry_feats, val_feats, bias=True)

        self.qry_feats = qry_feats

    def forward(self, g, feat_src, feat_dst):
        if self.batch_norm_q is not None:
            feat_src = self.batch_norm_q(feat_src)
            feat_dst = self.batch_norm_k(feat_dst)
        if self.feat_drop is not None:
            feat_src = self.feat_drop(feat_src)
            feat_dst = self.feat_drop(feat_dst)
        score = F.u_dot_v(g, feat_src, feat_dst)  # (num_edges, 1)
        weight = F.edge_softmax(g, score)
        rst = F.u_mul_e_sum(g, feat_src, weight)
        rst = th.relu(self.fc(rst))
        return rst


class DGRec(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        num_layers,
        batch_norm=False,
        feat_drop=0.0,
        residual=True,
        **kwargs,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, max_norm=1)
        self.item_embeeding = nn.Embedding(
            num_items + 1, embedding_dim, max_norm=1, padding_idx=0
        )
        self.item_indices = nn.Parameter(
            th.arange(1, num_items + 1, dtype=th.long), requires_grad=False
        )
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.lstm = nn.LSTM(embedding_dim, embedding_dim)
        self.W1 = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.layers = nn.ModuleList()
        input_dim = embedding_dim
        for _ in range(num_layers):
            layer = GAT(
                input_dim,
                input_dim,
                embedding_dim,
                batch_norm=batch_norm,
                feat_drop=feat_drop,
            )
            if not residual:
                input_dim += embedding_dim
            self.layers.append(layer)
        self.residual = residual
        self.W2 = nn.Linear(input_dim + embedding_dim, embedding_dim, bias=False)

    def forward(self, graphs, idx_maps, uids, padded_seqs, lens, cur_sidx):
        emb_seqs = self.item_embeeding(padded_seqs)
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)
        packed_seqs = pack_padded_sequence(
            emb_seqs, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed_seqs)

        long_term = self.user_embedding(uids)
        short_term = hn.squeeze(0)
        cur_u_short_term = short_term[cur_sidx]
        feat = th.cat((long_term, short_term), dim=1)
        feat = th.relu(self.W1(feat))
        # the node features of the current user are only the short-term interests
        # the node features of neighbors are the combination of short-term and long-term interests
        feat[cur_sidx] = cur_u_short_term
        for g, idx_map, layer in zip(graphs, idx_maps, self.layers):
            feat_src = feat
            feat_dst = feat[idx_map]
            feat = layer(g, feat_src, feat_dst)
            if self.residual:
                feat = feat_dst + feat
            else:
                feat = th.cat((feat_dst, feat), dim=1)
        sr = self.W2(th.cat((cur_u_short_term, feat), dim=1))
        logits = sr @ self.item_embeeding(self.item_indices).t()

        return logits


config = Dict({
    'Model': DGRec,
    'CollateFn': CollateFnDGRec,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})
