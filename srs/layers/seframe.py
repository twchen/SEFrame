import logging
from collections import defaultdict

import torch as th
from torch import nn
import dgl
import dgl.ops as F
from dgl.nn.pytorch import edge_softmax


class HomoAttentionAggregationLayer(nn.Module):
    def __init__(
        self,
        qry_feats,
        key_feats,
        val_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        activation=None,
        batch_norm=True,
    ):
        super().__init__()
        if batch_norm:
            self.batch_norm_q = nn.BatchNorm1d(qry_feats)
            self.batch_norm_k = nn.BatchNorm1d(key_feats)
        else:
            self.batch_norm_q = None
            self.batch_norm_k = None

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.fc_q = nn.Linear(qry_feats, val_feats, bias=True)
        self.fc_k = nn.Linear(key_feats, val_feats, bias=False)
        self.fc_v = nn.Linear(qry_feats, val_feats, bias=False)
        self.attn_e = nn.Parameter(
            th.randn(1, val_feats, dtype=th.float), requires_grad=True
        )
        self.activation = activation

        self.val_feats = val_feats
        self.num_heads = num_heads
        self.head_feats = val_feats // num_heads

    def extra_repr(self):
        return '\n'.join([
            f'num_heads={self.num_heads}', f'(attn_e): Parameter(1, {self.val_feats})'
        ])

    def forward(self, g, ft_q, ft_k, ft_e=None, return_ev=False):
        if self.batch_norm_q is not None:
            ft_q = self.batch_norm_q(ft_q)
            ft_k = self.batch_norm_k(ft_k)
        q = self.fc_q(self.feat_drop(ft_q))
        k = self.fc_k(self.feat_drop(ft_k))
        v = self.fc_v(self.feat_drop(ft_q)).view(-1, self.num_heads, self.head_feats)
        e = F.u_add_v(g, q, k)
        if ft_e is not None:
            e = e + ft_e
        e = (self.attn_e * th.sigmoid(e)).view(-1, self.num_heads, self.head_feats).sum(
            -1, keepdim=True
        )
        if return_ev:
            return e, v
        a = self.attn_drop(edge_softmax(g, e))
        rst = F.u_mul_e_sum(g, v, a).view(-1, self.val_feats)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class HeteroAttentionAggregationLayer(nn.Module):
    def __init__(
        self,
        kg,
        embedding_dim,
        num_heads=1,
        batch_norm=True,
        feat_drop=0.0,
        relu=False,
    ):
        super().__init__()
        self.batch_norm = nn.ModuleDict() if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.edge_aggregate = nn.ModuleDict()
        self.edge_embedding = nn.ModuleDict()
        self.linear_agg = nn.ModuleDict()
        self.linear_self = nn.ModuleDict()
        self.activation = nn.ModuleDict()
        self.vtype2eutypes = defaultdict(list)
        for utype, etype, vtype in kg.canonical_etypes:
            self.edge_aggregate[etype] = HomoAttentionAggregationLayer(
                embedding_dim,
                embedding_dim,
                embedding_dim,
                num_heads=num_heads,
                batch_norm=False,
                feat_drop=0.0,
                activation=None,
            )
            if 'cnt' in kg.edges[etype].data:
                num_cnt_embeddings = kg.edges[etype].data['cnt'].max() + 1
                self.edge_embedding[etype] = nn.Embedding(
                    num_cnt_embeddings, embedding_dim
                )
            self.vtype2eutypes[vtype].append((etype, utype))
        for vtype in self.vtype2eutypes:
            self.linear_agg[vtype] = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.linear_self[vtype] = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )
            self.activation[vtype] = nn.ReLU() if relu else nn.PReLU(embedding_dim)
        if self.batch_norm is not None:
            self.batch_norm.update({
                vtype: nn.BatchNorm1d(embedding_dim)
                for vtype in self.vtype2eutypes
            })

    def forward(self, g, ft_src):
        if self.batch_norm is not None:
            ft_src = {ntype: self.batch_norm[ntype](ft) for ntype, ft in ft_src.items()}
        if self.feat_drop is not None:
            ft_src = {ntype: self.feat_drop(ft) for ntype, ft in ft_src.items()}
        device = next(iter(ft_src.values())).device
        ft_dst = {
            vtype: ft_src[vtype][:g.number_of_dst_nodes(vtype)]
            for vtype in g.dsttypes
        }
        feats = {}
        for vtype, eutypes in self.vtype2eutypes.items():
            src_nid = []
            dst_nid = []
            num_utypes_nodes = 0
            src_val = []
            attn_score = []
            for etype, utype in eutypes:
                sg = g[etype]
                ft_e = (
                    self.edge_embedding[etype](sg.edata['cnt'].to(device))
                    if etype in self.edge_embedding else None
                )
                e, v = self.edge_aggregate[etype](
                    sg,
                    ft_src[utype],
                    ft_dst[vtype],
                    ft_e=ft_e,
                    return_ev=True,
                )
                uid, vid = sg.all_edges(form='uv', order='eid')
                src_nid.append(uid + num_utypes_nodes)
                dst_nid.append(vid)
                num_utypes_nodes += sg.number_of_src_nodes()
                src_val.append(v)
                attn_score.append(e)
            src_nid = th.cat(src_nid, dim=0)
            dst_nid = th.cat(dst_nid, dim=0)
            edge_softmax_g = dgl.heterograph(
                data_dict={('utypes', 'etypes', 'vtype'): (src_nid, dst_nid)},
                num_nodes_dict={
                    'utypes': num_utypes_nodes,
                    'vtype': g.number_of_dst_nodes(vtype)
                },
                device=device
            )
            src_val = th.cat(src_val, dim=0)  # (num_utypes_nodes, num_heads, num_feats)
            attn_score = th.cat(attn_score, dim=0)  # (num_edges, num_heads, 1)
            attn_weight = F.edge_softmax(edge_softmax_g, attn_score)
            agg = F.u_mul_e_sum(edge_softmax_g, src_val, attn_weight)
            agg = agg.view(g.number_of_dst_nodes(vtype), -1)
            feats[vtype] = self.activation[vtype](
                self.linear_agg[vtype](agg) + self.linear_self[vtype](ft_dst[vtype])
            )

        return feats


class KnowledgeGraphEmbeddingLayer(nn.Module):
    def __init__(
        self,
        knowledge_graph,
        node_feats,
        num_layers,
        residual=True,
        batch_norm=True,
        feat_drop=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroAttentionAggregationLayer(
                knowledge_graph,
                node_feats,
                batch_norm=batch_norm,
                feat_drop=feat_drop,
            ) for _ in range(num_layers)
        ])
        self.residual = residual

    def forward(self, graphs, feats):
        for layer, g in zip(self.layers, graphs):
            out_feats = layer(g, feats)
            if self.residual:
                feats = {
                    ntype: out_feats[ntype] + feat[:len(out_feats[ntype])]
                    for ntype, feat in feats.items()
                }
            else:
                feats = out_feats
        return feats


class SEFrame(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        knowledge_graph,
        num_layers,
        batch_norm=True,
        feat_drop=0.0,
        **kwargs,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, max_norm=1)
        self.user_indices = nn.Parameter(
            th.arange(num_users, dtype=th.long), requires_grad=False
        )
        self.item_embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.item_indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.knowledge_graph = knowledge_graph
        self.KGE_layer = KnowledgeGraphEmbeddingLayer(
            knowledge_graph,
            embedding_dim,
            num_layers,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
        )

    def precompute_KG_embeddings(self):
        self.eval()
        kg_device = self.knowledge_graph.device
        ft_device = self.user_indices.device
        if kg_device != ft_device:
            logging.debug(f'Copying knowledge graph from {kg_device} to {ft_device}')
            self.knowledge_graph = self.knowledge_graph.to(ft_device)
        with th.no_grad():
            graphs = [self.knowledge_graph] * len(self.KGE_layer.layers)
            feats = {
                'user': self.user_embedding(self.user_indices),
                'item': self.item_embedding(self.item_indices),
            }
            self.KG_embeddings = self.KGE_layer(graphs, feats)

    def forward(self, inputs):
        if inputs is None:
            return self.KG_embeddings
        else:
            graphs, used_nodes = inputs
            feats = {
                'user': self.user_embedding(used_nodes['user']),
                'item': self.item_embedding(used_nodes['item']),
            }
            return self.KGE_layer(graphs, feats)
