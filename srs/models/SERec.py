import torch as th
from torch import nn
import dgl

from srs.layers.seframe import SEFrame
from srs.layers.serec import SERecLayer
from srs.utils.data.collate import CollateFnGNN
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory_recursive
from srs.utils.data.transform import seq_to_weighted_graph


class SERec(SEFrame):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        knowledge_graph,
        num_layers,
        relu=False,
        batch_norm=True,
        feat_drop=0.0,
        **kwargs
    ):
        super().__init__(
            num_users,
            num_items,
            embedding_dim,
            knowledge_graph,
            num_layers,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
        )
        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.PSE_layer = SERecLayer(
            embedding_dim,
            num_steps=1,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            relu=relu,
        )
        input_dim = 3 * embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

    def forward(self, inputs, extra_inputs=None):
        KG_embeddings = super().forward(extra_inputs)

        uid, g = inputs
        iid = g.ndata['iid']  # (num_nodes,)
        feat_i = KG_embeddings['item'][iid]
        feat_u = KG_embeddings['user'][uid]
        feat = self.fc_i(feat_i) + dgl.broadcast_nodes(g, self.fc_u(feat_u))
        feat_i = self.PSE_layer(g, feat, feat_u)
        sr = th.cat([feat_i, feat_u], dim=1)
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        logits = self.fc_sr(sr) @ self.item_embedding(self.item_indices).t()
        return logits


seq_to_graph_fns = [seq_to_weighted_graph]

config = Dict({
    'Model': SERec,
    'seq_to_graph_fns': seq_to_graph_fns,
    'CollateFn': CollateFnGNN,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})
