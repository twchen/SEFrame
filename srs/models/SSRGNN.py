import torch as th
from torch import nn
import dgl

from srs.layers.seframe import SEFrame
from srs.layers.srgnn import SRGNNLayer
from srs.utils.data.collate import CollateFnGNN
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory_recursive
from srs.utils.data.transform import seq_to_unweighted_graph


class SSRGNN(SEFrame):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        knowledge_graph,
        num_layers,
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
        self.PSE_layer = SRGNNLayer(embedding_dim, feat_drop=feat_drop)
        self.fc_sr = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)

    def forward(self, inputs, extra_inputs=None):
        KG_embeddings = super().forward(extra_inputs)

        uid, g = inputs
        iid = g.ndata['iid']  # (num_nodes,)
        feat_i = KG_embeddings['item'][iid]
        feat_u = KG_embeddings['user'][uid]
        feat = self.fc_i(feat_i) + dgl.broadcast_nodes(g, self.fc_u(feat_u))
        feat_i = self.PSE_layer(g, feat)
        sr = th.cat([feat_i, feat_u], dim=1)
        logits = self.fc_sr(sr) @ self.item_embedding(self.item_indices).t()
        return logits


seq_to_graph_fns = [seq_to_unweighted_graph]

config = Dict({
    'Model': SSRGNN,
    'seq_to_graph_fns': seq_to_graph_fns,
    'CollateFn': CollateFnGNN,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})
