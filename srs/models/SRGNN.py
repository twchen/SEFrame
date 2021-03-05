import torch as th
from torch import nn

from srs.layers.srgnn import SRGNNLayer
from srs.utils.data.collate import collate_fn_for_gnn_factory
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory
from srs.utils.data.transform import seq_to_unweighted_graph


class SRGNN(nn.Module):
    def __init__(self, num_items, embedding_dim, feat_drop=0.0, **kwargs):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.item_indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.layer = SRGNNLayer(embedding_dim, feat_drop=feat_drop)
        self.fc_sr = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

    def forward(self, uid, g):
        iid = g.ndata['iid']  # (num_nodes,)
        feat = self.item_embedding(iid)
        sr = self.layer(g, feat)
        logits = self.fc_sr(sr) @ self.item_embedding(self.item_indices).t()
        return logits


seq_to_graph_fns = [seq_to_unweighted_graph]
collate_fn = collate_fn_for_gnn_factory(*seq_to_graph_fns)

config = Dict({
    'Model': SRGNN,
    'seq_to_graph_fns': seq_to_graph_fns,
    'collate_fn': collate_fn,
    'prepare_batch_factory': prepare_batch_factory,
})
