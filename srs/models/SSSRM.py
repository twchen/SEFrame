import torch as th
from torch import nn

from srs.layers.seframe import SEFrame
from srs.layers.ssrm import SSRMLayer
from srs.utils.data.collate import CollateFnRNNCNN
from srs.utils.data.load import BatchSampler
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory_recursive


class SSSRM(SEFrame):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        knowledge_graph,
        num_layers,
        w=0.5,
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
        self.PSE_layer = SSRMLayer(embedding_dim, w, feat_drop=feat_drop)
        self.fc_sr = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

    def forward(self, inputs, extra_inputs=None):
        KG_embeddings = super().forward(extra_inputs)

        uids, padded_seqs, lens = inputs
        emb_seqs = KG_embeddings['item'][padded_seqs]
        feat_u = KG_embeddings['user'][uids]
        sr = self.PSE_layer(emb_seqs, lens, feat_u)
        sr = th.cat([sr, feat_u], dim=1)
        logits = self.fc_sr(sr) @ self.item_embedding(self.item_indices).t()

        return logits


config = Dict({
    'Model': SSSRM,
    'CollateFn': CollateFnRNNCNN,
    'BatchSampler': BatchSampler,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})
