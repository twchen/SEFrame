import torch as th
from torch import nn

from srs.layers.seframe import SEFrame
from srs.layers.stamp import STAMPLayer
from srs.utils.data.collate import CollateFnRNNCNN
from srs.utils.data.load import BatchSampler
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory_recursive


class SSTAMP(SEFrame):
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
        self.PSE_layer = STAMPLayer(embedding_dim, feat_drop=feat_drop)
        self.fc_sr = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

    def forward(self, inputs, extra_inputs=None):
        KG_embeddings = super().forward(extra_inputs)

        uids, padded_seqs, lens = inputs
        emb_seqs = KG_embeddings['item'][padded_seqs]
        feat_u = KG_embeddings['user'][uids]
        feat = self.fc_i(emb_seqs) + self.fc_u(feat_u).unsqueeze(1)
        feat_i = self.PSE_layer(feat, lens)
        sr = th.cat([feat_i, feat_u], dim=1)
        logits = self.fc_sr(sr) @ self.item_embedding(self.item_indices).t()
        return logits


config = Dict({
    'Model': SSTAMP,
    'CollateFn': CollateFnRNNCNN,
    'BatchSampler': BatchSampler,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})
