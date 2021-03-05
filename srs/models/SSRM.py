import torch as th
from torch import nn

from srs.layers.ssrm import SSRMLayer
from srs.utils.data.collate import collate_fn_for_rnn_cnn
from srs.utils.data.load import BatchSampler
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory


class SSRM(nn.Module):
    def __init__(
        self, num_users, num_items, embedding_dim, w=0.5, feat_drop=0.0, **kwargs
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, max_norm=1)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.layer = SSRMLayer(embedding_dim, w, feat_drop=feat_drop)

    def forward(self, uids, padded_seqs, lens):
        feat_u = self.user_embedding(uids)
        emb_seqs = self.item_embedding(padded_seqs)
        sr = self.layer(emb_seqs, lens, feat_u)
        logits = sr @ self.item_embedding(self.indices).t()
        return logits


config = Dict({
    'Model': SSRM,
    'collate_fn': collate_fn_for_rnn_cnn,
    'BatchSampler': BatchSampler,
    'prepare_batch_factory': prepare_batch_factory,
})
