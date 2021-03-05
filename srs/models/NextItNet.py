import torch as th
from torch import nn

from srs.layers.nextitnet import NextItNetLayer
from srs.utils.data.collate import collate_fn_for_rnn_cnn
from srs.utils.data.load import BatchSampler
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory


class NextItNet(nn.Module):
    def __init__(
        self,
        num_items,
        embedding_dim,
        dilations=None,
        one_masked=False,
        kernel_size=3,
        feat_drop=0.0,
        **kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.layer = NextItNetLayer(
            embedding_dim, dilations, one_masked, kernel_size, feat_drop=feat_drop
        )
        self.fc_sr = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, uids, padded_seqs, lens):
        # padded_seqs: (B, L)
        emb_seqs = self.embedding(padded_seqs)  # (B, L, C)
        sr = self.layer(emb_seqs, lens)
        logits = self.fc_sr(sr) @ self.embedding(self.indices).t()
        return logits


config = Dict({
    'Model': NextItNet,
    'collate_fn': collate_fn_for_rnn_cnn,
    'BatchSampler': BatchSampler,
    'prepare_batch_factory': prepare_batch_factory,
})
