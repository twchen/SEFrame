import torch as th
from torch import nn

from srs.layers.narm import NARMLayer
from srs.utils.data.collate import collate_fn_for_rnn_cnn
from srs.utils.data.load import BatchSampler
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory


class NARM(nn.Module):
    def __init__(self, num_items, embedding_dim, feat_drop=0.0, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.narm_layer = NARMLayer(embedding_dim, feat_drop=feat_drop)
        self.fc_sr = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

    def forward(self, uids, padded_seqs, lens):
        emb_seqs = self.embedding(padded_seqs)
        sr = self.narm_layer(emb_seqs, lens)
        sr = self.fc_sr(sr)
        logits = sr @ self.embedding(self.indices).t()
        return logits


config = Dict(
    {
        'Model': NARM,
        'collate_fn': collate_fn_for_rnn_cnn,
        'BatchSampler': BatchSampler,
        'prepare_batch_factory': prepare_batch_factory,
    }
)
