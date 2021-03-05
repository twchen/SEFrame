import torch as th
from torch import nn


class STAMPLayer(nn.Module):
    def __init__(self, embedding_dim, feat_drop=0.0):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_a = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.fc_t = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.attn_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_t = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.attn_s = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_e = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, emb_seqs, lens):
        # emb_seqs: (batch_size, max_len, d)
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)
        batch_size, max_len, _ = emb_seqs.size()
        mask = th.arange(
            max_len, device=lens.device
        ).unsqueeze(0).expand(batch_size, max_len) >= lens.unsqueeze(-1)
        emb_seqs = th.masked_fill(emb_seqs, mask.unsqueeze(-1), 0)
        # emb_seqs = th.where(mask.unsqueeze(-1), emb_seqs, th.zeros_like(emb_seqs))

        ms = emb_seqs.sum(dim=1) / lens.unsqueeze(-1)  # (batch_size, d)

        xt = emb_seqs[th.arange(batch_size), lens - 1]  # (batch_size, d)
        ei = self.attn_i(emb_seqs)  # (batch_size, max_len, d)
        et = self.attn_t(xt).unsqueeze(1)  # (batch_size, 1, d)
        es = self.attn_s(ms).unsqueeze(1)  # (batch_size, 1, d)
        e = self.attn_e(th.sigmoid(ei + et + es)).squeeze(-1)  # (batch_size, max_len)
        alpha = th.masked_fill(e, mask, 0)
        alpha = alpha.unsqueeze(-1)  # (batch_size, max_len, 1)
        ma = th.sum(alpha * emb_seqs, dim=1)  # (batch_size, d)

        ha = self.fc_a(ma)
        ht = self.fc_t(xt)

        sr = ha * ht
        return sr
