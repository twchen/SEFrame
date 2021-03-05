import torch as th
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SSRMLayer(nn.Module):
    def __init__(self, embedding_dim, w=0.5, feat_drop=0.0):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.B = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.w = w

    def forward(self, emb_seqs, lens, feat_u):
        """
        emb_seqs: (batch_size, max_len, d)
        """
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)
            feat_u = self.feat_drop(feat_u)

        batch_size, max_len, _ = emb_seqs.size()
        mask = th.arange(
            max_len, device=lens.device
        ).unsqueeze(0).expand(batch_size, max_len) >= lens.unsqueeze(-1)

        packed_seqs = pack_padded_sequence(emb_seqs, lens.cpu(), batch_first=True)

        out, hn = self.gru(packed_seqs)
        out, _ = pad_packed_sequence(
            out, batch_first=True
        )  # out: (batch_size, max_len, d)
        h_t = hn.squeeze(0)  # (batch_size, d)

        alpha = (emb_seqs * feat_u.unsqueeze(1)).sum(dim=-1)  # (batch_size, max_len)
        alpha = th.masked_fill(alpha, mask, float('-inf'))
        alpha = alpha.softmax(dim=1).unsqueeze(-1)  # (batch_size, max_len, 1)
        h_sum = (alpha * out).sum(dim=1)  # (batch_size, d)
        ct = th.cat([h_sum, h_t], dim=1)

        sr = self.w * feat_u + (1 - self.w) * self.B(ct)
        return sr
