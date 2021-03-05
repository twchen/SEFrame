import torch as th
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NARMLayer(nn.Module):
    def __init__(self, input_dim, feat_drop=0.0):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.gru = nn.GRU(input_dim, input_dim)
        self.attn_i = nn.Linear(input_dim, input_dim, bias=False)
        self.attn_t = nn.Linear(input_dim, input_dim, bias=False)
        self.attn_e = nn.Linear(input_dim, 1, bias=False)

    def forward(self, emb_seqs, lens):
        batch_size, max_len, _ = emb_seqs.size()
        mask = th.arange(
            max_len, device=lens.device
        ).unsqueeze(0).expand(batch_size, max_len) >= lens.unsqueeze(-1)

        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)
        packed_seqs = pack_padded_sequence(emb_seqs, lens.cpu(), batch_first=True)
        out, ht = self.gru(packed_seqs)
        out, _ = pad_packed_sequence(out, batch_first=True)  # (batch_size, max_len, d)
        ht = ht.transpose(0, 1)  # (batch_size, 1, d)

        ei = self.attn_i(out)
        et = self.attn_t(ht)
        e = self.attn_e(th.sigmoid(ei + et))  # (batch_size, max_len, 1)
        e = e.squeeze(-1)  # (batch_size, max_len)
        alpha = th.masked_fill(e, mask, 0)

        ct_g = ht.squeeze(1)  # (batch_size, d)
        ct_l = th.sum(out * alpha.unsqueeze(-1), dim=1)  # (batch_size, d)
        sr = th.cat([ct_g, ct_l], dim=1)
        return sr
