import torch as th
from torch import nn


class NextItNetLayer(nn.Module):
    def __init__(self, channels, dilations, one_masked, kernel_size, feat_drop=0.0):
        super().__init__()
        if one_masked:
            ResBlock = ResBlockOneMasked
            if dilations is None:
                dilations = [1, 2, 4]
        else:
            ResBlock = ResBlockTwoMasked
            if dilations is None:
                dilations = [1, 4]
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, kernel_size, dilation) for dilation in dilations
        ])

    def forward(self, emb_seqs, lens):
        # emb_seqs: (B, L, C)
        batch_size, max_len, _ = emb_seqs.size()
        mask = th.arange(
            max_len, device=lens.device
        ).unsqueeze(0).expand(batch_size, max_len) >= lens.unsqueeze(-1)
        emb_seqs = th.masked_fill(emb_seqs, mask.unsqueeze(-1), 0)
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)

        x = th.transpose(emb_seqs, 1, 2)  # (B, C, L)
        for res_block in self.res_blocks:
            x = res_block(x)
        batch_idx = th.arange(len(lens))
        last_idx = lens - 1
        sr = x[batch_idx, :, last_idx]  # (B, C)
        return sr


class MaskedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.repr_str = (
            f'{self.__class__.__name__}(in_channels={in_channels}, '
            f'out_channels={out_channels}, kernel_size={kernel_size}, dilation={dilation})'
        )
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.padding = (kernel_size - 1) * dilation

    def forward(self, x):
        # x: (B, C, L)
        x = th.nn.functional.pad(x, [self.padding, 0])  # (B, C, L + self.padding)
        x = self.conv(x)
        return x

    def __repr__(self):
        return self.repr_str


class LayerNorm(nn.Module):
    def __init__(self, channels, epsilon=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(th.ones([1, channels, 1], dtype=th.float32))
        self.beta = nn.Parameter(th.zeros([1, channels, 1], dtype=th.float32))
        self.epsilon = epsilon

    def forward(self, x):
        # x: (B, C, L)
        var, mean = th.var_mean(x, dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / th.sqrt(var + self.epsilon)
        return x * self.gamma + self.beta


class ResBlockOneMasked(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        mid_channels = channels // 2
        self.layer_norm1 = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, mid_channels, kernel_size=1)
        self.layer_norm2 = LayerNorm(mid_channels)
        self.conv2 = MaskedConv1d(
            mid_channels, mid_channels, kernel_size=kernel_size, dilation=dilation
        )
        self.layer_norm3 = LayerNorm(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        y = x
        y = th.relu(self.layer_norm1(y))
        y = self.conv1(y)
        y = th.relu(self.layer_norm2(y))
        y = self.conv2(y)
        y = th.relu(self.layer_norm3(y))
        y = self.conv3(y)
        return y + x


class ResBlockTwoMasked(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = MaskedConv1d(channels, channels, kernel_size, dilation)
        self.layer_norm1 = LayerNorm(channels)
        self.conv2 = MaskedConv1d(channels, channels, kernel_size, 2 * dilation)
        self.layer_norm2 = LayerNorm(channels)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = th.relu(self.layer_norm1(y))
        y = self.conv2(y)
        y = th.relu(self.layer_norm2(y))
        return y + x
