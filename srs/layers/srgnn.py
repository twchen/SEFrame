import torch as th
from torch import nn
import dgl
import dgl.ops as F


class GGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        num_steps=1,
        batch_norm=False,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_in = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gru_cell = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.activation = activation

    def propagate(self, g, rg, feat):
        if g.number_of_edges() > 0:
            feat_in = self.fc_in(feat)
            feat_out = self.fc_out(feat)
            a_in = F.copy_u_mean(g, feat_in)
            a_out = F.copy_u_mean(rg, feat_out)
            # a: (num_nodes, 2 * hidden_dim)
            a = th.cat((a_in, a_out), dim=1)
        else:
            num_nodes = g.number_of_nodes()
            a = feat.new_zeros((num_nodes, 2 * self.hidden_dim))
        hn = self.gru_cell(a, feat)
        return hn

    def forward(self, g, rg, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        if self.feat_drop is not None:
            feat = self.feat_drop(feat)
        for _ in range(self.num_steps):
            feat = self.propagate(g, rg, feat)
        if self.activation is not None:
            feat = self.activation(feat)
        return feat


class AttentionReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        batch_norm=False,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim else None
        )
        self.activation = activation

    def forward(self, g, feat, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        if self.feat_drop is not None:
            feat = self.feat_drop(feat)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = dgl.broadcast_nodes(g, feat_v)
        e = self.fc_e(th.sigmoid(feat_u + feat_v))  # (num_nodes, 1)
        alpha = e * g.ndata['cnt'].view_as(e)
        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat * alpha, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class SRGNNLayer(nn.Module):
    def __init__(self, embedding_dim, feat_drop=0.0):
        super().__init__()
        self.ggnn = GGNN(embedding_dim, num_steps=1, feat_drop=feat_drop, activation=None)
        self.readout = AttentionReadout(embedding_dim, embedding_dim, feat_drop=feat_drop)

    def forward(self, g, feat):
        rg = dgl.reverse(g, False, False)
        feat = self.ggnn(g, rg, feat)
        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        ct_l = feat[last_nodes]
        ct_g = self.readout(g, feat, last_nodes)
        sr = th.cat([ct_g, ct_l], dim=1)
        return sr
