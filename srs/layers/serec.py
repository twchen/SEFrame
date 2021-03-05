import torch as th
from torch import nn

import dgl
import dgl.ops as F


class UpdateCell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.x2i = nn.Linear(input_dim, 2 * output_dim, bias=True)
        self.h2h = nn.Linear(output_dim, 2 * output_dim, bias=False)

    def forward(self, x, hidden):
        i_i, i_n = self.x2i(x).chunk(2, 1)
        h_i, h_n = self.h2h(hidden).chunk(2, 1)
        input_gate = th.sigmoid(i_i + h_i)
        new_gate = th.tanh(i_n + h_n)
        return new_gate + input_gate * (hidden - new_gate)


class PWGGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_steps=1,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_i2h = nn.Linear(
            input_dim, hidden_dim, bias=False
        ) if input_dim != hidden_dim else None
        self.fc_in = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.upd_cell = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.upd_cell = UpdateCell(2 * hidden_dim, hidden_dim)
        self.fc_h2o = nn.Linear(
            hidden_dim, output_dim, bias=False
        ) if hidden_dim != output_dim else None
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.activation = activation

    def propagate(self, g, rg, feat):
        if g.number_of_edges() > 0:
            feat_in = self.fc_in(feat)
            feat_out = self.fc_out(feat)
            a_in = F.u_mul_e_sum(g, feat_in, g.edata['iw'])
            a_out = F.u_mul_e_sum(rg, feat_out, rg.edata['ow'])
            # a: (num_nodes, 2 * hidden_dim)
            a = th.cat((a_in, a_out), dim=1)
        else:
            num_nodes = g.number_of_nodes()
            a = feat.new_zeros((num_nodes, 2 * self.hidden_dim))
        hn = self.upd_cell(a, feat)
        return hn

    def forward(self, g, rg, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        if self.feat_drop is not None:
            feat = self.feat_drop(feat)
        if self.fc_i2h is not None:
            feat = self.fc_i2h(feat)
        for _ in range(self.num_steps):
            feat = self.propagate(g, rg, feat)
        if self.fc_h2o is not None:
            feat = self.fc_h2o(feat)
        if self.activation is not None:
            feat = self.activation(feat)
        return feat


class PAttentionReadout(nn.Module):
    def __init__(self, embedding_dim, batch_norm=False, feat_drop=0.0, activation=None):
        super().__init__()
        if batch_norm:
            self.batch_norm = nn.ModuleDict({
                'user': nn.BatchNorm1d(embedding_dim),
                'item': nn.BatchNorm1d(embedding_dim)
            })
        else:
            self.batch_norm = None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_user = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.fc_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_e = nn.Linear(embedding_dim, 1, bias=False)
        self.activation = activation

    def forward(self, g, feat_i, feat_u, last_nodes):
        if self.batch_norm is not None:
            feat_i = self.batch_norm['item'](feat_i)
            feat_u = self.batch_norm['user'](feat_u)
        if self.feat_drop is not None:
            feat_i = self.feat_drop(feat_i)
            feat_u = self.feat_drop(feat_u)
        feat_val = feat_i
        feat_key = self.fc_key(feat_i)
        feat_u = self.fc_user(feat_u)
        feat_last = self.fc_last(feat_i[last_nodes])
        feat_qry = dgl.broadcast_nodes(g, feat_u + feat_last)
        e = self.fc_e(th.sigmoid(feat_qry + feat_key))  # (num_nodes, 1)
        e = e + g.ndata['cnt'].log().view_as(e)
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        rst = F.segment.segment_reduce(g.batch_num_nodes(), alpha * feat_val, 'sum')
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class SERecLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_steps=1,
        batch_norm=True,
        feat_drop=0.0,
        relu=False,
    ):
        super().__init__()
        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pwggnn = PWGGNN(
            embedding_dim,
            embedding_dim,
            embedding_dim,
            num_steps=num_steps,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.ReLU() if relu else nn.PReLU(embedding_dim),
        )
        self.readout = PAttentionReadout(
            embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.ReLU() if relu else nn.PReLU(embedding_dim),
        )

    def forward(self, g, feat, feat_u):
        rg = dgl.reverse(g, False, False)
        if g.number_of_edges() > 0:
            edge_weight = g.edata['w']
            in_deg = F.copy_e_sum(g, edge_weight)
            g.edata['iw'] = F.e_div_v(g, edge_weight, in_deg)
            out_deg = F.copy_e_sum(rg, edge_weight)
            rg.edata['ow'] = F.e_div_v(rg, edge_weight, out_deg)

        feat = self.pwggnn(g, rg, feat)
        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        ct_l = feat[last_nodes]
        ct_g = self.readout(g, feat, feat_u, last_nodes)
        sr = th.cat((ct_l, ct_g), dim=1)
        return sr
