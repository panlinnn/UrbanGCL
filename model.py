import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm
from utils import hard_sample, NTXentLossSC, NTXentLossTC


class GCN(nn.Module):
    def __init__(self, c_in, c_out, K):  # in:64   out:64
        super(GCN, self).__init__()
        self.K = K
        c_in_new = K * c_in  # 3 * 64
        self.conv = Conv2d(c_in_new, c_out, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x, adj):
        bs, feat, node, frame = x.shape  # [64, 2, 128, 19]

        # cheb:  l = 2 * G * (l-1) - (l-2)
        Ls = []
        L0, L1 = torch.eye(node).cuda(), adj
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,node, node]
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(bs, -1, node, frame)
        out = self.conv(x)
        return out


class STBlock(nn.Module):
    def __init__(self, args, c_in, c_out):
        super(STBlock, self).__init__()
        self.gcn = GCN(c_in, c_out, args.K)
        self.tc_conv = Conv2d(c_out, c_out, kernel_size=(1, args.Kt),
                              padding=(0, 1), stride=(1, 1), bias=True)
        self.conv = Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x, adj_mx):
        # gcn
        x1 = self.gcn(x, adj_mx)  # [64, 64, 128, 19]
        # tc
        x2 = self.tc_conv(x1)  # [64, 64, 128, 19]
        # gate unit
        xx = self.conv(x)  # x:[64, 2, 128, 19] -> xx:[64, 64, 128, 19]
        filterr, gate = x2, x2
        x3 = (filterr + xx) * torch.tanh(gate)
        return x3  # [64, 64, 128, 19]


class UrbanGCL(nn.Module):
    def __init__(self, args, adj_mx, num_nodes):
        super(UrbanGCL, self).__init__()
        self.adj_mx = adj_mx
        self.num_nodes = num_nodes
        self.xi = args.xi
        self.sc = args.sc
        self.tc = args.tc
        self.device = args.device
        self.d_input = args.d_input
        self.d_model = args.d_model

        self.block1 = STBlock(args, self.d_input, self.d_model)  # 2 -> 64
        self.block2 = STBlock(args, self.d_model, self.d_model)  # 64 -> 64
        self.block3 = STBlock(args, self.d_model, self.d_model)  # 64 -> 64

        self.conv1 = Conv2d(2 * args.d_model, 1, kernel_size=(1, args.input_length - 1),
                            padding=(0, 0), stride=(1, 1), bias=True)

        self.projection_head = nn.Sequential(
            Conv2d(args.d_model, args.d_model // 4, kernel_size=(1, 1), padding=(0, 0),
                   stride=(1, 1), bias=True),
            nn.BatchNorm2d(args.d_model // 4),
            nn.ReLU(inplace=True),
            Conv2d(args.d_model // 4, args.d_model, kernel_size=(1, 1), padding=(0, 0),
                   stride=(1, 1), bias=True),
        )

        self.s_con = NTXentLossSC(args)
        self.t_con = NTXentLossTC(args)

        self.ao = Parameter(torch.zeros(self.num_nodes, self.num_nodes), requires_grad=True)
        nn.init.uniform_(self.ao, a=0, b=0.0001)

    def forward(self, input_val):
        # masking
        x = input_val  # input_val:[64, 2, 128, 19]
        bs, feat, node, frame = x.shape
        rand = torch.rand(bs, node, frame).to(self.device)
        x[:, 1, :, :] = x[:, 1, :, :] * (rand >= self.xi)

        # local
        ident = (self.adj_mx != 0).float()  # mask:原来0位置取0，非零位置取1
        a_l = self.ao * ident
        d_l = 1 / (torch.sum(a_l, -1) + 0.0001)
        D_l = torch.diag_embed(d_l)
        a_l = torch.matmul(D_l, a_l)
        a_l = F.dropout(a_l, 0.3, self.training)

        h_l = self.block1(x, a_l)  # [64, 64, 128, 19]
        h_l = self.block2(h_l, a_l)  # [64, 64, 128, 19]
        h_l = self.block3(h_l, a_l)  # [64, 64, 128, 19]

        # global
        a_g = self.ao + self.adj_mx
        d_g = 1 / (torch.sum(a_g, -1) + 0.0001)
        D_g = torch.diag_embed(d_g)
        a_g = torch.matmul(D_g, a_g)
        a_g = F.dropout(a_g, 0.3, self.training)

        h_g = self.block1(x, a_g)
        h_g = self.block2(h_g, a_g)
        h_g = self.block3(h_g, a_g)  # [64, 64, 128, 19]

        sc_loss, tc_loss = 0, 0

        if self.training:
            p_l = (self.projection_head(h_l))  # [64, 64, 128, 19]
            p_g = (self.projection_head(h_g))  # [64, 64, 128, 19]

            if self.sc:
                sc_loss = self.s_con(p_l, p_g)

            if self.tc:
                p = torch.cat((h_l, h_g), 1)
                anchor, pos, neg = hard_sample(p)
                tc_loss = self.t_con(anchor, pos, neg)

        # prediction head
        x = torch.cat((h_l, h_g), 1)  # [64, 128, 128, 19]
        x = self.conv1(x)  # [64, 1, 128, 2]

        return x, sc_loss, tc_loss
