import random
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp


def init_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs


def sym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def load_graph(adj_file, device='cpu'):
    graph = np.load(adj_file)['adj_mx']
    adj = sym_adj(graph)
    return adj


def get_model_params(model_list):
    model_parameters = []
    for m in model_list:
        if m != None:
            model_parameters += list(m.parameters())
    return model_parameters


def hard_sample(p):
    # p (64, 64, 128, 19)
    anchor, pos, neg = [], [], []
    length = p.shape[3]  # 19
    for i in range(length):
        minn, maxn = float('inf'), -float('inf')
        kmin, kmax = i, i
        for j in range(length):
            if i == j:
                continue
            dist = torch.sqrt(torch.sum(torch.square(p[:, :, :, i] - p[:, :, :, j])))
            if dist > maxn:
                maxn = dist
                kmax = i
            if dist < minn:
                minn = dist
                kmin = j
        anchor.append(p[:, :, :, i])
        pos.append(p[:, :, :, kmin])
        neg.append(p[:, :, :, kmax])

    anchor = torch.stack(anchor)
    pos = torch.stack(pos)
    neg = torch.stack(neg)

    return anchor, pos, neg


class NTXentLossSC(torch.nn.Module):
    def __init__(self, args):
        super(NTXentLossSC, self).__init__()
        self.batch_size = args.batch_size  # 64
        self.temperature = args.temp  # 0.05
        self.device = args.device

    def forward(self, zis, zjs):
        shape = zis.shape  # [64, 64, 128, 19]
        zis = zis.reshape(shape[0], -1)  # [64, 65280]
        zjs = zjs.reshape(shape[0], -1)

        zis1 = F.normalize(zis, p=2, dim=-1)  # [64, 65280]
        zjs1 = F.normalize(zjs, p=2, dim=-1)

        similarity_matrix = torch.matmul(zis1, zjs1.permute(1, 0))  # [64, 64]

        shape = similarity_matrix.shape
        # filter out the scores from the positive samples  diagonal element: 64
        l_pos = torch.diag(similarity_matrix)
        positives = l_pos
        positives = positives / self.temperature

        diag = np.eye(self.batch_size)
        mask = torch.from_numpy((diag))
        # diagonal:false; other:ture
        mask = (1 - mask).type(torch.bool).cuda()
        # (64, 63)  delete diagonal
        negatives = similarity_matrix[mask].view(self.batch_size, self.batch_size - 1)

        negatives = negatives / self.temperature

        loss = -torch.log((torch.exp(positives)) /
                          (torch.exp(positives) + torch.sum(torch.exp(negatives), -1, True)))
        return loss.sum() / (shape[0])


class NTXentLossTC(torch.nn.Module):
    def __init__(self, args):
        super(NTXentLossTC, self).__init__()
        self.timestep = args.input_length  # 19
        self.temperature = args.temp  # 0.05
        self.device = args.device

    def forward(self, anchor, zis, zjs):
        shape = zis.shape  # [19, 64, 64, 128]
        anchor = anchor.reshape(shape[0], -1)
        zis = zis.reshape(shape[0], -1)  # [19, ...]
        zjs = zjs.reshape(shape[0], -1)

        anchor1 = F.normalize(anchor, p=2, dim=-1)
        zis1 = F.normalize(zis, p=2, dim=-1)  # [19, ...]
        zjs1 = F.normalize(zjs, p=2, dim=-1)

        sim_pos = torch.matmul(anchor1, zis1.permute(1, 0))  # [19, 19]
        sim_neg = torch.matmul(anchor1, zjs1.permute(1, 0))  # [19, 19]

        positives = sim_pos / self.temperature
        negatives = sim_neg / self.temperature

        loss = -torch.log((torch.exp(positives)) / (torch.exp(negatives)))

        return loss.mean()
