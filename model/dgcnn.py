import torch
import torch.nn as nn
import torch.nn.functional as F
from util.common_util import index_points


def max_pool_batch(x, offset):
    B = offset.shape[0]
    res = [x[0:offset[0], :].max(dim=0, keepdim=True)[0]]
    for i in range(1, B):
        res.append(x[offset[i-1]:offset[i], :].max(dim=0, keepdim=True)[0])
    return torch.cat(res, dim=0)


def repeat_batch(x, offset):
    B = offset.shape[0]
    res = [x[0, :].repeat(offset[0], 1)]
    for i in range(1, B):
        res.append(x[i, :].repeat(offset[i]-offset[i-1], 1))
    return torch.cat(res, dim=0)


def knn(x, k, index, metric='Euclidean'):  # N x C
    if metric == 'Euclidean':
        pairwise_similarity = torch.matmul(x, x.transpose(0, 1))
        pairwise_similarity -= torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_similarity -= torch.sum(x ** 2, dim=1, keepdim=True).transpose(0, 1)
        # pairwise_similarity = -norm2 + 2 * inner - norm2.transpose(0, 1)  # negative_pairwise_distance
    elif metric == 'Cosine':
        x_normed = F.normalize(x, p=2, dim=1)
        pairwise_similarity = torch.matmul(x_normed, x_normed.transpose(0, 1))
    else:
        exit('Unknown distance metric!')
        return None

    val, idx = pairwise_similarity.topk(k=k, dim=-1)  # (num_points, k)
    return val, index[idx]


def knn_batch(x, offset, k, metric='Euclidean'):
    B = offset.shape[0]
    res = [knn(x[0:offset[0], :], k, torch.arange(0, offset[0]), metric)]
    for i in range(1, B):
        res.append(knn(x[offset[i-1]:offset[i], :], k, torch.arange(offset[i-1], offset[i]), metric))
    val, idx = list(zip(*res))

    return torch.cat(val), torch.cat(idx).cuda()


class EdgeConv(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(EdgeConv, self).__init__()
        self.CBLs = nn.ModuleList()
        in_channel = input_size * 2
        for out_channel in hidden_sizes:
            self.CBLs.append(nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
                                           nn.BatchNorm1d(out_channel),
                                           nn.LeakyReLU(negative_slope=0.2)))
            in_channel = out_channel

    def forward(self, feats, neighbor_idx):  # N x C, N x 3, B, N, N x K
        N = neighbor_idx.size(0)
        K = neighbor_idx.size(1)
        x = index_points(feats, neighbor_idx)  # N x K x C
        x = torch.cat((feats[:, None].expand(-1, K, -1), x-feats[:, None]), dim=-1).transpose(1, 2).contiguous()  # N x 2C x K
        for i, CBL in enumerate(self.CBLs):
            x = CBL(x)  # N x D x K
        x = x.max(dim=-1, keepdim=False)[0]
        return x  # N x D


class FixedGCNN(nn.Module):
    def __init__(self, k):
        super(FixedGCNN, self).__init__()
        self.k = k
        self.conv1 = EdgeConv(6, [64, 64])
        self.conv2 = EdgeConv(64, [64, 64])
        self.conv3 = EdgeConv(64, [64, ])

        self.conv6 = nn.Sequential(nn.Linear(192, 1024, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Linear(1216, 512, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Linear(256, 13, bias=False)

    def forward(self, feats, xyz, offset, batch, neighbor_idx):
        # _, idx = knn_batch(xyz, offset, self.k, metric='Euclidean')
        x1 = self.conv1(feats, neighbor_idx)
        x2 = self.conv2(x1, neighbor_idx)
        x3 = self.conv3(x2, neighbor_idx)
        x = torch.cat((x1, x2, x3), dim=1)  # N x 64*3

        x = self.conv6(x)  # N x 1024
        x = max_pool_batch(x, offset)  # B x 1024

        x = repeat_batch(x, offset)  # N x 1024
        x = torch.cat((x, x1, x2, x3), dim=1)  # N x (1024+64*3)

        x = self.conv7(x)  # N x 512
        x = self.conv8(x)  # N x 256
        x = self.dp1(x)
        x = self.conv9(x)  # N x 13

        return x
