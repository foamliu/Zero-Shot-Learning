import os

import torch

from config import attributes_per_class
from config import device


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def KNN(mat, k):
    mat = mat.float()
    mat_square = torch.mm(mat, mat.t())
    diag = torch.diagonal(mat_square)
    # print(diag)
    # print(diag.size())
    val, index = diag.topk(k, largest=False, sorted=True)
    return val, index


def batched_KNN(query, k):
    attributes = attributes_per_class
    batch_size = query.size()[0]
    val_list = torch.zeros(batch_size, dtype=torch.float, device=device)
    index_list = torch.zeros(batch_size, dtype=torch.int, device=device)
    for i in range(batch_size):
        q = query[i].to(device)
        attributes = attributes.to(device)
        diff = q - attributes
        diff = torch.tensor(diff)
        diff = diff.to(device)
        val, index = KNN(diff, k)
        val_list[i] = val
        index_list[i] = index
    return val_list, index_list
