import os

import torch


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def parse_attributes(attr_str):
    tokens = attr_str.split(' ')
    attr_list = []
    for i in range(1, len(tokens) - 1):
        attr_list.append(float(tokens[i]))

    return attr_list


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
