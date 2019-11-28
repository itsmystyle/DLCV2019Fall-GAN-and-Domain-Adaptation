import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


def MSELoss(input, target):
    diffs = input - target

    return torch.sum(diffs.pow(2)) / torch.numel(diffs)


def SI_MSELoss(input, target):
    diffs = input - target

    return torch.sum(diffs).pow(2) / (torch.numel(diffs) ** 2)


def DiffLoss(input1, input2):
    bs = input1.shape[0]
    input1 = input1.view(bs, -1)
    input2 = input2.view(bs, -1)

    input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-20)

    input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-20)

    loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

    return loss
