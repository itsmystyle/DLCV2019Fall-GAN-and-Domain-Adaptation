import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


def SI_MSELoss(input, target):
    errors = input - target

    return torch.sum(errors).pow(2) / (torch.numel(errors) ** 2)


def DiffLoss(shared_input, private_input):
    bs = shared_input.shape[0]
    shared_input = shared_input.view(bs, -1)
    private_input = private_input.view(bs, -1)

    shared_l2_norm = torch.norm(shared_input, p=2, dim=1, keepdim=True).detach()
    shared_l2 = shared_input.div(shared_l2_norm.expand_as(shared_input) + 1e-20)

    private_l2_norm = torch.norm(private_input, p=2, dim=1, keepdim=True).detach()
    private_l2 = private_input.div(private_l2_norm.expand_as(private_input) + 1e-20)

    loss = torch.mean((shared_l2.t().mm(private_l2)).pow(2))

    return loss
