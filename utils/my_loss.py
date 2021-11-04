import torch

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, target):

        return torch.norm((output - target),'fro') / torch.norm(target,'fro')

