import torch.nn as nn

class Identity(nn.Module):
    def forward(self, x):
        return x
