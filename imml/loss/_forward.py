import torch.nn as nn

class ForwardLoss(nn.Module):
    
    def forward(self, input, target=None):
        return input