import torch.nn as nn

net_sequential = nn.Sequential(
        nn.Linear(in_features=3, out_features=6),
        nn.ReLU(),
        nn.Linear(in_features=6, out_features=3)
) 

print(net_sequential)