import torch
import torch.nn as nn

net_sequential = nn.Sequential(
        nn.Linear(in_features=3, out_features=6),
        nn.ReLU(),
        nn.Linear(in_features=6, out_features=3)
) 

print(net_sequential)

 # pipenv install --index https://download.pytorch.org/whl/ "torch==2.0.1+cu123" "torchvision==0.15.2+cu123" "torchaudio==2.0.2+cu123"