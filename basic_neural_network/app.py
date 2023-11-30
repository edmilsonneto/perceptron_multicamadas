
import torch
from torch import nn
from sklearn import datasets

from wine_classifier_net import WineClassifierNet

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    wine = datasets.load_wine()
    data = wine.data
    target = wine.target
    
    input_size = data.shape[1]
    hidden_size = 32
    out_size = len(wine.target_names)
    
    net = WineClassifierNet(input_size=input_size, hidden_size=hidden_size, out_size=out_size).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    
    xTns = torch.from_numpy(data).float()
    yTns = torch.from_numpy(target).long()
    
    xTns = xTns.to(device=device)
    yTns = yTns.to(device=device)
    
    pred = net(xTns)
    
    loss = criterion(pred, yTns)
    
    print(loss)

if __name__ == '__main__':
    main() 