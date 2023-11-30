
import torch
from sklearn import datasets
from torch import nn

from diabetes_classifier_net import DiabetesClassifierNet

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    diabetes = datasets.load_diabetes()
    data = diabetes.data
    target = diabetes.target
    
    # print(data.shape, target.shape)
    
    # print(data[14])
    # print(target[14])
        
    input_size = data.shape[1]
    hidden_size = 32
    out_size = 1
    
    net = DiabetesClassifierNet(input_size=input_size, hidden_size=hidden_size, out_size=out_size).to(device=device)
    
    criterion = nn.MSELoss().to(device=device)
    
    xTns = torch.from_numpy(data).float().to(device=device)
    yTns = torch.from_numpy(target).long().to(device=device)
    
    # print(xTns.shape, yTns.shape)
    
    pred = net(xTns)
    
    loss = criterion(pred.squeeze(), yTns)
    
    print(loss.data)

if __name__ == '__main__':
    main() 