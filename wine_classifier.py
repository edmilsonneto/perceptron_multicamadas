from torch import nn

class WineClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(WineClassifier, self).__init__()
        
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, X):
        feature = self.relu(self.hidden(X))
        output = self.softmax(self.out(feature))
        
        return output