import torch
import torch.nn as nn
import torch.nn.functional as F

class FontClassifierCNN(nn.Module):
    def __init__(self):
        super(FontClassifierCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)  
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1) 

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)  

    def forward(self, x):
        # Conv Layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # FCs
        x = x.view(-1, 256 * x.size(2) * x.size(3)) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
