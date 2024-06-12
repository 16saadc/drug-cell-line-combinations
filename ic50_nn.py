import torch
import torch.nn as nn

class IC50Net(nn.Module):
    def __init__(self, input_dim):
        super(IC50Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 128)
        # self.bn3 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(128, 64)
        # self.bn4 = nn.BatchNorm1d(64)
        self.dp4 = nn.Dropout(0.3)

        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.dp1(self.fc1(x)))
        x = torch.relu(self.dp2(self.fc2(x)))
        x = torch.relu(self.dp3(self.fc3(x)))
        x = torch.relu(self.dp4(self.fc4(x)))
        x = self.fc5(x)
        return x
