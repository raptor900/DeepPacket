import torch
from torch import nn


class DeepPacketCNN(nn.Module):

    def __init__(self, n_classes):
        super(DeepPacketCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 200, 5, 2, 0)
        self.bn1 = nn.BatchNorm1d(200)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(1, 100, 4, 1, 0)
        self.bn2 = nn.BatchNorm1d(100)

        self.pool = nn.AvgPool1d(2)

        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Sequential(nn.Linear(100 * 372, 600),
                                 nn.Dropout(p=0.25), nn.ReLU(True)
                                 )
        self.fc2 = nn.Sequential(nn.Linear(600, 500),
                                 nn.Dropout(p=0.25), nn.ReLU(True)
                                 )
        self.fc3 = nn.Sequential(nn.Linear(500, 400),
                                 nn.Dropout(p=0.25), nn.ReLU(True)
                                 )
        self.fc4 = nn.Sequential(nn.Linear(400, 300),
                                 nn.Dropout(p=0.25), nn.ReLU(True)
                                 )
        self.fc5 = nn.Sequential(nn.Linear(300, 200),
                                 nn.Dropout(p=0.25), nn.ReLU(True)
                                 )
        self.fc6 = nn.Sequential(nn.Linear(200, 100),
                                 nn.Dropout(p=0.25), nn.ReLU(True)
                                 )
        self.fc7 = nn.Sequential(nn.Linear(100, 50),
                                 nn.Dropout(p=0.25), nn.ReLU(True)
                                 )

        self.fc_out = nn.Linear(50, n_classes)

        self.lsm = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x).view(-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)

        x = self.fc_out(x)

        y = self.lsm(x)

        return y



class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(AutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(nn.Linear(input_size, output_size),
                                          nn.Dropout(p=0.05),
                                          nn.ReLU(True)
                                          )

        self.backward_pass = nn.Linear(input_size, output_size)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        x = x.detach()

        y = self.forward_pass(x)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, x.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach()


class StackedAutoEncoder(nn.Module):
    def __init__(self, n_classes):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = AutoEncoder(1500, 400)
        self.ae2 = AutoEncoder(400, 300)
        self.ae3 = AutoEncoder(300, 200)
        self.ae4 = AutoEncoder(200, 100)
        self.ae5 = AutoEncoder(100, 50)

        self.fc_out = nn.Linear(50, n_classes)

        self.lsm = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.ae1(x)
        x = self.ae2(x)
        x = self.ae3(x)
        x = self.ae4(x)
        x = self.ae5(x)

        y = self.fc_out(x)

        y = self.lsm(y)

        return y
