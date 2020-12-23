import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入图片大小为 100x40
        self.conv1 = nn.Conv2d(3, 6,kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16,kernel_size=3)
        self.fc1 = nn.Linear(16 * 8 * 23, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20,2)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16*8*23)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x