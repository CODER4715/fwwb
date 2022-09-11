import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

import netron


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=2, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=stride)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_class):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, kernel_size=2)
        self.blk1 = ResBlk(8, 16, stride=1)
        self.blk2 = ResBlk(16, 32, stride=1)
        self.blk3 = ResBlk(32, 64, stride=1)
        self.blk4 = ResBlk(64, 128, stride=1)
        self.blk5 = ResBlk(128, 256, stride=1)

        self.outlayer = nn.Linear(2816, num_class)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


input = torch.rand(1, 1, 12)
model = ResNet(3)
model.load_state_dict(torch.load('best.pt'))
output = model(input)

onnx_path = "model.onnx"
torch.onnx.export(model, input, onnx_path, training=True)

netron.start(onnx_path, 5555)