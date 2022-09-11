import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=2, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, l => [b, ch_out, l]
            self.extra = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=stride)

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # short cut.
        # extra module: [b, ch_in, l] => [b, ch_out, l]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_class):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=2)
        # followed 5 blocks
        # [b, 8, l] => [b, 16, l]
        self.blk1 = ResBlk(8, 16, stride=1)
        # [b, 16, l] => [b, 32, l]
        self.blk2 = ResBlk(16, 32, stride=1)
        # [b, 32, l] => [b, 64, l]
        self.blk3 = ResBlk(32, 64, stride=1)
        # [b, 64, l] => [b, 128, l]
        self.blk4 = ResBlk(64, 128, stride=1)
        # [b, 128, l] => [b, 256, l]
        self.blk5 = ResBlk(128, 256, stride=1)
        self.outlayer = nn.Linear(2816, num_class)
    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        # [b, 1, l] => [b, 256, l]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


def main():
    blk = ResBlk(8, 16)
    tmp = torch.randn(1, 8, 12)
    out = blk(tmp)
    print('block:', out.shape)

    model = ResNet(3)
    tmp = torch.randn(1, 1, 12)
    out = model(tmp)
    print('resnet:', out)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()
