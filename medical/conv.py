import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.pad = kernel_size // 2 
        
        self.conv1 = nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride,
                        padding=self.pad)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
                        in_channels=out_channels, 
                        out_channels=out_channels, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding=self.pad)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(
                                in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=1, 
                                stride=2
                                    )
        else:
            self.downsample = None
    
    def forward(self, x):
        
        res = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        if self.downsample:
            res = self.downsample(res)
        x = x + res
        
        x = self.act2(x)
        
        return x 

class Block(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Block, self).__init__()
        stride = 1 if in_channels == out_channels else 2
        self.residual1 = ResidualBlock(in_channels, out_channels, 3, stride)
        self.residual2 = ResidualBlock(out_channels, out_channels, 3, 1)
    
    def forward(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels) -> None:
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_dim, out_channels)


class CondNet1D(nn.Module):

    def __init__(self, in_channels, num_classes) -> None:
        super(CondNet1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.block1 = Block(in_channels=64, out_channels=64)
        self.block2 = Block(in_channels=64, out_channels=128)
        self.block3 = Block(in_channels=128, out_channels=256)
        self.block4 = Block(in_channels=256, out_channels=512)

        self.pool = nn.AdaptiveAvgPool1d((1))

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
