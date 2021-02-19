import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3,3), padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3,3), stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        y = self.layers(x)
        return y
    
class ConvolutionClassifier(nn.Module):
    def __init__(self, output_size):
        
        super().__init__()

        self.blocks = nn.Sequential(
            ConvolutionBlock(1,32),
            ConvolutionBlock(32,64),
            ConvolutionBlock(64,128),
            ConvolutionBlock(128,256),
            ConvolutionBlock(256,512),
        )

        self.layers = nn.Sequential(
            nn.Linear(512,50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50,10),
            nn.Softmax(dim = -1)
        )

    def forward(self, x):
        assert x.dim() > 2  

        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))

        z = self.blocks(x)

        y = self.layers(z.squeeze())

        return y