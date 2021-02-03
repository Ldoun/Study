import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    def __init__(self,input_c,output_c):
        self. input_channel = input_c
        self.ouput_channel = output_c

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(self.input_channel, self.ouput_channel, (3,3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.ouput_channel),
            nn.Conv2d(self.ouput_channel, self.ouput_channel,kernel_size=(3,3),stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.ouput_channel)
        )

    def forward(self, x):
        y = self.layers(x)
        return y

class ConvolutionClassifier(nn.Module):
    def __init__(self,ouput_size):
        self.output_size = ouput_size

        super().__init__() 
        #32 -> 16 -> 8 -> 4-> 2 -> 1
        self.blocks = nn.Sequential(
            ConvolutionBlock(3,32),
            ConvolutionBlock(32,64),
            ConvolutionBlock(64,128),
            ConvolutionBlock(128,256),
            ConvolutionBlock(256,512),
        )

        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50,10),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        assert x.size(0) > 3
        z = self.blocks(x)
        y = self.layers(z.squeeze())
        return y
        
