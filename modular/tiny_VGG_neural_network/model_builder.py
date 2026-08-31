"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn

class TinyVGGConvolutionalModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units):
        """
            This is the VGG Convolutional model with 2 convolutional layers followed by a max pool layer and non-linear relu function, this layer is repeated two times
            
            Args:
                input_shape: number of input channels (color channels)
                output_shape: number of output classes
                hidden_units: number of hidden units
            
            Output:
                A Convolutional Neural Network model that can be used for image classification.
                Model returns raw logits for `output_shape` classes.
        """
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16*16, out_features=output_shape)
        )
        
        
    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))