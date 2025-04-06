from typing import List

from dcgan_consts import (
    CONV_GEN_CHANNELS,
    CONV_GEN_INPUT_SIZE,
    CONV_KERNEL_SIZE,
    CONV_PADDING,
)
from torch import nn


class Discriminator(nn.Module):
    IMAGE_SIZE = 64
    DCONV1_INPUT_CHANNELS = 128

    def __init__(self, device="mps"):
        super().__init__()
        conv_layer_parts = (
            self.conv_layer(3, CONV_GEN_CHANNELS[-1], is_first_layer=True)
            + self.conv_layer(
                CONV_GEN_CHANNELS[-1],
                CONV_GEN_CHANNELS[-2],
                is_first_layer=False,
            )
            + self.conv_layer(
                CONV_GEN_CHANNELS[-2],
                CONV_GEN_CHANNELS[-3],
                is_first_layer=False,
            )
            + self.conv_layer(
                CONV_GEN_CHANNELS[-3],
                CONV_GEN_CHANNELS[0],
                is_first_layer=False,
            )
        )
        self.conv_block = nn.Sequential(*conv_layer_parts)
        self.final_prediction = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                CONV_GEN_CHANNELS[0] * CONV_GEN_INPUT_SIZE * CONV_GEN_INPUT_SIZE, 1
            ),
            nn.Sigmoid(),
        )

        self.to(device)  # try deepseek suggestion to move model to device
        self._initialize_weights()

    def _initialize_weights(self):
        """Used to align with the DCGAN paper of weight initialization"""
        for m in self.modules():  # Iterates through all layers
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)  # Main weights
                if hasattr(m, "bias") and m.bias is not None:
                    # Zero the biases, Don't want this to be crazy big or small.
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_block(x)
        return self.final_prediction(out)

    @staticmethod
    def conv_layer(in_channels, out_channels, is_first_layer=False) -> List:
        conv_parts = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=CONV_KERNEL_SIZE,
                padding=CONV_PADDING,
                stride=2,
            )
        ]
        if not is_first_layer:
            conv_parts.append(nn.BatchNorm2d(num_features=out_channels))

        conv_parts.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        return conv_parts
