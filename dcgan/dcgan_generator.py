from dcgan_consts import (
    CONV_GEN_CHANNELS,
    CONV_GEN_INPUT_SIZE,
    CONV_KERNEL_SIZE,
    CONV_PADDING,
)
from torch import nn


class Generator(nn.Module):
    LATENT_DIM = 100

    def __init__(self, device="mps"):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.LATENT_DIM,
                out_features=CONV_GEN_CHANNELS[0]
                * CONV_GEN_INPUT_SIZE
                * CONV_GEN_INPUT_SIZE,
                bias=True,  # unsure if this is needed, just add
            )
        )
        conv_layer_parts = (
            self.conv_layer(
                in_channels=CONV_GEN_CHANNELS[0],
                out_channels=CONV_GEN_CHANNELS[1],
                is_final_layer=False,
            )
            + self.conv_layer(
                in_channels=CONV_GEN_CHANNELS[1],
                out_channels=CONV_GEN_CHANNELS[2],
                is_final_layer=False,
            )
            + self.conv_layer(
                in_channels=CONV_GEN_CHANNELS[2],
                out_channels=CONV_GEN_CHANNELS[3],
                is_final_layer=False,
            )
            + self.conv_layer(
                in_channels=CONV_GEN_CHANNELS[3],
                out_channels=3,
                is_final_layer=True,
            )
        )
        self.conv_block = nn.Sequential(*conv_layer_parts)
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

    def forward(self, z):
        # proj
        proj_and_reshape = self.linear(z)
        # reshape with view
        proj_and_reshape = proj_and_reshape.view(
            -1, CONV_GEN_CHANNELS[0], CONV_GEN_INPUT_SIZE, CONV_GEN_INPUT_SIZE
        )
        # conv blocks
        return self.conv_block(proj_and_reshape)

    @staticmethod
    def conv_layer(in_channels, out_channels, is_final_layer: bool = False):
        conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=CONV_KERNEL_SIZE,
            stride=2,
            padding=CONV_PADDING,
        )
        if is_final_layer:
            return [
                conv_transpose,
                nn.Tanh(),
            ]
        else:
            return [
                conv_transpose,
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
            ]
