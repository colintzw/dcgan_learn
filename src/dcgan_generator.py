from torch import nn


class DCGan_Generator(nn.Module):
    LATENT_DIM = 100
    CONV1_INPUT_SIZE = 4
    CONV1_INPUT_CHANNELS = 1024

    def __init__(self, device="mps"):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.LATENT_DIM,
                out_features=self.CONV1_INPUT_CHANNELS
                * self.CONV1_INPUT_SIZE
                * self.CONV1_INPUT_SIZE,
                bias=True,  # unsure if this is needed, just add
            )
        )
        conv_layer_parts = (
            self.conv_layer(
                in_channels=self.CONV1_INPUT_CHANNELS,
                out_channels=self.CONV1_INPUT_CHANNELS // 2,
                is_final_layer=False,
            )
            + self.conv_layer(
                in_channels=self.CONV1_INPUT_CHANNELS // 2,
                out_channels=self.CONV1_INPUT_CHANNELS // 4,
                is_final_layer=False,
            )
            + self.conv_layer(
                in_channels=self.CONV1_INPUT_CHANNELS // 4,
                out_channels=self.CONV1_INPUT_CHANNELS // 8,
                is_final_layer=False,
            )
            + self.conv_layer(
                in_channels=self.CONV1_INPUT_CHANNELS // 8,
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
            -1, self.CONV1_INPUT_CHANNELS, self.CONV1_INPUT_SIZE, self.CONV1_INPUT_SIZE
        )
        # conv blocks
        return self.conv_block(proj_and_reshape)

    @staticmethod
    def conv_layer(in_channels, out_channels, is_final_layer: bool = False):
        if is_final_layer:
            return [
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                ),
                nn.Tanh(),
            ]
        else:
            return [
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
            ]
