import torch
import torch.nn as nn
from typing import Literal

# Used https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
# as a reference


class DoubleConv(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, n_dimensions=2,
            activation="LeakyReLU"):
        super(DoubleConv, self).__init__()

        def get_conv(in_channels, out_channels):
            # 1-dimensional convolution is not supported
            if n_dimensions == 3:
                return nn.Conv3d(
                    in_channels, out_channels, kernel_size=(3, 3, 1),
                    padding=(1, 1, 0))
            elif n_dimensions == 2:
                return nn.Conv2d(
                    in_channels, out_channels, kernel_size=(3, 3),
                    padding=(1, 1))
            else:
                raise ValueError(
                    f"Unsupported number of dimensions: {n_dimensions}")

        def get_activation():
            if activation == "LeakyReLU":
                return nn.LeakyReLU(negative_slope=0.01, inplace=True)
            elif activation == "ReLU":
                return nn.ReLU(inplace=True)
            else:
                raise ValueError(
                    f"Unsupported activation function: {activation}")

        self.conv_block = nn.Sequential(
            get_conv(in_channels, out_channels), get_activation(),
            get_conv(out_channels, out_channels), get_activation())

    def forward(self, x: torch.Tensor):
        return self.conv_block(x)


class EncodeBlock2d(nn.Module):
    def __init__(
            self, in_channels: int,
            activation="LeakyReLU",
            downsampling_kernel=(2, 2), downsampling_mode="max"):
        super(EncodeBlock2d, self).__init__()

        len = downsampling_kernel[0]  # Assume kernel has shape (len, len)
        assert list(downsampling_kernel) == [len, len], \
            f"Expected a flat square kernel like {(len, len)}, " + \
            f"got {downsampling_kernel}"
        # Stride 2x2 to halve each side
        stride = (2, 2)
        # Padding (len-1) // 2 to exactly halve each side
        padding = ((len-1)//2, (len-1)//2)
        if downsampling_mode == "max_pool":
            self.pool = nn.MaxPool2d(
                kernel_size=downsampling_kernel, stride=stride,
                padding=padding)
        elif downsampling_mode == "avg_pool":
            self.pool = nn.AvgPool2d(
                kernel_size=downsampling_kernel, stride=stride,
                padding=padding)
        else:
            raise ValueError(f"Unknown pooling method: {downsampling_mode}")

        self.double_conv = DoubleConv(
            in_channels, in_channels * 2, n_dimensions=2,
            activation=activation)

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class DecodeBlock2d(nn.Module):
    def __init__(
            self, in_channels: int,
            activation="LeakyReLU",
            upsampling_kernel=(2, 2),
            upsampling_mode: Literal[
                "linear_interpolation"] = "linear_interpolation"):
        super(DecodeBlock2d, self).__init__()

        if upsampling_mode == "linear_interpolation":
            self.upsampling = nn.Sequential(
                nn.Upsample(
                    # Assume the shape is (Nx, Ny) where
                    # Nx is the image width and Ny is the image height.
                    scale_factor=(2, 2),
                    # Is there any difference if align_corners is True or
                    # False? Preserving symmetry?
                    mode='bilinear',
                    # align_corners=False),
                    align_corners=True),
                # 1x1 convolution to reduce the number of channels while
                # keeping the size the same
                nn.Conv2d(
                    in_channels, in_channels // 2,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0)
                )
            )
            self.up_conv = nn.Conv2d(
                in_channels, in_channels // 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                # padding_mode="zeros"
            )
        # elif upsampling_mode == "transposed_convolution":
        #     len = upsampling_kernel[0]  # Assume kernel has shape (len, len)
        #     # assert upsampling_kernel == (len, len), \
        #     #     f"Expected a flat square kernel like {(len, len)}, " + \
        #     #     f"got {upsampling_kernel}"
        #     # Stride 2x2 to double the length of each side
        #     stride = (2, 2)
        #     # Padding (len-1) // 2 to exactly double each side
        #     padding = ((len-1)//2, (len-1)//2)
        #     self.upsampling = nn.ConvTranspose2d(
        #         in_channels, in_channels // 2,
        #         kernel_size=upsampling_kernel, stride=stride, padding=padding,
        #         output_padding=padding  # TODO: Should this be same as padding?
        #     )
        else:
            raise ValueError(
                f"Unsupported upsampling method: {upsampling_mode}")

        self.double_conv = DoubleConv(
            in_channels, in_channels // 2, n_dimensions=2,
            activation=activation)

    def forward(self, x: torch.Tensor, x_encoder_output: torch.Tensor):
        x = self.upsampling(x)
        # x_encoder_output_shape = x_encoder_output.shape
        # x = nn.functional.interpolate(
        #     input=x, size=x_encoder_output_shape[2:], mode='bilinear',
        #     align_corners=False)
        # x = self.up_conv(x)

        # print(f"x_encoder_output.shape = {x_encoder_output.shape}")
        # print(f"x.shape = {x.shape}")

        # skip-connection. # No cropping since the size is the same.
        # x = torch.cat([x, x_encoder_output], dim=1)
        x = torch.cat([x_encoder_output, x], dim=1)
        x = self.double_conv(x)
        return x


class UNet2d(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=2, init_filters=32, n_blocks=3,
            activation="LeakyReLU",
            downsampling_kernel=(2, 2), downsampling_mode="max_pool",
            upsampling_kernel=(2, 2), upsampling_mode="linear_interpolation",
            # padding_mode="zeros",
    ):
        """
        Assume that input is 4D tensor of shape (batch_size, channels, N1, N2).
        Usually batch_size = 1 (one image per batch),
        channels = 1 (greyscale), N1 = N2 (square image).
        ("channels" is equivalent to the number of filters or features)

        Example from U-Net paper (2015,
        Olaf Ronneberger https://arxiv.org/abs/1505.04597):
            - in_channels = 1
            - out_channels = 2
            - init_filters = 64
            - n_blocks = 4
            - pooling: max pooling 2x2
            - pool padding = 0
                - 0 padding will reduce the size of the "image" by 2 in
                each dimension after each convolution. The skip-connection
                will have to crop the encoder's output to match the
                decoder's input.
            - upsampling kernel: 2x2
            - up_mode: ? (linear interpolation or transposed convolution)

        >>> unet = UNet2d(init_filters=8, n_blocks=2)
        >>> x = torch.randn(1, 1, 16, 16) # 4D, normal use case when training
        >>> print(x.shape)
        torch.Size([1, 1, 16, 16])
        >>> y = unet(x)
        >>> print(y.shape)
        torch.Size([1, 2, 16, 16])

        """

        # print(f"type(self) = {type(self)}")
        # print(f"isinstance(self, UNet2d) = {isinstance(self, UNet2d)}")
        super(UNet2d, self).__init__()
        # super(nn.Module, self).__init__()
        print("Using my own UNet2d class")

        self.c0x0 = DoubleConv(  # TODO: Find a better variable name
            in_channels=in_channels,
            out_channels=init_filters,
            activation=activation,
            n_dimensions=2,
        )
        self.encoder = nn.ModuleList([
            EncodeBlock2d(
                in_channels=init_filters * 2**i,
                activation=activation,
                downsampling_kernel=downsampling_kernel,
                downsampling_mode=downsampling_mode
            ) for i in range(n_blocks)
        ])
        self.decoder = nn.ModuleList([
            DecodeBlock2d(
                in_channels=init_filters * 2**(n_blocks-i),
                activation=activation,
                upsampling_kernel=upsampling_kernel,
                upsampling_mode=upsampling_mode
            ) for i in range(n_blocks)
        ])
        # 1x1x1 convo
        self.c1x1 = nn.Conv2d(  # TODO: Find a better name
            in_channels=init_filters,
            out_channels=out_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

    def forward(self, x: torch.Tensor):
        # Assume that x is 5D tensor of shape (batch_size, channels, Nx, Ny, Nt)
        # where Nx is the image width and Ny is the image height.
        # Aslo assume that batch_size = 1, channels = 1,
        # Nx = Ny (square image), Nt = 1 (static image).
        # NOTE: The convention used in pytorch documentation is
        # (batch_size, channels, Nt, Ny, Nx).
        # assert len(x.size()) == 5, f"Expected 5D tensor, got {x.size()}"
        # batch_size, channels, Nx, Ny, Nt = x.size()
        # batch_size, channels, Nx, Ny= x.size()
        # assert channels == 1, f"Expected 1 channel, got {channels}"
        # # TODO: Allow multiple channels (colour images)
        # assert Nx == Ny, f"Expected square image, got ({Nx}, {Ny})"
        # # TODO: Allow non-square images
        # assert Nt == 1, f"Expected 1 time step, got {Nt}"
        # # TODO: Allow multiple time steps (dynamic images, video)
        # assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
        # # TODO: Might train with larger batch size

        # print(f"x.size() = {x.size()}")

        Nx, Ny = x.size()[-2], x.size()[-1]
        n_blocks = len(self.encoder)
        assert Nx >= 2**n_blocks, \
            f"Expected width (Nx) of at least {2**n_blocks}, got {Nx}"
        assert Ny >= 2**n_blocks, \
            f"Expected height (Ny) of at least {2**n_blocks}, got {Ny}"

        x = self.c0x0(x)

        encoder_outputs = []
        for i, enc_block in enumerate(self.encoder):
            encoder_outputs.append(x)
            x = enc_block(x)
        for i, dec_block in enumerate(self.decoder):
            x = dec_block(x, encoder_outputs[-i-1])  # skip-connection inside

        x = self.c1x1(x)

        for enc_output in encoder_outputs:
            del enc_output
        del encoder_outputs

        return x
